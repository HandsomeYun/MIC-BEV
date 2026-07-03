
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp
import re

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class SaveBestTopKMixin:
    def _init_save_best_top_k(self, save_best_top_k):
        if save_best_top_k is None:
            save_best_top_k = 1
        if save_best_top_k <= 0:
            raise ValueError(
                f'save_best_top_k must be positive, got {save_best_top_k}.')
        self.save_best_top_k = save_best_top_k

    def _checkpoint_tag(self, runner):
        if self.by_epoch:
            return f'epoch_{runner.epoch + 1}', 'epoch', runner.epoch + 1
        return f'iter_{runner.iter + 1}', 'iter', runner.iter + 1

    def _score_sort_reverse(self):
        return self.rule == 'greater'

    def _sanitize_key_indicator(self):
        return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(self.key_indicator))

    def _save_ckpt(self, runner, key_score):
        if self.save_best_top_k == 1:
            return super()._save_ckpt(runner, key_score)

        current, cur_type, cur_time = self._checkpoint_tag(runner)
        if runner.meta is None:
            runner.meta = {}
        hook_msgs = runner.meta.setdefault('hook_msgs', {})
        best_ckpts = hook_msgs.get('best_ckpts', [])
        if not isinstance(best_ckpts, list):
            best_ckpts = []

        best_ckpts = [
            item for item in best_ckpts
            if isinstance(item, dict) and 'score' in item and 'path' in item
        ]
        best_ckpts.sort(key=lambda item: item['score'],
                        reverse=self._score_sort_reverse())

        if len(best_ckpts) >= self.save_best_top_k:
            worst_score = best_ckpts[-1]['score']
            if not self.compare_func(key_score, worst_score):
                runner.logger.info(
                    f'{self.key_indicator}={key_score:0.4f} at '
                    f'{cur_time} {cur_type} did not enter the best '
                    f'{self.save_best_top_k} checkpoints.')
                return

        best_ckpt_name = (
            f'best_{self._sanitize_key_indicator()}_{current}.pth')
        best_ckpt_path = self.file_client.join_path(self.out_dir,
                                                   best_ckpt_name)
        best_ckpts.append(
            dict(score=float(key_score), path=best_ckpt_path,
                 step=cur_time, step_type=cur_type))
        best_ckpts.sort(key=lambda item: item['score'],
                        reverse=self._score_sort_reverse())

        runner.save_checkpoint(
            self.out_dir, best_ckpt_name, create_symlink=False)
        runner.logger.info(
            f'Checkpoint saved as {best_ckpt_name} with '
            f'{self.key_indicator}={key_score:0.4f}.')

        removed_ckpts = best_ckpts[self.save_best_top_k:]
        best_ckpts = best_ckpts[:self.save_best_top_k]
        for removed in removed_ckpts:
            removed_path = removed['path']
            if (removed_path != best_ckpt_path
                    and self.file_client.isfile(removed_path)):
                self.file_client.remove(removed_path)
                runner.logger.info(
                    f'The out-of-top-{self.save_best_top_k} checkpoint '
                    f'{removed_path} was removed.')

        hook_msgs['best_ckpts'] = best_ckpts
        hook_msgs['best_score'] = best_ckpts[0]['score']
        hook_msgs['best_ckpt'] = best_ckpts[0]['path']
        self.best_ckpt_path = best_ckpts[0]['path']
        runner.logger.info(
            f'Best {self.key_indicator} is {best_ckpts[0]["score"]:0.4f} '
            f'at {best_ckpts[0]["step"]} {best_ckpts[0]["step_type"]}; '
            f'keeping top {len(best_ckpts)} checkpoints.')


class CustomEvalHook(SaveBestTopKMixin, BaseEvalHook):

    def __init__(self, *args, save_best_top_k=1, **kwargs):
        self._init_save_best_top_k(save_best_top_k)
        super(CustomEvalHook, self).__init__(*args, **kwargs)


class CustomDistEvalHook(SaveBestTopKMixin, BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, save_best_top_k=1,
                 **kwargs):
        self._init_save_best_top_k(save_best_top_k)
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.mmdet3d_plugin.micbev.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
  
