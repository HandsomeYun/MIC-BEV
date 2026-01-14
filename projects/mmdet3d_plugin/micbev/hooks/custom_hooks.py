from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

@HOOKS.register_module()
class EpochTrackerHook(Hook):
    def __init__(self, **kwargs):  # Accept kwargs to avoid build_from_cfg error
        super().__init__()

    def before_train_epoch(self, runner):
        # Set the epoch value in the dataset so it can be accessed in __getitem__
        runner.data_loader.dataset.epoch = runner.epoch

