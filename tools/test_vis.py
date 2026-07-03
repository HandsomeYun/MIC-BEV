
import argparse
import json
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.micbev.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from torch.utils.data import Subset
import numpy as np
import cv2

# Register visualization-enabled dataset variants for this entry point only.
import projects.mmdet3d_plugin.datasets.m2i_dataset_vis  # noqa: F401
import projects.mmdet3d_plugin.datasets.roscenes_dataset_vis  # noqa: F401


def count_trainable_and_total_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='first dataset index to run during quick inference tests')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='number of dataset samples to run during quick inference tests')
    parser.add_argument(
        '--bbox-only',
        action='store_true',
        help='skip test-time map/robustness preprocessing and disable '
        'segmentation heads for faster bbox-only inference')
    parser.add_argument(
        '--fast-cuda',
        action='store_true',
        help='enable cudnn benchmark and TF32 math for faster CUDA inference')
    parser.add_argument(
        '--fp16-test',
        action='store_true',
        help='try MMCV fp16 inference without changing the model architecture')
    parser.add_argument(
        '--pin-memory',
        action='store_true',
        help='enable DataLoader pinned host memory for faster CPU-to-GPU copies')
    parser.add_argument(
        '--prefetch-factor',
        type=int,
        default=None,
        help='DataLoader prefetch factor per worker')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--save-map', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.bbox_only:
        if isinstance(cfg.data.test, dict) and 'pipeline' in cfg.data.test:
            skip_types = {'LoadBEVSegFromFile', 'RandomMaskMultiView'}
            cfg.data.test.pipeline = [
                step for step in cfg.data.test.pipeline
                if step.get('type') not in skip_types
            ]
        if hasattr(cfg.model, 'pts_bbox_head'):
            cfg.model.pts_bbox_head.with_seg = False
        print('[INFO] bbox-only mode: disabled test BEV seg/mask pipeline '
              'steps and segmentation heads.')
    if args.fp16_test:
        cfg.fp16 = cfg.get('fp16', dict(loss_scale='dynamic'))
        print('[INFO] fp16-test mode: enabled MMCV fp16 wrapper.')
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.fast_cuda:
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print('[INFO] fast-cuda mode: enabled cudnn benchmark and TF32.')
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)
        
    test_pipeline = cfg.data.test.get('pipeline', []) if isinstance(cfg.data.test, dict) else []
    random_mask_cfg = next(
        (step for step in test_pipeline if step.get('type') == 'RandomMaskMultiView'),
        None)
    dataset = build_dataset(cfg.data.test)
    if args.show_dir is not None:
        dataset.vis_out_dir = args.show_dir
    if random_mask_cfg is not None:
        dataset.visual_corruption_cfg = {
            key: random_mask_cfg[key]
            for key in (
                'mask_prob',
                'blur_kernel_size',
                'blur_kernel_size_range',
                'blur_sigma',
                'blur_levels',
                'blur_level_probs',
                'mask_ratio',
                'blur_ratio',
            )
            if key in random_mask_cfg
        }
    if args.start or args.num_samples is not None:
        if args.start < 0:
            raise ValueError('--start must be non-negative')
        end = None if args.num_samples is None else args.start + args.num_samples
        if hasattr(dataset.data_infos, '_indexing'):
            dataset.data_infos._indexing = dataset.data_infos._indexing[args.start:end]
        else:
            dataset.data_infos = dataset.data_infos[args.start:end]
        if hasattr(dataset, 'flag'):
            dataset.flag = dataset.flag[args.start:end]
        if hasattr(dataset, 'ego_position_map'):
            dataset.ego_position_map = {
                info["token"]: tuple(info["ego2global_translation"][:2])
                for info in dataset.data_infos
            }
        dataset.visualization_start_index = args.start
        print(f'[INFO] Running {len(dataset.data_infos)} samples '
              f'from dataset index {args.start}.')
    if args.save_map:
        num_to_run = 25000
        # # #===========M2I===============
        # dataset.data_infos = dataset.data_infos[:num_to_run]
        # #===========Roscenes===============
        orig_evaluate = dataset.evaluate
        dataset = Subset(dataset, list(range(num_to_run)))
        dataset.evaluate = orig_evaluate
        
    dataloader_kwargs = {}
    if args.prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = args.prefetch_factor

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        pin_memory=args.pin_memory,
        **dataloader_kwargs,
    )
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    dump_camera_weights = False
    model.dump_camera_weights = dump_camera_weights
    if hasattr(model, 'pts_bbox_head'):
        model.pts_bbox_head.dump_camera_weights = dump_camera_weights
    if dump_camera_weights and hasattr(model, 'module'):
        model.module.dump_camera_weights = dump_camera_weights
        if hasattr(model.module, 'pts_bbox_head'):
            model.module.pts_bbox_head.dump_camera_weights = dump_camera_weights
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    total_params, trainable_params = count_trainable_and_total_params(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_start = time.time()

    if not distributed:
        # assert False
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        if args.save_map:
            #==========Save Map================
            CLASS_ID = {
                "background": 0,
                "driving":    1,
                "sidewalk":   2,
                "crosswalk":  3,
                "shoulder":   4,
                "border":     5,
                "parking":    6,
            }

            # a prettier palette indexed by class ID
            palette = np.array([
                [255, 255, 255],   # 0 background – white
                [99,165,112],   # 1 driving   –  green
                [193, 182, 255],   # 2 sidewalk  – dark pink
                [212,154,158],   # 3 crosswalk – pink
                [116,60,56],   # 4 shoulder  – coral red
                [33,64,43],   # 5 border    – orchid purple
                [  0, 188, 212],   # 6 parking   – teal
            ], dtype=np.uint8)
                    
             # create separate folders
            os.makedirs('./seg_preds/gt',   exist_ok=True)
            os.makedirs('./seg_preds/pred', exist_ok=True)
            try:
                indices = dataset.indices
                base_dataset = dataset.dataset
            except AttributeError:
                indices = list(range(len(dataset)))
                base_dataset = dataset

            for local_idx, out in enumerate(outputs):
                tok = f"{local_idx:03d}"

                # prediction map + legend
                pred = out['semantic_map']  # H×W numpy array
                color_pred = palette[pred]
                # ground-truth map + legend
                orig_idx = indices[local_idx]
                info = base_dataset.data_infos[orig_idx]
                gt = np.load(info['map_path'])
                color_gt = palette[gt]

                # build legend panel below
                legend_h = 30 * len(CLASS_ID)
                legend = np.full((legend_h, color_gt.shape[1], 3), 50, dtype=np.uint8)
                y = 25
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness  = 1
                for cls_name, cls_id in CLASS_ID.items():
                    color = tuple(int(c) for c in palette[cls_id].tolist())
                    cv2.rectangle(legend, (10, y-20), (30, y), color, -1)
                    cv2.putText(legend, f"{cls_name} ({cls_id})", (40, y),
                                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    y += 30

                # stitch and save prediction
                canvas_pred = np.vstack([color_pred, legend])
                out_pred = cv2.resize(canvas_pred,
                                    (canvas_pred.shape[1]*2, canvas_pred.shape[0]*2),
                                    interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'./seg_preds/pred/{tok}.png', out_pred)

                # stitch and save ground truth
                canvas_gt = np.vstack([color_gt, legend])
                out_gt = cv2.resize(canvas_gt,
                                    (canvas_gt.shape[1]*2, canvas_gt.shape[0]*2),
                                    interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(f'./seg_preds/gt/{tok}.png', out_gt)
            #================================
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    infer_time = time.time() - infer_start

    rank, _ = get_dist_info()
    if rank == 0:
        num_outputs = len(outputs)
        fps = num_outputs / infer_time if infer_time > 0 else float('inf')
        latency_ms = infer_time * 1000.0 / num_outputs if num_outputs else float('inf')
        peak_mem_gb = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0)
        print('[INFO] Runtime summary:')
        print(f'[INFO]   Params: {total_params / 1e6:.2f} M '
              f'(trainable {trainable_params / 1e6:.2f} M)')
        print(f'[INFO]   Peak CUDA memory: {peak_mem_gb:.2f} GB')
        print(f'[INFO]   Latency: {latency_ms:.2f} ms/sample')
        print(f'[INFO] Inference time: {infer_time:.2f}s '
              f'for {num_outputs} samples ({fps:.2f} FPS)')

        if args.out:
            print(f'\nwriting results to {args.out}')
            # assert False
            # mmcv.dump(outputs['bbox_results'], args.out)
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            eval_dict = dataset.evaluate(outputs, **eval_kwargs)
            print(eval_dict)

if __name__ == '__main__':
    main()