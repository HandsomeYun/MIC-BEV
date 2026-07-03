import os.path as osp
import cv2
import numpy as np
import torch
import mmcv
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadBEVSegFromFile:
    """Read the ground-truth BEV semantic label map.

    Expects `results['map_path']` (absolute or relative) that points to
    a .npy, .png or .tif file containing a single‐channel label image.
    The label tensor is stored in `results['bev_seg_label']`.
    """
    def __init__(self, bev_size=None, file_client_args=None):
        self.bev_size = bev_size
        self.file_client = mmcv.FileClient.infer_client(file_client_args)

    def __call__(self, results):
        seg_path = results['map_path']
        if not osp.isabs(seg_path):
            seg_path = osp.join(results['data_root'], seg_path)

        if seg_path.endswith('.npy'):
            seg = np.load(seg_path)
        else:
            seg = mmcv.imread(seg_path, flag='unchanged')
        if self.bev_size is not None and tuple(seg.shape[:2]) != tuple(self.bev_size):
            target_h, target_w = self.bev_size
            seg = cv2.resize(
                seg,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST)

        bev_seg_label = torch.from_numpy(seg).long()
        results['bev_seg_label'] = bev_seg_label
    
        return results
