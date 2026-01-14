import cv2
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
import os

@PIPELINES.register_module()
class GetBEVObjLabel:
    """Create BEV‐object occupancy mask from gt_bboxes_3d and gt_labels_3d."""
    def __init__(self, bev_h, bev_w, pc_range):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.y_min, _, self.x_max, self.y_max, _ = pc_range
        self.real_w = self.x_max - self.x_min
        self.real_h = self.y_max - self.y_min

    def __call__(self, results):
        # pull out boxes & labels
        gt_boxes = results['gt_bboxes_3d']    # (N,7) numpy
        gt_labels = results['gt_labels_3d']   # (N,)   numpy or memoryview

        # init mask to 0 (background class)
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        for box, cls in zip(gt_boxes, gt_labels):
            x, y = box[0], box[1]
            w, l = box[3], box[4]
            yaw   = box[6]
            corners = np.array([
               [ w/2,  l/2],
               [ w/2, -l/2],
               [-w/2, -l/2],
               [-w/2,  l/2],
            ])
            R = np.array([
              [ np.cos(yaw), -np.sin(yaw)],
              [ np.sin(yaw),  np.cos(yaw)]
            ])
            pts = (corners @ R.T) + np.array([x,y])
            ix = ((pts[:,0] - self.x_min)/self.real_w * self.bev_w).astype(np.int32)
            iy = ((pts[:,1] - self.y_min)/self.real_h * self.bev_h).astype(np.int32)
            poly = np.stack([ix, iy], axis=-1)
            # Fill mask with cls + 1 to reserve 0 for background
            cv2.fillPoly(mask, [poly], int(cls) + 1)
        bev_label = torch.from_numpy(mask).long()
        results['bev_obj_label'] = bev_label
        
        return results
