#    RoScenes
#    Copyright (C) 2024  Alibaba Cloud
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import json
import math
import os
import shutil
import cv2
import numpy as np
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet.datasets import DATASETS
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

from roscenes.data import Clip, Frame, Scene, ConcatScene
from roscenes.data.metadata import Split
from roscenes.evaluation.detection import MultiView3DEvaluator, ThresholdMetric, DetectionEvaluationConfig
from roscenes.evaluation.detection import Prediction
from roscenes.transform import xyzwlhq2kitti, kitti2xyzwlhq, kitti2corners, xyzwlhq2bevbox


COLOR_PALETTE = [
  [0, 0, 0],       # 0 → "other"      → black
  [0, 0, 255],     # 1 → "truck"      → red (in OpenCV BGR format)
  [255, 0, 0],     # 2 → "bus"        → blue
  [0, 255, 0],     # 3 → "van"        → green
  [0, 255, 255],   # 4 → "car"        → yellow
]


@DATASETS.register_module(force=True)
class RoScenesDataset(Custom3DDataset):
    """Customized 3D dataset.

    This is the base dataset of SUNRGB-D, ScanNet, nuScenes, and KITTI
    dataset.

    .. code-block:: none

    [
        {'sample_idx':
         'lidar_points': {'lidar_path': velodyne_path,
                           ....
                         },
         'annos': {'box_type_3d':  (str)  'LiDAR/Camera/Depth'
                   'gt_bboxes_3d':  <np.ndarray> (n, 7)
                   'gt_names':  [list]
                   ....
               }
         'calib': { .....}
         'images': { .....}
        }
    ]

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR'. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
    """

    CLASSES = [
        "other",
        "truck",
        "bus",
        "van",
        "car",
    ]

    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

    data_infos: Scene

    def __init__(self,
                 data_root,
                 ann_file,
                 data_list=None,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 use_valid_flag=False,
                 test_mode=False):
        super().__init__(data_root, ann_file, pipeline, classes, modality, box_type_3d, filter_empty_gt, test_mode)
        self.seq_split_num = 1
        self._set_sequence_group_flag()

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            
            try:
                prev = self.data_infos[idx].previous
            except NotImplementedError:
                prev = None
            if idx != 0 and prev is not None:
                # Not first frame and previous is None  -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0,
                                bin_counts[curr_flag],
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            Scene
        """
        scene = Scene.load(self.data_root)
        print('load a scene with length:', len(scene))
        return scene

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - ann_info (dict): Annotation info.
        """
        # a frame
        frame: Frame = self.data_infos[index]

        input_dict = dict(
            sample_idx=index,
            scene_token=frame.parent.token,
            timestamp=frame.timeStamp / 1e6)

        clip: Clip = frame.parent

        # NOTE: Copy it to avoid inplace manipulation on raw data --- This causes a messd up.
        intrinsics = [c.intrinsic.copy() for c in clip.cameras.values()]
        extrinsics = [c.extrinsic.copy() for c in clip.cameras.values()]
        world2image = [c.world2image.copy() for c in clip.cameras.values()]
        input_dict.update(dict(
            img_timestamp=[frame.timeStamp / 1e6 for _ in range(len(frame.imagePaths))],
            img_filename=list(frame.images.values()),
            lidar2img=world2image,
            lidar2cam=extrinsics,
            cam_intrinsic=intrinsics
        ))

        input_dict['can_bus'] = np.zeros(18, dtype=np.float32)

        if not self.test_mode:
            gt_bboxes = LiDARInstance3DBoxes(np.concatenate([xyzwlhq2kitti(frame.boxes3D), frame.velocities], -1).astype(np.float32), box_dim=7 + 2, origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            annos = dict(
                gt_bboxes_3d=gt_bboxes,
                gt_labels_3d=frame.labels.copy(),
                gt_names=self.CLASSES,
                bboxes_ignore=None
            )
            input_dict['ann_info'] = annos
        return input_dict

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=True,
                 out_dir="results",
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        if isinstance(self.data_infos, ConcatScene):
            metadata = self.data_infos.scenes[0].metadata
        else:
            metadata = self.data_infos.metadata

        visFolder = "/data3/yun/visualization-roscenes"
        os.makedirs(visFolder, exist_ok=True)
        previousClip = None
        predictionList = list()
        clips = list()
        
        # BEV parameters (meters and pixels)
        xlim     = (-400, 400)
        ylim     = (-100, 100)
        meters_per_px = 0.2  # e.g. 1 px = 0.2 m
        W = int((xlim[1] - xlim[0]) / meters_per_px)
        H = int((ylim[1] - ylim[0]) / meters_per_px)
        bev_size = (W, H)

        def world2bev(pts2d):
            pix = []
            for box in pts2d:
                coords = []
                for x, y in box:
                    # uniform‐scale mapping:
                    px = (x - xlim[0]) / meters_per_px
                    py = (ylim[1] - y)       / meters_per_px
                    coords.append([int(px), int(py)])
                pix.append(np.array(coords, dtype=np.int32))
            return pix

        def draw(bev_img, pix, color):
            for box in pix:
                cv2.polylines(bev_img, [box], True, color, 2)
            return bev_img
        
        for i, res in enumerate(results):
            frame = self.data_infos[i]
            view  = frame
            boxes_3d  = res['pts_bbox']['boxes_3d']
            scores_3d = res['pts_bbox']['scores_3d']      # torch.Tensor, shape [N]
            labels_3d = res['pts_bbox']['labels_3d']      # torch.Tensor, shape [N]

            # now filter low‐confidence
            keep = scores_3d >= 0.35
            boxes_3d  = boxes_3d[keep]
            scores_3d = scores_3d[keep]
            labels_3d = labels_3d[keep]

            # Flip yaw in the original boxes_3d tensor to ensure consistency across all operations
            boxes_3d.tensor[:, 6] = -boxes_3d.tensor[:, 6]
            # Normalize yaw to (-pi, pi]
            boxes_3d.tensor[:, 6] = (boxes_3d.tensor[:, 6] + math.pi) % (2 * math.pi) - math.pi
            
            # [N, 7+2]
            xyzwlhr, velocities = boxes_3d.tensor[:, :7].detach().clone(), boxes_3d.tensor[:, 7:9].detach().clone()

            xyzwlhq = kitti2xyzwlhq(xyzwlhr.cpu().numpy().copy())
            print(f"\n processing [Frame {i}")
            #=========coordinate tranform for bev========
            pred_corners = boxes_3d.corners.detach().cpu().numpy()
            all_footprints = []
            for corners in pred_corners:               # corners: (8,3)
                # 1) select bottom corners
                # corners: (8,3)
                z_vals = corners[:, 2]
                # get indices of the 4 smallest z’s
                bottom_idxs = np.argsort(z_vals)[:4]
                pts = corners[bottom_idxs, :2]

                # 2) sort them around centroid
                center = pts.mean(axis=0)                  # (x_c, y_c)
                # compute angle of each point relative to center
                angles = np.arctan2(pts[:,1] - center[1],
                                    pts[:,0] - center[0])
                order = np.argsort(angles)                 # ascending angle → CCW order
                sorted_pts = pts[order]

                all_footprints.append(sorted_pts)

            pred_footprints = np.stack(all_footprints, 0)  # (N,4,2)
            # GT corners from Scene
            gt_kitti   = xyzwlhq2kitti(frame.boxes3D)
            gt_corners = kitti2corners(gt_kitti)
            gt_footprints = gt_corners[:, :4, :2]
            
            if i % 60 == 59:
            # if True:
                #==================Plot BEV===========================
                # 1) bev_all: both GT and pred
                bev_all = np.zeros((H, W, 4), dtype=np.uint8)
                
                # these both are shape (N,4,2)—loop through them
                # gt_pixs   = world2bev(gt_footprints)
                pred_pixs = world2bev(pred_footprints)
                for poly, label in zip(pred_pixs, labels_3d):
                    color = COLOR_PALETTE[label]  # RGB triplet
                    contour = poly.reshape(-1,1,2).astype(np.int32)
                    cv2.polylines(
                        bev_all,
                        [contour],
                        True,
                        tuple(color) + (255,),   # add alpha if using 4‑ch image
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                out_dir_i = os.path.join(visFolder, str(i))
                os.makedirs(out_dir_i, exist_ok=True)
                # OpenCV will honor the alpha channel when writing PNGs:
                cv2.imwrite(os.path.join(out_dir_i, f"bev_transparent_{i:03d}.png"), bev_all)
                print(f"Saved BEV image here", {os.path.join(visFolder, str(i))})
                # #================Plot projection to image ====================
                # Use the same filtered boxes for image projection as used for BEV
                boxes2vis   = boxes_3d
                orig_scores = scores_3d
                orig_labels = labels_3d
                # projectedResults = view.parent.projection(kitti2corners(boxes_3d.tensor.detach().clone().cpu().numpy()[..., :7]))
                projectedResults = view.parent.projection(boxes2vis.corners.detach().cpu().numpy())
                # gt_projected = view.parent.projection(gt_corners)
                for k, (imagePath, (boxes, vis)) in enumerate(zip(view.images.values(), projectedResults)):
                    img = cv2.imread(imagePath)
                    cleanImg = img.copy()
                    
                    # #=======Plot gt=====================
                    # gt_boxes2d, gt_vis2d = gt_projected[k]
                    # for gt_box, flag in zip(gt_boxes2d, gt_vis2d):
                    #     if not flag:
                    #         continue
                    #     # gt_box.shape == (8,2)
                    #     top_pts    = gt_box[0:4, :2].astype(np.int32)
                    #     bottom_pts = gt_box[4:8, :2].astype(np.int32)
                    #     # draw top face
                    #     cv2.polylines(img, [top_pts],    True, (0,255,0), 2, cv2.LINE_AA)
                    #     # draw bottom face (footprint)
                    #     cv2.polylines(img, [bottom_pts], True, (0,255,0), 2, cv2.LINE_AA)
                    #     # draw vertical edges
                    #     for idx in range(4):
                    #         p0 = tuple(top_pts[idx])
                    #         p1 = tuple(bottom_pts[idx])
                    #         cv2.line(img, p0, p1, (0,255,0), 2, cv2.LINE_AA)
                    #=======Plot prediction==============
                    scores2vis = orig_scores
                    labels2vis = orig_labels

                    # then sort
                    sortIds    = np.argsort(-np.mean(boxes[..., -1], -1))
                    boxes = boxes[sortIds, ..., :2]
                    scores2vis = scores2vis[sortIds]
                    labels2vis = labels2vis[sortIds]
                    vis        = vis[sortIds]

                    # [4] in xy format
                    for box3d, score, label in zip(boxes[vis], scores2vis[vis], labels2vis[vis]):
                        # crop the clean object region
                        # paste to current image
                        # then draw line
                        objectPoly = Polygon(box3d)
                        objectPoly = np.array(objectPoly.convex_hull.exterior.coords, dtype=np.int32)
                        mask = np.zeros_like(cleanImg[..., 0])
                        cv2.drawContours(mask, [objectPoly], -1, (255, 255, 255), -1, cv2.LINE_AA)
                        # print(img.shape, cleanImg.shape, mask.shape)
                        fg = cv2.bitwise_and(cleanImg, cleanImg, mask=mask)
                        bg = (img * (1 - mask[..., None] / 255.)).astype(np.uint8)
                        img = fg + bg
                        cv2.polylines(img, [box3d[:4].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
                        cv2.polylines(img, [box3d[4:].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
                        cv2.polylines(img, [box3d[[0, 1, 5, 4]].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)
                        cv2.polylines(img, [box3d[[2, 3, 7, 6]].astype(int)], True, COLOR_PALETTE[label], 3, cv2.LINE_AA)

                    os.makedirs(os.path.join(visFolder, str(i)), exist_ok=True)
                    cv2.imwrite(os.path.join(visFolder, str(i), f"{view.token}_{k}.jpg"), img)
                    print(f"Saved image here", {os.path.join(visFolder, str(i), f"{view.token}_{k}.jpg"),})
              # #================End Plot projection to image ====================
            prediction = Prediction(
                            timeStamp=frame.timeStamp,
                            boxes3D=xyzwlhq,
                            velocities=velocities.cpu().numpy().copy(),
                            labels=labels_3d.cpu().numpy().copy(),
                            scores=scores_3d.cpu().numpy().copy(),
                            token=frame.token
                        )
            predictionList.append(prediction)

        groundtruth = self.data_infos

        evaluator = MultiView3DEvaluator(DetectionEvaluationConfig(
            self.CLASSES,
            [0.5, 1., 2., 4.],
            2.,
            ThresholdMetric.CenterDistance,
            500,
            0.0,
            [-400., -40., 0., 400., 40., 6.],
            # ["TranslationError", "ScaleError", "OrientationError"]
            [
                "PrecisionRecall",      # rank‑list metric for PR curves
                "TranslationError",     # ATE
                "ScaleError",           # ASE
                "OrientationError"      # AOE
            ]
        ))
        result = evaluator(groundtruth, predictionList)

        summary = result.summary
        os.makedirs(out_dir, exist_ok=True)
        print(summary)
        with open(os.path.join(out_dir, "result.json"), "w") as fp:
            json.dump(summary, fp)
        return summary