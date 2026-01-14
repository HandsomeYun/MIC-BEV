import copy
import json
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
import torch
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
from pyquaternion import Quaternion
from .nuscence_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox,center_distance,accumulate,DetectionMetricDataList,calc_ap, calc_tp, quaternion_yaw
from projects.mmdet3d_plugin.datasets.pipelines.bev_obj_label import GetBEVObjLabel

def eval_map(preds, gts, num_classes, ignore_index=255):
    total_inter = np.zeros(num_classes, dtype=np.float64)
    total_union = np.zeros(num_classes, dtype=np.float64)

    for pred, gt in zip(preds, gts):
        valid = gt != ignore_index
        for c in range(num_classes):
            p = (pred == c) & valid
            g = (gt   == c) & valid
            total_inter[c] += (p & g).sum()
            total_union[c] += (p | g).sum()

    class_iou = total_inter / (total_union + 1e-10)
    mean_iou  = float(np.nanmean(class_iou))
    return mean_iou, class_iou


@DATASETS.register_module()
class M2IDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, 
                 queue_length=4, 
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 eval_cfg = None,
                 path_prefix_replace=None,
                **kwargs):
        super().__init__(**kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.pc_range = pc_range
        # list of (old_prefix, new_prefix) tuples to remap absolute paths from PKL
        self.path_prefix_replace = path_prefix_replace or []
        self.ego_position_map = {
            info["token"]: tuple(info["ego2global_translation"][:2])
            for info in self.data_infos
        }
        if eval_cfg is not None:
            self.eval_cfg  = eval_cfg
        else:
            self.eval_cfg = {
                "dist_ths": [0.5, 1.0, 2.0, 4.0],
                "dist_th_tp": 2.0,
                "min_recall": 0.1,
                "min_precision": 0.1,
                "mean_ap_weight": 1,
                "class_names":['car'],
                "tp_metrics":['trans_err', 'scale_err', 'orient_err', 'vel_err'],
                "err_name_maping":{'trans_err': 'mATE','scale_err': 'mASE','orient_err': 'mAOE','vel_err': 'mAVE','attr_err': 'mAAE'}
            }
        
        # Add semantic segmentation attributes based on config
        self.sem_seg_classes = [
            'background', 'driving', 'sidewalk', 'crosswalk',
            'shoulder', 'border', 'parking'
        ]
        self.num_map_classes = len(self.sem_seg_classes)
        self.num_obj_classes = len(self.eval_cfg['class_names']) + 1
        
        print(f"[INFO] Loaded {len(self.data_infos)} total samples in split.")
        print(f"[INFO] Semantic classes: {self.sem_seg_classes}")
        print(f"[INFO] Number of map classes: {self.num_map_classes}")
    
    def _remap_path(self, p):
        """Optionally remap an old absolute prefix to a new one."""
        for old, new in self.path_prefix_replace:
            if p.startswith(old):
                return p.replace(old, new, 1)
        return p
    
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                print(f"[Dataset] get_data_info({i}) returned None. ")
                return None
            img_paths = input_dict.get('img_filename', None)
            if img_paths is None or not isinstance(img_paths, (list, tuple)) or len(img_paths) == 0:
                print(f"[Dataset] No image paths found at index {i}. Skipping this sample. input_dict {input_dict}")
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)

            # Debug to check if ObjectRangeFilter is even called
            if example is not None:
                if 'gt_labels_3d' in example:
                    gt_labels = example['gt_labels_3d']._data
            else:
                # print(f"[DEBUG] idx {i} ➜ pipeline returned None", flush=True)
                return None

            if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                # print(f"[Dataset] Skipped idx {i} because gt_labels_3d is invalid.", flush=True)
                return None


            queue.append(example)
        return self.union2one(queue)


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                # metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][:3] = [0.0, 0.0, 0.0]
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = (torch.tensor(metas_map[i]['can_bus'][:3]) - torch.tensor(prev_pos)).tolist()
                # metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        # 🟩 Add current epoch from dataset object
        for i in metas_map:
            metas_map[i]['epoch'] = getattr(self, 'epoch', 0)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue
    
    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        map_path = self._remap_path(info['map_path'])
        if not osp.isabs(map_path):
            map_path = osp.join(self.data_root, map_path)

        input_dict = dict(
            sample_idx=info['token'],
            # pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            map_path=map_path,  
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                img_path = self._remap_path(cam_info['data_path'])
                if not osp.isabs(img_path):
                    img_path = osp.join(self.data_root, img_path)
                image_paths.append(img_path)
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                #intrinsic = cam_info['cam_intrinsic']
                intrinsic = np.array(cam_info['cam_intrinsic'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        
        return input_dict
    
    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                print(f"[Dataset] Skipped idx {idx} due to invalid sample.", flush= True)
                idx = self._rand_another(idx)
                continue
            return data

    def format_results(self, results, jsonfile_prefix=None):
        # pull out only the detector outputs
        pts_results = [res['pts_bbox'] for res in results]

        # call the parent
        result_files, tmp_dir = super().format_results(
            pts_results, jsonfile_prefix)

        # if the parent gave us a single-string path, wrap it
        if isinstance(result_files, str):
            result_files = {'pts_bbox': result_files}

        return result_files, tmp_dir
    
    def evaluate(self,
                results,
                metric='bbox',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None,
                pipeline=None):
        """Evaluation using internal logic like B2D_Dataset."""
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_path = result_files['pts_bbox']

        with open(result_path) as f:
            result_data = json.load(f)
        pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
        meta = result_data['meta']

        gt_boxes = self.load_gt()
        class_range = self.eval_cfg.get('class_range', None)
        if class_range:
            gt_boxes = filter_boxes_by_class_range(gt_boxes, class_range, self.ego_position_map)

        metric_data_list = DetectionMetricDataList()
        metrics = DetectionMetrics(self.eval_cfg)

        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        for class_name in self.eval_cfg['class_names']:
            for dist_th in self.eval_cfg['dist_ths']:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                metrics.add_label_ap(class_name, dist_th, ap)

            for metric_name in self.eval_cfg['tp_metrics']:
                metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = meta.copy()

        print('mAP: %.4f' % metrics_summary['mean_ap'])
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (self.eval_cfg['err_name_maping'][tp_name], tp_val))
        print('NDS: %.4f' % metrics_summary['nd_score'])

        print('\nPer-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                % (class_name, class_aps[class_name],
                    class_tps[class_name]['trans_err'],
                    class_tps[class_name]['scale_err'],
                    class_tps[class_name]['orient_err'],
                    class_tps[class_name]['vel_err']))

        detail = dict()
        metric_prefix = 'bbox_NuScenes'
        for name in self.eval_cfg['class_names']:
            for k, v in metrics_summary['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics_summary['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics_summary['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail[f'{metric_prefix}/{self.eval_cfg["err_name_maping"][k]}'] = val
        detail[f'{metric_prefix}/NDS'] = metrics_summary['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics_summary['mean_ap']
        
        per_sample_ap = []
        for info in self.data_infos:
            token = info['token']
            if token not in gt_boxes.boxes or token not in pred_boxes.boxes:
                print(f"[WARNING] Skipped token {token}: not in GT or prediction boxes.")
                continue

            gt = {token: gt_boxes.boxes[token]}
            pred = {token: pred_boxes.boxes[token]}
            
            gt_eval = EvalBoxes()
            pred_eval = EvalBoxes()
            gt_eval.add_boxes(token, gt[token])
            pred_eval.add_boxes(token, pred[token])

            metric_data = DetectionMetricDataList()
            for class_name in self.eval_cfg['class_names']:
                if not any(b.detection_name == class_name for b in gt[token]):
                    continue  # skip classes not in GT for this sample
                md = accumulate(gt_eval, pred_eval, class_name, center_distance, dist_th=2.0)
                metric_data.set(class_name, 2.0, md)
                ap = calc_ap(md, min_recall=0.1, min_precision=0.1)
                per_sample_ap.append(ap)
            
        np.save("/home/yun/MIC-BEV_Official/data_dumping/bevformer_bbox_ap.npy", per_sample_ap)
        print(f"Saved at {'/home/yun/MIC-BEV_Official/data_dumping/bevformer_bbox_ap.npy'}")
        
        if not any('semantic_map' in r for r in results):
            print("[INFO] No semantic_map in results; skipping segmentation evaluation.")
            return detail
        
        # ──────────────────Map Generation────────────────
        # 2)Collect BEV-segmentation predictions & GTs
        map_preds = [res['semantic_map'] for res in results]
        
        # Load ground truth semantic maps from files
        map_gts = []
        for info in self.data_infos:  
            seg_path = info['map_path']
            if not osp.isabs(seg_path):
                seg_path = osp.join(self.data_root, seg_path)
            if seg_path.endswith('.npy'):
                seg = np.load(seg_path)
            else:
                seg = mmcv.imread(seg_path, flag='unchanged')
            map_gts.append(seg)

        # 3) compute mean IoU + per-class IoU
        mean_iou, class_iou = eval_map(
            map_preds,
            map_gts,
            num_classes=self.num_map_classes,
            ignore_index=255)
        # 4) pixel‐accuracy
        pix_acc = np.mean([
            (pred == gt).sum() / gt.size
            for pred, gt in zip(map_preds, map_gts)
        ])
        
        # 5) print segmentation metrics to stdout
        print("\nSemantic BEV‐segmentation results:")
        print(f"  mIoU:  {mean_iou:.4f}")
        print(f"  pAcc:  {pix_acc:.4f}")
        print("  Per‐class IoU:")
        for cls_name, iou_val in zip(self.sem_seg_classes, class_iou):
            print(f"    {cls_name:10s}: {iou_val:.4f}")

        # 6) merge into detail dict
        detail['seg_mIoU'] = float(f"{mean_iou:.4f}")
        detail['seg_pAcc'] = float(f"{pix_acc:.4f}")
        for cls_name, iou_val in zip(self.sem_seg_classes, class_iou):
            detail[f'seg_iou_{cls_name}'] = float(f"{iou_val:.4f}")
        
        # ──────────────Object‐Occupancy Segmentation Evaluation───────────────
        if not any('semantic_obj' in r for r in results):
            print("[INFO] No semantic_obj in results; skipping object‐occupancy evaluation.")
        else:
             # 1) collect your predictions
            obj_preds = [r['semantic_obj'] for r in results]
            # 2) directly collect the GT masks you threaded through
            obj_gts   = self.load_bev_obj_gt()

            # 3) compute your metrics exactly as before
            mean_obj_iou, class_obj_iou = eval_map(
                obj_preds, obj_gts,
                num_classes=self.num_obj_classes
            )
            pix_acc_obj = np.mean([
                (pred == gt).sum() / gt.size
                for pred, gt in zip(obj_preds, obj_gts)
            ])

            # 4) print & store
            print("\nObject‐Occupancy BEV‐segmentation results:")
            print(f"  mIoU:  {mean_obj_iou:.4f}")
            print(f"  pAcc:  {pix_acc_obj:.4f}")
            print("  Per‐class IoU:")
            for cls_idx, iou_val in enumerate(class_obj_iou):
                print(f"    class_{cls_idx:2d}: {iou_val:.4f}")

            detail['occ_mIoU']   = float(f"{mean_obj_iou:.4f}")
            detail['occ_pAcc']   = float(f"{pix_acc_obj:.4f}")
            for idx, iou_val in enumerate(class_obj_iou):
                detail[f'occ_iou_{idx}'] = float(f"{iou_val:.4f}")

        return detail


    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            info = self.data_infos[i]
            frame_token = info['token']
            annos = self.get_ann_info(i)

            # 1) pull out the bboxes, whether box-object or raw ndarray
            raw = annos['gt_bboxes_3d']
            if hasattr(raw, 'tensor'):
                gt_boxes = raw.tensor.numpy()
            else:
                gt_boxes = np.array(raw)

            # 2) normalize dims: single‐box → (1,D), empty → (0,D)
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes[None, :]
            if gt_boxes.size == 0:
                all_annotations.add_boxes(frame_token, [])
                # no GT this frame → skip entirely
                continue
            sample_boxes = []
            sample_data = self.data_infos[i]
            annos = self.get_ann_info(i)
            # gt_boxes = annos['gt_bboxes_3d'].tensor.numpy()
            gt_names = annos['gt_names']
            num_points = sample_data['num_lidar_pts']
            frame_token = sample_data['token']

            for j in range(gt_boxes.shape[0]):
                class_name = gt_names[j]
                if class_name not in self.eval_cfg['class_names']:
                    continue
                sample_boxes.append(DetectionBox(
                    sample_token=frame_token,
                    translation=gt_boxes[j, :3],
                    size=gt_boxes[j, 3:6],
                    # rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j, 6] - np.pi/2)),
                    rotation = Quaternion(axis=[0, 0, 1], radians=gt_boxes[j, 6]),
                    velocity=gt_boxes[j, 7:9] if gt_boxes.shape[1] >= 9 else [0.0, 0.0],
                    num_pts=int(num_points[j]) if isinstance(num_points[j], (int, float)) else 1,
                    detection_name=class_name,
                    detection_score=-1.0,
                    attribute_name=class_name
                ))
            all_annotations.add_boxes(frame_token, sample_boxes)
        return all_annotations
    
    def load_bev_obj_gt(self):
        """Return a list of (H×W) numpy masks of the ground-truth BEV object occupancy."""
        labeler = GetBEVObjLabel(self.bev_size[0], self.bev_size[1], self.pc_range)
        obj_gts = []
        for idx in range(len(self.data_infos)):
            # 1) pull the GT boxes & labels from annotations
            annos = self.get_ann_info(idx)

            # 2) convert to numpy
            bboxes = annos['gt_bboxes_3d']
            if hasattr(bboxes, 'tensor'):
                boxes_np = bboxes.tensor.numpy()
            else:
                boxes_np = np.array(bboxes)
            labels = annos['gt_labels_3d']
            if isinstance(labels, torch.Tensor):
                labels_np = labels.cpu().numpy()
            else:
                labels_np = np.array(labels)

            # 3) feed them into your pipeline transform
            results = {
                'gt_bboxes_3d': boxes_np,
                'gt_labels_3d': labels_np
            }
            out = labeler(results)

            # 4) grab the mask
            obj_gts.append(out['bev_obj_label'].cpu().numpy())

        return obj_gts

def filter_boxes_by_class_range(boxes, class_range, ego_position_map):
    """Filter EvalBoxes by class-specific spatial limits."""
    filtered = EvalBoxes()
    for token, box_list in boxes.boxes.items():
        x_ego, y_ego = ego_position_map[token]
        class_counts = {}
        kept_counts = {}
        kept_boxes = []
        for box in box_list:
            class_name = box.detection_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if class_name not in class_range:
                # Only warn once per class per frame
                if class_counts[class_name] == 1:
                    print(f"[WARN] No class range for {class_name}, skipping.", flush=True)
                continue
            x_global, y_global = box.translation[:2]
            x_rel = x_global - x_ego
            y_rel = y_global - y_ego
            x_range, y_range = class_range[class_name]
            if abs(x_rel) <= x_range and abs(y_rel) <= y_range:
                kept_boxes.append(box)
                kept_counts[class_name] = kept_counts.get(class_name, 0) + 1
        if kept_boxes:
            # for class_name in kept_counts:
                # print(f"[Filter] {class_counts[class_name]} → {kept_counts[class_name]} boxes kept for class '{class_name}' in frame {token}")
            filtered.add_boxes(token, kept_boxes)
    return filtered