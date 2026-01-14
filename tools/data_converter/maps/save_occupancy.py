import os, sys, pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from mmcv import Config

# 1) Make sure your plugin dir is on sys.path
PLUGIN_DIR = '/home/yun/MIC-BEV_Official/projects/mmdet3d_plugin'
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

# 2) Import your BEV‐label generator directly
from projects.mmdet3d_plugin.datasets.pipelines.bev_obj_label import GetBEVObjLabel

# --- Settings ---
target_token = 'town04_intersection1_3cam_t_c_day_s71_-125_000034'
info_path    = '/home/yun/MIC-BEV_Official/data/clear_day/v2xset_infos_temporal_test.pkl'
cfg_path     = '/home/yun/MIC-BEV_Official/projects/configs/bevformer/mic-bev-seg-gnn.py'

# Your four object classes
class_names = ['car','bicycle','truck','pedestrian']
name2id     = {n:i for i,n in enumerate(class_names)}

# Palette for the semantic map
palette = np.array([
    [191,216,109],
    [211,211,211],
    [240,230,180],
    [229,204,255],
    [128,128,128],
    [169,169,169],
    [  0,188,212],
],dtype=np.uint8)

# --- 1) Load infos + find your sample by token ---
with open(info_path,'rb') as f:
    data = pickle.load(f)
infos = data.get('infos', data)
sample = next((i for i in infos if i['token']==target_token), None)
assert sample, f"Token {target_token} not found"

# --- 2) Build global‐frame GT boxes + labels ---
gt_boxes  = sample['gt_boxes']        # (N,7) in global coords
gt_names  = sample['gt_names']        # (N,) strings
gt_labels = np.array([name2id[n] for n in gt_names], dtype=np.int64)

# --- 3) Load config + semantic map shape + instantiate BEV generator ---
cfg      = Config.fromfile(cfg_path)
sem      = np.load(sample['map_path'])   # (H, W)
bev_h, bev_w = sem.shape
pc_range    = cfg.data.train.pc_range
bev_gen     = GetBEVObjLabel(bev_h, bev_w, pc_range)

# --- 4) Transform boxes global→ego ---
ego_t = np.array(sample['ego2global_translation'])      # (3,)
q     = Quaternion(sample['ego2global_rotation'])       # expects [w,x,y,z]
q_inv = q.inverse
ego_yaw,_,_ = q.yaw_pitch_roll                        # yaw in radians

boxes = gt_boxes.copy()
# 4a) center coordinates
centers_global = boxes[:, :3] - ego_t  # shape (N,3)
# rotate each point individually into ego frame
centers_local = np.stack([q_inv.rotate(c) for c in centers_global], axis=0)
boxes[:, :3] = centers_local
# 4b) rotate yaw angle back into ego frame
boxes[:, 6] = boxes[:, 6] - ego_yaw - np.pi/2

# --- 5) Generate BEV‐object mask and save PNG ---
bev_mask = bev_gen({
    'gt_bboxes_3d': boxes,
    'gt_labels_3d': gt_labels
})['bev_obj_label'].numpy()   # (H, W)

palette_rgba = np.array([
    [128, 128, 128, 128],    # 0 transparent
    [255, 158,   0, 255],    # 1 car
    [220,  20,  60, 255],    # 2 cyclist
    [255,  99,  71, 255],    # 3 truck
    [0,   0, 230, 255],    # 4 pedestrian
], dtype=np.uint8)


cmap    = plt.get_cmap('tab20', bev_mask.max()+1)
bev_rgb = (cmap(bev_mask)[...,:3] * 255).astype(np.uint8)
out1    = f"bev_obj_label_{target_token}.png"
# 5b) map your mask → RGBA image
bev_rgba = palette_rgba[bev_mask]   # shape (H, W, 4)
# 5c) OpenCV wants BGRA order when writing PNG
bev_bgra = bev_rgba[..., [2,1,0,3]]
out1 = f"bev_obj_label_{target_token}.png"
cv2.imwrite(out1, bev_bgra)
print("Wrote (transparent BG, colored objs):", out1)

# --- 6) Save the raw semantic map as PNG ---
sem_rgb = palette[sem]    # (H, W, 3)
out2    = f"semantic_map_{target_token}.png"
cv2.imwrite(out2, cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))
print("Wrote", out2)
