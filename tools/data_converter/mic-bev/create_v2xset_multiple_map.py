#!/usr/bin/env python3
import os
import re
import yaml
import json
import mmcv
import numpy as np
from PIL import Image
from yaml import CLoader as Loader
import random
import shutil
from tqdm import tqdm
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from tools.data_converter.maps.xodr_parser.parse_and_visualize import compute_filtered_pickles
from collections import OrderedDict, defaultdict
from tools.data_converter.maps.map_converter import generate_bev_map
import cv2


# ─── UTILITIES ─────────────────────────────────────────────────────────────────
# Only do this once at import‑time:
_float_pattern = re.compile(r'''^(?:
    [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
   |\.[0-9_]+(?:[eE][-+][0-9]+)?
   |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
   |[-+]?\.(?:inf|Inf|INF)
   |\.(?:nan|NaN|NAN))$''', re.X)

_yaml_loader = Loader
_yaml_loader.add_implicit_resolver(u'tag:yaml.org,2002:float',
                                  _float_pattern,
                                  list(u'-+0123456789.'))


def load_yaml(path: Path):
    with path.open('r') as f:
        data = yaml.load(f, Loader=_yaml_loader)
    # if you absolutely need the yaml_parser hook, do it here—but avoid eval.
    return data


#Carla UE4 world frame: X forward, Y right, Z up
#CV/Nuscenes world/LIDAR frame: X forward, Y left, Z up
#Ego frame: Origin at the vehicle (or lidar) center, keep as Nuscenes. 
# Standard (OpenCV/Nuscnes) camera frame: X right, Y down, Z forward
# Image (pixel) frame: 2D uv pixel.

# UE4 camera axes → standard
CV_to_ue4_rotate = np.array([[ 0, 0, 1, 0], # new X = old Z  
                                [ 1, 0, 0, 0], # new Y = old X  
                                [ 0,-1, 0, 0], # new Z = - old Y  
                                [ 0, 0, 0, 1]])

# lidar frame → right-handed ego frame
lidar_to_righthand_ego = np.array([[  0, 1, 0, 0], 
                                   [ -1, 0, 0, 0],
                                   [  0, 0, 1, 0],
                                   [  0, 0, 0, 1]])


lefthand_ego_to_lidar = np.array([[ 0, 1, 0, 0],
                                  [ 1, 0, 0, 0],
                                  [ 0, 0, 1, 0],
                                  [ 0, 0, 0, 1]])

# flip Y for left-to-right handed
left2right = np.eye(4)
left2right[1,1] = -1

# Inputs [x, y, z, roll, yaw, pitch]
def ego2world_LH(p):
    x, y, z, pitch, yaw, roll = p
    R = ( Quaternion(axis=[1,0,0], radians=np.deg2rad(roll)) *
          Quaternion(axis=[0,0,1], radians=np.deg2rad(yaw))  *
          Quaternion(axis=[0,1,0], radians=np.deg2rad(pitch)) ).rotation_matrix
    H = np.eye(4)
    H[:3,:3] = R
    H[:3, 3] = [x, y, z]
    return H                       # ego ➜ world  (LH)

def apply_trans(vec,world2ego):
    vec = np.concatenate((vec,np.array([1])))
    t = world2ego @ vec
    return t[0:3]

def auto_split_pkls(data_roots,PKL_ROOT, VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, USED_PERCENT):
    scenes = []
    for root in data_roots:
        scenes.extend(sorted([d for d in Path(root).iterdir() if d.is_dir()]))
    random.seed(42)
    random.shuffle(scenes)
    n_total = len(scenes)

    # Only keep a subset of scenes
    n_used = int(USED_PERCENT * n_total)
    scenes = scenes[:n_used]

    # Split into train/test
    n_train = int(0.7 * n_used)
    n_val   = int(0.1 * n_used)
    n_test  = n_used - n_train - n_val
    
    train_scenes = scenes[:n_train]
    val_scenes   = scenes[n_train:n_train + n_val]
    test_scenes  = scenes[n_train + n_val:]

    used_percent = (n_used / n_total) * 100
    print(f"[INFO] Using {used_percent:.1f}% of total {n_total} scenes (={n_used} scenes)")

    # Training data in ego frame
    create_v2xset_infos(train_scenes, PKL_ROOT, 'train', VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, use_ego_frame=True)
    
    # # # Val data in ego frame
    create_v2xset_infos(val_scenes, PKL_ROOT, 'val',  VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version,  use_ego_frame=False)

    #Same test and val
    create_v2xset_infos(test_scenes, PKL_ROOT, 'test',  VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version,  use_ego_frame=False)

    # # Test data in global frame
    create_v2xset_infos(scenes[:n_used], PKL_ROOT, 'all',  VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, use_ego_frame=False)
    # print("✅ splits ready\n")

# ─── STEP 1: split raw into PKLs ──────────────────────────────────────────────
def create_v2xset_infos(scenarios, out_path, split, VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, use_ego_frame=False):
    infos = []
    for sc in tqdm(scenarios, desc=f"Building {split} infos"):
        prot_path = sc / "data_protocol.yaml"
        if not prot_path.exists():
            raise FileNotFoundError(f"{prot_path} not found")
        datap = load_yaml(prot_path)
        town_name = datap['world']['town']
        # Look for all RSU directories, e.g., -1, -2, ...
        rsu_dirs = sorted([d for d in sc.iterdir() if d.is_dir() and d.name.startswith("-")])
        if not rsu_dirs:
            continue

        for rsu_dir in rsu_dirs:
            yamls = sorted(
                rsu_dir.glob("*.yaml"),
                key=lambda p: (m.group(0) if (m := re.search(r"\d+", p.stem)) else p.stem)
            )
            if not yamls:
                print(f"[WARN] {rsu_dir} has no *.yaml — skipping")
                continue
            frame_ids = [y.stem for y in yamls]
            # #===============Create Map============================================
            fi = load_yaml(rsu_dir/f"{frame_ids[0]}.yaml")
            pose_CARLA = fi.get("lidar_pose") or fi.get("lidar_pose0")
            center_xy = [pose_CARLA[0], -pose_CARLA[1]]
            ego_yaw_rad = -np.deg2rad(pose_CARLA[4])
            xy_str = f"{center_xy[0]:.2f}" + ',' + f"{center_xy[1]:.2f}"
            save_folder = MAP_OUT / town_name / xy_str
            save_folder.mkdir(parents=True, exist_ok=True)
            xodr_file = XODR_ROOT / f"{town_name}.xodr"
            if not xodr_file.exists():
                raise FileNotFoundError(f"Missing {xodr_file} for {rsu_dir}")
            npy_file = save_folder / "bev_label_map_200.npy"
            # only generate if it doesn’t already exist
            if not npy_file.exists():
                print(f"Path not exist {npy_file}")
                # 1) build the *north-up* raster 
                total_areas, crosswalks = compute_filtered_pickles(
                    xodr_file,
                    center_pose=center_xy,
                    range_x=51.2, range_y=51.2,
                    town_name=town_name,
                    out_dir=None
                )
                _ = generate_bev_map(
                    total_areas, crosswalks,
                    center_xy=center_xy,
                    town_name = town_name,
                    save_dir= str(save_folder),
                    downsample_size=200
                )
                
                # 2) load, rotate once by −yaw, overwrite in-place
                global_bev = np.load(npy_file)
                rotated     = rotate_full_bev(global_bev, ego_yaw_rad)
                flipped = np.flipud(rotated)
                np.save(npy_file, flipped)

            rel_map = str(npy_file)
            assert os.path.exists(npy_file), f"Map not saved: {npy_file}"
        
            for i, fid in enumerate(frame_ids):
                fi = load_yaml(rsu_dir/f"{fid}.yaml")
                pose_carla = fi.get("lidar_pose") or fi.get("lidar_pose0")
                E2W_LH = ego2world_LH(pose_carla)      
                E2W_RH = left2right @ E2W_LH @ left2right     # world→ego, RH  ← store this
                
                rotation_quat = Quaternion(matrix=E2W_RH[:3, :3]).elements.tolist()
                translation_xyz = E2W_RH[:3, 3].tolist()
                
                info = {
                    'token': f"{sc.name}_{rsu_dir.name}_{fid}",
                    'scene_token': f"{sc.name}_{rsu_dir.name}",
                    'prev': f"{sc.name}_{rsu_dir.name}_{frame_ids[i-1]}" if i > 0 else None,
                    'next': f"{sc.name}_{rsu_dir.name}_{frame_ids[i+1]}" if i < len(frame_ids) - 1 else None,
                    'cams': OrderedDict(),
                    "map_path": rel_map,
                    'can_bus': [0.0]*18,
                    'sweeps': [],
                    'frame_idx': int(fid),
                    'timestamp': int(fid),
                    'ego2global_translation': translation_xyz,
                    'ego2global_rotation': rotation_quat,
                    'lidar2ego_translation': [0,0,0],
                    'lidar2ego_rotation': [1,0,0,0],
                }
                
                # cameras
                cam_keys = sorted(k for k in fi.keys() if k.startswith("camera"))
                # now remap them to 0–3 **within this RSU**
                for local_idx, ck in enumerate(cam_keys[:4]):
                    cd = fi[ck]
                    if ck not in fi: continue
                    I = np.array(cd["intrinsic"])
                    #Sensor2ego
                    # Extrinsic: L to C. So C2L is inverse of extrinsic.
                    cam2lidar_LH  = np.linalg.inv(np.array(cd['extrinsic']))  # shape (4,4)
                    #Expect input is in CV style camera frame, output is in RH ego frame. 
                    sensor2ego = left2right @ cam2lidar_LH @ CV_to_ue4_rotate
                    R_s2e = sensor2ego[:3,:3]
                    t_s2e = sensor2ego[:3,3]
                    
                    #World2cam
                    cam_pose = np.array(cd['cords'])
                    E2W_LH = ego2world_LH(cam_pose)
                    W2C_LH_source = np.linalg.inv(E2W_LH)  
                    W2C_RH_CV_A = np.linalg.inv(CV_to_ue4_rotate) @ W2C_LH_source @ left2right
                    
                    info['cams'][f"CAM_{local_idx}"] = {
                        'data_path':                    str((rsu_dir / f"{fid}_{ck}.png")),
                        'sensor2lidar_rotation':         R_s2e.tolist(),
                        'sensor2lidar_translation':      t_s2e.tolist(),
                        'ego2global_translation':      translation_xyz,
                        'ego2global_rotation':         rotation_quat,
                        'world2cam':                   W2C_RH_CV_A,
                        'cam_intrinsic':               I,
                        'timestamp':                   int(fid),
                    }
                
                # gt boxes - transform based on split
                boxes, names, vels = [], [], []
                # Process all object types
                for obj_type in ['vehicles', 'pedestrians', 'cyclists', 'cars', 'trucks']:
                    for obj in fi.get(obj_type, {}).values():
                        # Map to standardized class name
                        class_name = OBJECT_CLASS_MAPPING.get(obj_type.lower(), obj_type.lower())
                        
                        # Skip if not a valid class
                        if class_name not in VALID_CLASSES:
                            print(f"Warning: Unknown class {obj_type} mapped to {class_name}")
                            continue

                        # center_carla = np.array(obj['center']) + np.array(obj['location'])
                        if class_name == "bicycle":
                            center = np.array(obj['center'])
                            location = np.array(obj['location'])
                            center_carla = [center[0] + location[0], location[1], center[2] + location[2]]
                        else:
                            center_carla = np.array(obj['center']) + np.array(obj['location'])

                        center_RH = [center_carla[0], -center_carla[1], center_carla[2]]
                        w = obj['extent'][1] * 2
                        l = obj['extent'][0] * 2
                        h = obj['extent'][2] * 2
                        extent = [w, l, h]

                        speed_kmh = obj.get('speed', 0.0)  # Default to 0 if speed not present
                        speed_ms  = speed_kmh * 0.27778     

                        yaw_car = np.deg2rad(obj['angle'][1])
                        yaw_nusc = -yaw_car
                        # normalize to [-π, π)
                        yaw_nusc = (yaw_nusc - np.pi) % (2*np.pi) - np.pi
                        
                        if use_ego_frame:
                            W2E_RH = np.linalg.inv(E2W_RH)
                            center_ego = apply_trans(center_RH, W2E_RH)
                            yaw_ego = -yaw_nusc - np.pi / 2
                            box = center_ego.tolist() + extent + [yaw_ego]
                            vel = [speed_ms * np.cos(yaw_ego),
                                speed_ms * np.sin(yaw_ego)]
                        else:
                            box = center_RH + extent + [yaw_nusc]
                            vel = [speed_ms * np.cos(yaw_nusc),
                                speed_ms * np.sin(yaw_nusc)]
                        
                        boxes.append(box)
                        names.append(class_name)
                        vels.append(vel)
                
                info.update({
                    'gt_boxes':     np.array(boxes),
                    'gt_names':     np.array(names),
                    'gt_velocity':  np.array(vels),
                    'valid_flag':   np.ones(len(boxes),dtype=bool),
                    'num_lidar_pts':np.ones(len(boxes)),
                    'num_radar_pts':np.zeros(len(boxes)),
                })
                
                # ── build temporal sweeps (previous camera frames) ───────────────────────────
                sweeps = []

                # Pre-compute the global → key-ego transform once per key-frame ────────
                # Key-frame lidar (ego_k) pose in CARLA → right-handed global frame
                E2W_RH_k = left2right @ ego2world_LH(pose_carla) @ left2right
                R_e2g_k  = E2W_RH_k[:3, :3]                 # ego_k  →  global   rotation
                t_e2g_k  = E2W_RH_k[:3,  3]                 # ego_k  →  global   translation
                R_g2e_k  = R_e2g_k.T                        # global →  ego_k    (inverse)
                t_g2e_k  = -t_e2g_k @ R_g2e_k               # row-vector form

                # ------------------------------------------------ iterate backward in time ---
                for s in range(1, MAX_SWEEPS + 1):
                    idx = i - s
                    if idx < 0:
                        break

                    pid        = frame_ids[idx]                             # past frame ID
                    yaml_path  = rsu_dir / f"{pid}.yaml"
                    if not yaml_path.exists():
                        continue

                    # ── pose of sweep lidar (ego_s) in global coordinates ────────────────
                    fi_s     = load_yaml(yaml_path)
                    pose_carla_s = fi_s.get("lidar_pose0") or fi_s.get("lidar_pose")

                    E2W_RH_s = left2right @ ego2world_LH(pose_carla_s) @ left2right
                    R_e2g_s  = E2W_RH_s[:3, :3]              # ego_s  →  global
                    t_e2g_s  = E2W_RH_s[:3,  3]

                    # ── iterate over the four cameras of that sweep frame ───────────────
                    for ci in range(4):
                        ck = f"camera{ci}"
                        if ck not in fi_s:
                            continue
                        cd_s = fi_s[ck]

                        # 1)  camera  →  sweep-ego  (right-handed)
                        cam2lidar_LH = np.linalg.inv(np.array(cd_s["extrinsic"]))   # L→C given
                        cam2ego_s    = left2right @ cam2lidar_LH @ CV_to_ue4_rotate
                        R_c2e_s      = cam2ego_s[:3, :3]
                        t_c2e_s      = cam2ego_s[:3, 3]

                        # 2)  sweep-ego → global      (R_e2g_s, t_e2g_s)
                        # 3)  global     → key-ego    (R_g2e_k, t_g2e_k)
                        # 4)  key-ego    → lidar_top  (identity in this dataset)
                        # Compose rotations: R_tot = R_c2e_s · R_e2g_s · R_g2e_k
                        R_tot = R_c2e_s @ R_e2g_s @ R_g2e_k

                        # Compose translations (row-vector maths, same order):
                        t_tot = (
                            t_c2e_s @ R_e2g_s @ R_g2e_k   # cam→ego_s→global→ego_k
                            + t_e2g_s @ R_g2e_k             #     ego_s→global→ego_k
                            + t_g2e_k                       #                     global→ego_k
                        )

                        sweeps.append({
                            'data_path':            str((rsu_dir / f"{fid}_{ck}.png")),
                            "type":                    "camera",
                            # "sample_data_token":       f"{sc.name}_{rsu_dir.name}_{pid}_{ck}",
                            "sample_data_token":        f"{sc.name}_{rsu_dir.name}_{pid}_CAM_{local_idx}",
                            # raw extrinsics (camera frame itself)
                            "sensor2ego_translation":  t_c2e_s.tolist(),
                            "sensor2ego_rotation":     Quaternion(matrix=R_c2e_s).elements.tolist(),
                            "ego2global_translation":  t_e2g_s.tolist(),
                            "ego2global_rotation":     Quaternion(matrix=R_e2g_s).elements.tolist(),
                            "timestamp":               int(pid),
                            # composed transform into *key-frame* Top-LiDAR
                            "sensor2lidar_rotation":   R_tot.tolist(),
                            "sensor2lidar_translation":t_tot.tolist(),
                            # intrinsic matrix for BEVFormer's image backbone
                            "cam_intrinsic":           np.array(cd_s["intrinsic"]).tolist(),
                        })

                info["sweeps"] = sweeps
                infos.append(info)

    os.makedirs(out_path, exist_ok=True)
    out = dict(infos=infos, metadata={'version':version})
    mmcv.dump(out, Path(out_path)/f"v2xset_infos_temporal_{split}.pkl")
    print(f"→ saved {split}: {len(infos)} samples @ {out_path}")

# ─── STEP 2: convert VAL PKL to NuScenes JSONs ─────────────────────────────────
def save_json(name, data, OUT_ROOT):
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(Path(OUT_ROOT)/name, 'w') as f:
        json.dump(data, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x,'tolist') else x)
    print("✅", name)

def find_existing_image(data_path):
    """
    If data_path exists return it.
    Otherwise, try +1 and –1 frame IDs.
    """
    p = Path(data_path)
    if p.exists():
        return str(p)

    stem, suffix = p.stem, p.suffix           # e.g. "000037_camera0", ".png"
    # split number from the rest
    m = re.match(r"(\d+)(_camera\d+)", stem)
    if not m:
        raise FileNotFoundError(f"Cannot parse frame index from {data_path!r}")
    idx_str, tail = m.groups()
    idx = int(idx_str)

    # try +1 then –1
    for delta in (1, -1):
        new_idx = idx + delta
        new_name = f"{new_idx:06d}{tail}{suffix}"
        candidate = p.with_name(new_name)
        if candidate.exists():
            return str(candidate)

    # nothing found
    raise FileNotFoundError(f"No image found at {data_path!r}, nor ±1 frame")

def pkl2json(PKL_ROOT, VALID_CLASSES, OUT_ROOT):
    pkl = mmcv.load(Path(PKL_ROOT)/"v2xset_infos_temporal_all.pkl")
    infos = pkl['infos']
    info_map = {i['token']: i for i in infos}

    sample, sample_data, sample_ann = [], [], []
    calib, inst, cat, ego, sen, log, scene, vis = ([] for _ in range(8))
    ct = 0
    prev = {}

    # Create categories for all valid classes
    cat = []
    for class_name in VALID_CLASSES:
        cat.append({
            'token': class_name,
            'name': class_name
        })
    
    # Add required sensors (including LIDAR_TOP)
    sen.extend([
        {'token':'LIDAR_TOP','channel':'LIDAR_TOP','modality':'lidar','mount_pose_token':'LIDAR_TOP'},
        {'token':'CAM_FRONT','channel':'CAM_FRONT','modality':'camera','mount_pose_token':'CAM_FRONT'},
        {'token':'CAM_FRONT_RIGHT','channel':'CAM_FRONT_RIGHT','modality':'camera','mount_pose_token':'CAM_FRONT_RIGHT'},
        {'token':'CAM_FRONT_LEFT','channel':'CAM_FRONT_LEFT','modality':'camera','mount_pose_token':'CAM_FRONT_LEFT'},
        {'token':'CAM_BACK','channel':'CAM_BACK','modality':'camera','mount_pose_token':'CAM_BACK'},
    ])

    # Add calibration for LIDAR_TOP
    calib.append({
        'token': 'LIDAR_TOP_calib',
        'sensor_token': 'LIDAR_TOP',
        'translation': [0, 0, 0],  # Lidar at ego center
        'rotation': [1, 0, 0, 0]   # Identity rotation
    })

    # build samples & sample_data & calibrated_sensor & annotations & instances
    for e in infos:
        tk, sc = e['token'], e['scene_token']
        sm = {
            'token': tk, 'scene_token': sc, 'timestamp': e['timestamp'],
            'prev': prev.get(sc, ""), 'next': "",
            'data': {
                'LIDAR_TOP': f"{tk}_LIDAR"  # Add lidar reference
            }, 
            'anns': [], 
            'is_key_frame': True
        }
        prev[sc] = tk
        sample.append(sm)
        
        # Add lidar sample data
        sample_data.append({
            'is_key_frame': True,
            'token': f"{tk}_LIDAR",
            'sample_token': tk,
            'ego_pose_token': tk,
            'calibrated_sensor_token': 'LIDAR_TOP_calib',
            'filename': '',  # Empty since we don't have actual lidar data
            'timestamp': e['timestamp'],
            'sensor_modality': 'lidar'
        })

        # cameras
        for ci in range(4):
            cn = f"CAM_{ci}"
            c = e['cams'].get(cn)
            if not c: continue
            
            # Map V2X cameras to nuScenes camera names
            nu_cam_name = {
                0: 'CAM_FRONT',
                1: 'CAM_FRONT_RIGHT',
                2: 'CAM_BACK',
                3: 'CAM_FRONT_LEFT'
            }.get(ci, f"CAM_{ci}")
            
            ctok = f"{tk}_{nu_cam_name}"
            sm['data'][nu_cam_name] = ctok

            # new
            img_path = find_existing_image(c['data_path'])
            with Image.open(img_path) as img:
                width, height = img.size
            #with Image.open(c['data_path']) as img:
                #width, height = img.size
            
            sample_data.append({
                'is_key_frame': True, 
                'token': ctok, 
                'sample_token': tk,
                'ego_pose_token': tk, 
                'calibrated_sensor_token': ctok,
                'filename': c['data_path'], 
                'timestamp': c['timestamp'],
                'sensor_modality': 'camera', 
                'camera_intrinsic': c['cam_intrinsic'].tolist(),
                'width': width, 
                'height': height  
            })
            
            R_s2e = c["sensor2lidar_rotation"]      # 3×3
            t_s2e = c['sensor2lidar_translation']

            calib.append({
                'token':            ctok,
                'sensor_token':     nu_cam_name,
                'translation':      t_s2e,           # sensor→ego
                'rotation':         Quaternion(matrix=np.array(R_s2e)).elements.tolist(),
                'camera_intrinsic': c['cam_intrinsic'].tolist()
            })

        # annotations + instances
        for j, (bb, class_name) in enumerate(zip(e['gt_boxes'], e['gt_names'])):
            atok = f"{tk}_ann_{j}"
            sample_ann.append({
                'token': atok, 
                'sample_token': tk,
                'translation': bb[:3].tolist(), 
                'size': bb[3:6].tolist(),
                'rotation': Quaternion(axis=[0, 0, 1], radians=bb[6]).elements.tolist(),
                'category_name': class_name,  # Use the mapped class name
                'instance_token': f'instance_{ct}', 
                'attribute_tokens':[],
                'visibility_token': f"{atok}_vis",
                'velocity': e['gt_velocity'][j].tolist()
            })
            sm['anns'].append(atok)
            inst.append({
                'token': f'instance_{ct}', 
                'category_token': class_name  # Use the mapped class name
            })
            vis.append({'token': f"{atok}_vis", 'name':'default', 'value':1.0})
            ct += 1

    # ─── link prev & next pointers ─────────────────────────────────────────────
    groups = defaultdict(list)
    
    # build maps from each sample to its sample.prev/sample.next
    sample_prev = {s['token']: s['prev'] for s in sample}
    sample_next = {s['token']: s['next'] for s in sample}

    for ann in sample_ann:
        # extract the index (the "_ann_24" part) so we can reconstruct
        ann_idx = ann['token'].rsplit('_', 1)[1]

        # PREV
        prev_s = sample_prev[ann['sample_token']]
        if prev_s:
            ann['prev'] = f"{prev_s}_ann_{ann_idx}"
        else:
            ann['prev'] = ""    # no predecessor, empty by convention

        # NEXT (if you want it)
        next_s = sample_next[ann['sample_token']]
        if next_s:
            ann['next']  = f"{next_s}_ann_{ann_idx}"
        else:
            ann['next'] = ""

    # scene, ego, visibility, sensor
    for sc,tl in groups.items():
        scene.append({
            'token': sc, 'name': sc, 'log_token': 'default_log',
            'first_sample_token': tl[0], 'last_sample_token': tl[-1],
            'nbr_samples': len(tl)
        })
    for sm in sample:
        ego.append({
            'token': sm['token'],
            'translation': info_map[sm['token']]['ego2global_translation'],
            'rotation':    info_map[sm['token']]['ego2global_rotation'],
            'timestamp':  sm['timestamp']
        })

    # write JSONs
    save_json('category.json',       cat, OUT_ROOT)
    save_json('log.json',            log, OUT_ROOT)
    save_json('scene.json',          scene, OUT_ROOT)
    save_json('sample.json',         sample, OUT_ROOT)
    save_json('sample_data.json',    sample_data, OUT_ROOT)
    save_json('sample_annotation.json', sample_ann, OUT_ROOT)
    save_json('instance.json',       inst, OUT_ROOT)
    save_json('calibrated_sensor.json', calib, OUT_ROOT)
    save_json('sensor.json',         sen, OUT_ROOT)
    save_json('ego_pose.json',       ego, OUT_ROOT)
    save_json('visibility.json',     vis, OUT_ROOT)
    save_json('attribute.json',      [], OUT_ROOT)
    seen = {}
    for e in infos:
        tok = e['scene_token']                   # e.g. "town01_-1"
        if tok not in seen:
            seen[tok] = e['map_path']            # the .npy you saved

    map_entries = []
    for i, (scene_tok, map_path) in enumerate(seen.items()):
        map_entries.append({
            "token":        f"v2xset_map_{i}",    # unique map token
            "filename":     map_path,            # e.g. "/home/.../bev_label_map_200.npy"
            "log_tokens":   [scene_tok]          # link it back to that RSU's log
        })

    save_json('map.json', map_entries, OUT_ROOT)

    # # Write splits/val.json
    # splits_dir = Path(OUT_ROOT) / "splits"
    # splits_dir.mkdir(parents=True, exist_ok=True)
    # val_tokens = [s["token"] for s in sample]
    # with open(splits_dir / "val.json", "w") as f:
    #     json.dump(val_tokens, f, indent=2)
    # print(f"✅ Wrote {len(val_tokens)} tokens to {splits_dir/'val.json'}")

    print(f"\n🎉 JSONs for VAL written under:\n    {OUT_ROOT}\n")

def merge_nuscenes_tables(dataroot: str, version):
    """
    Reads per-table NuScenes JSONs from `dataroot/version/` and writes a single
    `dataroot/version.json` with all of them merged under the standard keys.
    """
    table_dir = Path(dataroot) / version
    output_path = Path(dataroot) / f"{version}.json"

    # Map filenames to top-level JSON keys
    key_map = {
        "sample.json":           "samples",
        "sample_data.json":      "sample_data",
        "ego_pose.json":         "ego_pose",
        "calibrated_sensor.json": "calibrated_sensor",
        "sensor.json":           "sensor",
        "attribute.json":        "attribute",
        "category.json":         "category",
        "instance.json":         "instance",
        "visibility.json":       "visibility",
        "map.json":              "maps",
        "log.json":              "logs",
        "scene.json":            "scenes",
        "sample_annotation.json": "annotations",
    }

    merged = {}
    # Load each individual JSON table
    for fname, top_key in key_map.items():
        path = table_dir / fname
        if path.exists():
            merged[top_key] = json.loads(path.read_text())
        else:
            print(f"Warning: {fname} not found in {table_dir}")

    # Load splits (e.g. splits/val.json → merged['splits']['val'])
    splits_dir = table_dir / "splits"
    if splits_dir.exists():
        merged["splits"] = {}
        for split_file in splits_dir.glob("*.json"):
            merged["splits"][split_file.stem] = json.loads(split_file.read_text())

    # Ensure output directory exists and write the merged JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2))
    print(f"✅ Created merged NuScenes JSON at:\n    {output_path}")
    
# ─── rotate full 200×200 raster by −ego yaw ──────────────────────────────────
def rotate_full_bev(bev_200, ego_yaw_rad):
    """Rotate the 200×200 map so that +X points along the sensor heading."""
    M = cv2.getRotationMatrix2D((100, 100), -np.degrees(ego_yaw_rad), 1.0)
    return cv2.warpAffine(bev_200, M, (200, 200), flags=cv2.INTER_NEAREST)


def create_all(OUT_ROOT, data_roots,PKL_ROOT, VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, USED_PERCENT):
    auto_split_pkls(data_roots,PKL_ROOT, VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, USED_PERCENT)
    pkl2json(PKL_ROOT, VALID_CLASSES, OUT_ROOT)
    merge_nuscenes_tables(OUT_ROOT, version=version)

def append_data_all(new_data_roots: list,
        PREVIOUS_PKL_ROOT,
        PREVIOUS_JSON_ROOT,
        TEMP_PKL_ROOT,
        NEW_PKL_ROOT,
        OUT_JSON_ROOT,
        VERSION,
        VALID_CLASSES,
        SPLITS,
        MAP_OUT,
        XODR_ROOT,
        MAX_SWEEPS,
        OBJECT_CLASS_MAPPING,
        used_percent: float = 1.0):
    """
    1) Generate PKLs for new_data_roots into TEMP_PKL_ROOT.
    2) Merge TEMP PKLs with existing splits in PREVIOUS_PKL_ROOT → NEW_PKL_ROOT.
    3) Regenerate JSON tables from merged 'all' PKL into OUT_JSON_ROOT.
    4) Copy over previous JSON tables and merge into final nuScenes JSON at OUT_JSON_ROOT/VERSION.json.
    """
    # 1) Create temporary PKLs for new data
    auto_split_pkls(
        data_roots=new_data_roots,
        PKL_ROOT=TEMP_PKL_ROOT,
        VALID_CLASSES=VALID_CLASSES,
        OBJECT_CLASS_MAPPING=OBJECT_CLASS_MAPPING,
        MAP_OUT=MAP_OUT,
        XODR_ROOT=XODR_ROOT,
        MAX_SWEEPS=MAX_SWEEPS,
        version=VERSION,
        USED_PERCENT=used_percent
    )

    # 2) Merge PKLs for each split
    os.makedirs(NEW_PKL_ROOT, exist_ok=True)
    for split in SPLITS:
        prev_pkl = Path(PREVIOUS_PKL_ROOT) / f"v2xset_infos_temporal_{split}.pkl"
        new_pkl  = Path(TEMP_PKL_ROOT)    / f"v2xset_infos_temporal_{split}.pkl"
        out_pkl  = Path(NEW_PKL_ROOT)     / f"v2xset_infos_temporal_{split}.pkl"

        data_prev = mmcv.load(str(prev_pkl)) if prev_pkl.exists() else {'infos': []}
        data_new  = mmcv.load(str(new_pkl))
        data_prev['infos'].extend(data_new['infos'])
        data_prev['metadata'] = data_new.get('metadata', data_prev.get('metadata', {}))
        mmcv.dump(data_prev, str(out_pkl))
        print(f"✅ Merged '{split}' → {out_pkl}")

    # 3) Regenerate JSON tables from merged 'all'
    pkl2json(str(NEW_PKL_ROOT), VALID_CLASSES, OUT_JSON_ROOT)

    # 4) Copy previous JSON tables and merge final JSON
    prev_json_dir = Path(PREVIOUS_JSON_ROOT) / VERSION
    curr_json_dir = Path(OUT_JSON_ROOT)        / VERSION
    os.makedirs(curr_json_dir, exist_ok=True)
    for tbl in prev_json_dir.glob('*.json'):
        dest = curr_json_dir / tbl.name
        if not dest.exists():
            dest.write_text(tbl.read_text())

    merge_nuscenes_tables(OUT_JSON_ROOT, VERSION)
    print("🎉 Append & regenerate complete.")
    
def append_data_trainval(new_data_root_train,
                         new_data_root_val, 
                        new_data_root_test,
                        PREVIOUS_PKL_ROOT,
                        TEMP_PKL_ROOT,
                        NEW_PKL_ROOT,
                        OUT_JSON_ROOT,
                        VERSION,
                        VALID_CLASSES,
                        SPLITS,
                        MAP_OUT,
                        XODR_ROOT,
                        MAX_SWEEPS,
                        OBJECT_CLASS_MAPPING):
    """
    1) Generate PKLs for new_data_roots into TEMP_PKL_ROOT.
    2) Merge TEMP PKLs with existing splits in PREVIOUS_PKL_ROOT → NEW_PKL_ROOT.
    3) Regenerate JSON tables from merged 'all' PKL into OUT_JSON_ROOT.
    4) Copy over previous JSON tables and merge into final nuScenes JSON at OUT_JSON_ROOT/VERSION.json.
    """
    # 1) Create temporary PKLs for new data
    # This bypasses auto_split_pkls
    train_scenes = sorted([p for p in Path(new_data_root_train).iterdir() if p.is_dir()])
    val_scenes   = sorted([p for p in Path(new_data_root_val).iterdir()   if p.is_dir()])

    create_v2xset_infos(train_scenes, TEMP_PKL_ROOT, 'train',
                        VALID_CLASSES, OBJECT_CLASS_MAPPING,
                        MAP_OUT, XODR_ROOT, MAX_SWEEPS, VERSION,
                        use_ego_frame=True)
    create_v2xset_infos(val_scenes, TEMP_PKL_ROOT, 'val',
                        VALID_CLASSES, OBJECT_CLASS_MAPPING,
                        MAP_OUT, XODR_ROOT, MAX_SWEEPS, VERSION,
                        use_ego_frame=False)
    
    
    #Same test and val
    val_pkl  = os.path.join(TEMP_PKL_ROOT, 'v2xset_infos_temporal_val.pkl')
    test_pkl = os.path.join(TEMP_PKL_ROOT, 'v2xset_infos_temporal_test.pkl')
    shutil.copyfile(val_pkl, test_pkl)
    
    all_scenes = train_scenes + val_scenes
    create_v2xset_infos(
        all_scenes,
        TEMP_PKL_ROOT,
        'all',
        VALID_CLASSES,
        OBJECT_CLASS_MAPPING,
        MAP_OUT,
        XODR_ROOT,
        MAX_SWEEPS,
        VERSION,
        use_ego_frame=False,
    )

    # 2) Merge PKLs for each split
    os.makedirs(NEW_PKL_ROOT, exist_ok=True)
    for split in SPLITS:
        prev_pkl = Path(PREVIOUS_PKL_ROOT) / f"v2xset_infos_temporal_{split}.pkl"
        new_pkl  = Path(TEMP_PKL_ROOT)    / f"v2xset_infos_temporal_{split}.pkl"
        out_pkl  = Path(NEW_PKL_ROOT)     / f"v2xset_infos_temporal_{split}.pkl"

        data_prev = mmcv.load(str(prev_pkl)) if prev_pkl.exists() else {'infos': []}
        data_new  = mmcv.load(str(new_pkl))
        data_prev['infos'].extend(data_new['infos'])
        data_prev['metadata'] = data_new.get('metadata', data_prev.get('metadata', {}))
        mmcv.dump(data_prev, str(out_pkl))
        print(f"✅ Merged '{split}' → {out_pkl}")

    # 3) Regenerate JSON tables from merged 'all'
    pkl2json(str(NEW_PKL_ROOT), VALID_CLASSES, Path(OUT_JSON_ROOT)/ VERSION)

    merge_nuscenes_tables(OUT_JSON_ROOT, VERSION)
    print("🎉 Append & regenerate complete.")


def create_data(new_data_root_train,
                        new_data_root_val, 
                        new_data_root_test,
                        PKL_ROOT,
                        OUT_JSON_ROOT,
                        VERSION,
                        VALID_CLASSES,
                        MAP_OUT,
                        XODR_ROOT,
                        MAX_SWEEPS,
                        OBJECT_CLASS_MAPPING):
    """
    1) Generate PKLs for new_data_roots into TEMP_PKL_ROOT.
    2) Merge TEMP PKLs with existing splits in PREVIOUS_PKL_ROOT → NEW_PKL_ROOT.
    3) Regenerate JSON tables from merged 'all' PKL into OUT_JSON_ROOT.
    4) Copy over previous JSON tables and merge into final nuScenes JSON at OUT_JSON_ROOT/VERSION.json.
    """
    # 1) Create temporary PKLs for new data
    # This bypasses auto_split_pkls
    train_scenes = sorted([p for p in Path(new_data_root_train).iterdir() if p.is_dir()])
    val_scenes   = sorted([p for p in Path(new_data_root_val).iterdir()   if p.is_dir()])
    test_scenes   = sorted([p for p in Path(new_data_root_test).iterdir()   if p.is_dir()])

    create_v2xset_infos(train_scenes, PKL_ROOT, 'train',
                        VALID_CLASSES, OBJECT_CLASS_MAPPING,
                        MAP_OUT, XODR_ROOT, MAX_SWEEPS, VERSION,
                        use_ego_frame=True)
    create_v2xset_infos(val_scenes, PKL_ROOT, 'val',
                        VALID_CLASSES, OBJECT_CLASS_MAPPING,
                        MAP_OUT, XODR_ROOT, MAX_SWEEPS, VERSION,
                        use_ego_frame=False)
    create_v2xset_infos(test_scenes, PKL_ROOT, 'test',
                        VALID_CLASSES, OBJECT_CLASS_MAPPING,
                        MAP_OUT, XODR_ROOT, MAX_SWEEPS, VERSION,
                        use_ego_frame=False)
    
    all_scenes = train_scenes + val_scenes + test_scenes
    create_v2xset_infos(
        all_scenes,
        PKL_ROOT,
        'all',
        VALID_CLASSES,
        OBJECT_CLASS_MAPPING,
        MAP_OUT,
        XODR_ROOT,
        MAX_SWEEPS,
        VERSION,
        use_ego_frame=False,
    )

    # 3) Regenerate JSON tables from merged 'all'
    pkl2json(str(PKL_ROOT), VALID_CLASSES, Path(OUT_JSON_ROOT)/ VERSION)

    merge_nuscenes_tables(OUT_JSON_ROOT, VERSION)
    print("🎉 Generate complete.")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # # ========================================================================================================================================================
    #=================== Case 1: Create data fro all: ===================
    # data_roots = [
    #         "/data3/unseen_scenarios",
    #         "/data4/multi-cam-new/1cam",
    #         "/data4/multi-cam-new/2cam",
    #         "/data4/multi-cam-new/3cam",
    #         "/data4/multi-cam-new/4cam",
    #     ]
    # PKL_ROOT  = "/home/yun/MIC-BEV_Official/data/v2x_4cam_map_final_filtered" # where to dump PKLs 
    # OUT_ROOT  = "/data3/multi-cam-test"  
    # # for map
    # XODR_ROOT = Path("/data4/multi-cam-new/CARLA_map_origional")
    # MAP_OUT   = Path("/home/yun/MIC-BEV_Official/data/maps")
    # version = "v1.0-trainval"
    # MAX_SWEEPS = 2
    # USED_PERCENT = 1 #Value between 0 and 1

    # OBJECT_CLASS_MAPPING = {
    #     # Map from original class names to standardized names
    #     'vehicles': 'car',
    #     'cars': 'car',
    #     'trucks': 'truck',
    #     'cyclists': 'bicycle',
    #     'pedestrians': 'pedestrian',
    # }

    # # Define valid classes for the dataset
    # VALID_CLASSES = {
    #     'car',
    #     'truck',
    #     'bicycle',
    #     'pedestrian',
    # }
    # create_all(OUT_ROOT, data_roots,PKL_ROOT, VALID_CLASSES, OBJECT_CLASS_MAPPING, MAP_OUT, XODR_ROOT, MAX_SWEEPS, version, USED_PERCENT)
    
    # ========================================================================================================================================================
    # =================== Case 2: Append new whole data to previous: ===================
    # new_data_roots = [
    #         "/data4/multi-cam-new/1cam",
    #         "/data4/multi-cam-new/2cam",
    #         "/data4/multi-cam-new/3cam",
    #         "/data4/multi-cam-new/4cam",
    #     ]
    # PREVIOUS_PKL_ROOT = "/home/yun/MIC-BEV_Official/data/v2x_4cam_map_final_filtered" #Older pkl for append
    # TEMP_PKL_ROOT = "/home/yun/MIC-BEV_Official/data/temp_weather" #Where new generate data will be temporaryly saved
    # NEW_PKL_ROOT = "/home/yun/MIC-BEV_Official/data/v2x_4cam_map_weather"
    
    # PREVIOUS_JSON_ROOT = "/data4/multi-cam-json"
    # OUT_JSON_ROOT      = "/data4/multi-cam-json-weather"
    # VERSION            = "v1.0-trainval"
    # VALID_CLASSES      = { 'car', 'truck', 'bicycle', 'pedestrian' }
    # SPLITS             = ['train', 'val', 'test', 'all']
    # XODR_ROOT = Path("/data4/multi-cam-new/CARLA_map_origional")
    # MAP_OUT   = Path("/home/yun/MIC-BEV_Official/data/map")
    # MAX_SWEEPS = 2
    # OBJECT_CLASS_MAPPING = {
    #     # Map from original class names to standardized names
    #     'vehicles': 'car',
    #     'cars': 'car',
    #     'trucks': 'truck',
    #     'cyclists': 'bicycle',
    #     'pedestrians': 'pedestrian',
    # }
    # USED_PERCENT = 1
    
    # append_data(new_data_roots, PREVIOUS_PKL_ROOT, PREVIOUS_JSON_ROOT,
    #     TEMP_PKL_ROOT,
    #     NEW_PKL_ROOT,
    #     OUT_JSON_ROOT,
    #     VERSION,
    #     VALID_CLASSES,
    #     SPLITS,
    #     MAP_OUT,
    #     XODR_ROOT,
    #     MAX_SWEEPS,
    #     OBJECT_CLASS_MAPPING,
    #     USED_PERCENT: float = 1.0):
    # ===========================================================================================================
    #=============================Case 3: append splited train val to existing one ====================================================
    # new_data_root_train = "/data3/yun/M2I_dataset/M2I_split_dataset/train"
    # new_data_root_val = "/data3/yun/M2I_dataset/M2I_split_dataset/val"
    # new_data_root_test = "/data3/yun/M2I_dataset/M2I_split_dataset/test"
    # PREVIOUS_PKL_ROOT = None
    # TEMP_PKL_ROOT = "/data3/yun/M2I_dataset/M2I_pkl" #Where new generate data will be temporaryly saved
    # NEW_PKL_ROOT = None
    
    # OUT_JSON_ROOT      = "/data3/yun/M2I_dataset/M2I_json"
    # VERSION            = "v1.0-trainval"
    # VALID_CLASSES      = { 'car', 'truck', 'bicycle', 'pedestrian' }
    # SPLITS             = ['train', 'val', 'test', 'all']
    # XODR_ROOT = Path("/data3/yun/M2I_dataset/CARLA_map_origional")
    # MAP_OUT   = Path("/data3/yun/M2I_dataset/maps")
    # MAX_SWEEPS = 2
    # OBJECT_CLASS_MAPPING = {
    #     # Map from original class names to standardized names
    #     'vehicles': 'car',
    #     'cars': 'car',
    #     'trucks': 'truck',
    #     'cyclists': 'bicycle',
    #     'pedestrians': 'pedestrian',
    # }
    
    # append_data_trainval(new_data_root_train,
    #                      new_data_root_val, 
    #                      PREVIOUS_PKL_ROOT,
    #                      TEMP_PKL_ROOT,
    #                      NEW_PKL_ROOT,
    #                      OUT_JSON_ROOT,
    #                      VERSION,
    #                      VALID_CLASSES,
    #                      SPLITS,
    #                      MAP_OUT,
    #                      XODR_ROOT,
    #                      MAX_SWEEPS,
    #                      OBJECT_CLASS_MAPPING)
    
    #=========Append unseen scenarios to all scenarios=========
    # prev_pkl = Path("/home/yun/MIC-BEV_Official/data/v2x_4cam_map_weather/v2xset_infos_temporal_all.pkl")
    # new_pkl  = Path("/home/yun/MIC-BEV_Official/data/unseen_scenarios/v2xset_infos_temporal_all.pkl")
    # out_pkl  = Path("/home/yun/MIC-BEV_Official/data/all_scenarios/v2xset_infos_temporal_all.pkl")

    # data_prev = mmcv.load(str(prev_pkl)) if prev_pkl.exists() else {'infos': []}
    # data_new  = mmcv.load(str(new_pkl))
    # data_prev['infos'].extend(data_new['infos'])
    # data_prev['metadata'] = data_new.get('metadata', data_prev.get('metadata', {}))
    # mmcv.dump(data_prev, str(out_pkl))
    #=======================Create split dataset=======================
    new_data_root_train = "/data3/yun/M2I_dataset/M2I_split_dataset/train"
    new_data_root_val = "/data3/yun/M2I_dataset/M2I_split_dataset/val"
    new_data_root_test = "/data3/yun/M2I_dataset/M2I_split_dataset/test"
    PKL_ROOT = "/data3/yun/M2I_dataset/M2I_pkl" #Where new generate data will be temporaryly saved
    
    OUT_JSON_ROOT      = "/data3/yun/M2I_dataset/M2I_json"
    VERSION            = "v1.0-trainval"
    VALID_CLASSES      = { 'car', 'truck', 'bicycle', 'pedestrian' }
    SPLITS             = ['train', 'val', 'test', 'all']
    XODR_ROOT = Path("/data3/yun/M2I_dataset/CARLA_map_origional")
    MAP_OUT   = Path("/data3/yun/M2I_dataset/maps")
    MAX_SWEEPS = 2
    OBJECT_CLASS_MAPPING = {
        # Map from original class names to standardized names
        'vehicles': 'car',
        'cars': 'car',
        'trucks': 'truck',
        'cyclists': 'bicycle',
        'pedestrians': 'pedestrian',
    }
    
    create_data(new_data_root_train,
                new_data_root_val, 
                new_data_root_test,
                PKL_ROOT,
                OUT_JSON_ROOT,
                VERSION,
                VALID_CLASSES,
                MAP_OUT,
                XODR_ROOT,
                MAX_SWEEPS,
                OBJECT_CLASS_MAPPING)