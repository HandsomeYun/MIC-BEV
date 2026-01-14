#!/usr/bin/env python3
import os
import re
import yaml
import json
import mmcv
import numpy as np
from PIL import Image
import random
import shutil
from tqdm import tqdm
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from collections import OrderedDict, defaultdict


# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT = "/data2/mcbev-testdata/v2xset-inf"    # raw v2x subfolders
PKL_ROOT  = "./data/v2x_inf"                            # where to dump PKLs
OUT_ROOT  = "/data2/mcbev-testdata/v2xset-inf/v1.0-trainval"  
MAX_SWEEPS = 0
USED_PERCENT = 1

# ─── UTILITIES ─────────────────────────────────────────────────────────────────
def load_yaml(file):
    """Load YAML preserving floats."""
    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(r'''^(?:
             [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    data = yaml.load(stream, Loader=loader)
    if "yaml_parser" in data:
        data = eval(data["yaml_parser"])(data)
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
    x, y, z, roll, yaw, pitch = p
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

def auto_split_pkls():
    scenes = sorted([d for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
    random.shuffle(scenes)
    n_total = len(scenes)

    # Only keep a subset of scenes
    n_used = int(USED_PERCENT * n_total)
    scenes = scenes[:n_used]

    # Split into train/test
    ntr = int(0.8 * n_used)  # 70% train, 30% test (you can change)

    used_percent = (n_used / n_total) * 100
    print(f"[INFO] Using {used_percent:.1f}% of total {n_total} scenes (={n_used} scenes)")

    # Training data in ego frame
    create_v2xset_infos(DATA_ROOT, scenes[:ntr], PKL_ROOT, 'train', use_ego_frame=True)
    
    # # Val data in ego frame
    # create_v2xset_infos(DATA_ROOT, scenes[ntr:n_used], PKL_ROOT, 'val', use_ego_frame=False)

    # #Same test and val
    # val_pkl  = os.path.join(PKL_ROOT, 'v2xset_infos_temporal_val.pkl')
    # test_pkl = os.path.join(PKL_ROOT, 'v2xset_infos_temporal_test.pkl')
    # shutil.copyfile(val_pkl, test_pkl)

    # # # Test data in global frame
    # # create_v2xset_infos(DATA_ROOT, scenes[ntr:n_used], PKL_ROOT, 'global', use_ego_frame=False)
    # print("✅ splits ready\n")

# ─── STEP 1: split raw into PKLs ──────────────────────────────────────────────
def create_v2xset_infos(data_root, scenarios, out_path, split, use_ego_frame=False):
    infos = []
    for sc in tqdm(scenarios, desc=f"Building {split} infos"):
        rsu_dir = sc / "-1"
        if not rsu_dir.exists():
            continue
        yamls = sorted(rsu_dir.glob("*.yaml"))
        frame_ids = [y.stem for y in yamls]
        i = 0
        for i, fid in enumerate(frame_ids):
            if (i ==1):
                break
            i +=1
            fi = load_yaml(str(rsu_dir/f"{fid}.yaml"))
            pose_carla = fi["lidar_pose"] # degrees, CARLA world
            E2W_LH = ego2world_LH(pose_carla)      
            E2W_RH = left2right @ E2W_LH @ left2right     # world→ego, RH  ← store this
            rotation_quat = Quaternion(matrix=E2W_RH[:3, :3]).elements.tolist()
            translation_xyz = E2W_RH[:3, 3].tolist()
            info = {
                'token': f"{sc.name}_{fid}",
                'scene_token': sc.name,
                'prev': frame_ids[i-1] if i>0 else None,
                'next': frame_ids[i+1] if i<len(frame_ids)-1 else None,
                'cams': OrderedDict(),
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
            for ci in range(4):
                ck = f"camera{ci}"
                if ck not in fi: continue
                cd = fi[ck]
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
                
                info['cams'][f"CAM_{ci}"] = {
                    'data_path':                   str(Path(data_root)/sc.name/"-1"/f"{fid}_{ck}.png"),
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
            for obj in fi.get("vehicles",{}).values():
                center_carla = np.array(obj['center']) + np.array(obj['location'])
                # center_carla = np.array(obj['location']) 
                center_RH = [center_carla[0], -center_carla[1], center_carla[2]]
                w = obj['extent'][1] * 2
                l = obj['extent'][0] * 2
                h = obj['extent'][2] * 2
                extent = [w, l, h]

                # convert degrees → radians
                yaw_car = np.deg2rad(obj['angle'][1])
                yaw_nusc = -yaw_car
                # normalize to [-π, π)
                yaw_nusc = (yaw_nusc - np.pi) % (2*np.pi) - np.pi
                print(f"yaw_carla: {yaw_car}")
                print(f"yaw_nusc: {yaw_nusc}")

                yaw_local_second = -yaw_nusc - np.pi / 2
                yaw_local_second = (yaw_local_second+ np.pi) % (2 * np.pi) - np.pi #Normalize in [-π, π]  
                print(f"yaw_local_second: {yaw_local_second}")
                #=======================================
                if use_ego_frame:
                    W2E_RH = np.linalg.inv(E2W_RH)
                    center_ego = apply_trans(center_RH, W2E_RH)
                    box = center_ego.tolist() + extent + [yaw_local_second]
                else:
                    yaw_nusc = -yaw_car
                    # normalize to [-π, π)
                    yaw_nusc = (yaw_nusc - np.pi) % (2*np.pi) - np.pi
                    box = center_RH + extent + [yaw_nusc]
                
                boxes.append(box)
                names.append("car")
            
            info.update({
                'gt_boxes':     np.array(boxes),
                'gt_names':     np.array(names),
                'gt_velocity':  np.array(vels),
                'valid_flag':   np.ones(len(boxes),dtype=bool),
                'num_lidar_pts':np.ones(len(boxes)),
                'num_radar_pts':np.zeros(len(boxes)),
            })
            
            # sweeps
            sw = []
            for s in range(1, MAX_SWEEPS+1):
                if i-s < 0: break
                pid = frame_ids[i-s]
                if (rsu_dir/f"{pid}.yaml").exists():
                    sw.append({
                        'data_path': None,
                        'sensor2lidar_rotation': np.eye(3).tolist(),
                        'sensor2lidar_translation': [0,0,0],
                        'type': 'camera',
                        'timestamp': int(pid)
                    })
            info['sweeps'] = sw
            infos.append(info)

    os.makedirs(out_path, exist_ok=True)
    out = dict(infos=infos, metadata={'version':'v1.0-trainval'})
    mmcv.dump(out, Path(out_path)/f"v2xset_infos_temporal_{split}.pkl")
    print(f"→ saved {split}: {len(infos)} samples @ {out_path}")

# ─── STEP 2: convert VAL PKL to NuScenes JSONs ─────────────────────────────────
def save_json(name, data):
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(Path(OUT_ROOT)/name, 'w') as f:
        json.dump(data, f, indent=2,
                  default=lambda x: x.tolist() if hasattr(x,'tolist') else x)
    print("✅", name)

def pkl2json():
    pkl = mmcv.load(Path(PKL_ROOT)/"v2xset_infos_temporal_test.pkl")
    infos = pkl['infos']
    info_map = {i['token']: i for i in infos}

    sample, sample_data, sample_ann = [], [], []
    calib, inst, cat, ego, sen, log, scene, vis = ([] for _ in range(8))
    ct = 0
    prev = {}

    # one category & one log
    cat.append({'token':'car','name':'car'})
    log.append({'token':'default_log','logfile':'','date_captured':'2025-04-21'})

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
            img = mmcv.imread(c['data_path'], flag='unchanged')
            height, width = img.shape[:2]
            
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
        for j, bb in enumerate(e['gt_boxes']):
            atok = f"{tk}_ann_{j}"
            sample_ann.append({
                'token': atok, 
                'sample_token': tk,
                'translation': bb[:3].tolist(), 
                'size': bb[3:6].tolist(),
                'rotation': Quaternion(axis=[0, 0, 1], radians=bb[6]).elements.tolist(),
                'category_name':'car',
                'instance_token': f'instance_{ct}', 
                'attribute_tokens':[],
                'visibility_token': f"{atok}_vis"
            })
            sm['anns'].append(atok)
            inst.append({'token': f'instance_{ct}', 'category_token':'car'})
            vis.append({'token': f"{atok}_vis", 'name':'default', 'value':1.0})
            ct += 1

    # link next pointers
    groups = defaultdict(list)
    for sm in sample:
        groups[sm['scene_token']].append(sm['token'])
    for toklist in groups.values():
        for idx_scene in range(len(toklist) - 1):
            curr, nxt = toklist[idx_scene], toklist[idx_scene + 1]
            sample_idx = next(
                j for j, s in enumerate(sample) 
                if s['token'] == curr
            )
            sample[sample_idx]['next'] = nxt

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
    save_json('category.json',       cat)
    save_json('log.json',            log)
    save_json('scene.json',          scene)
    save_json('sample.json',         sample)
    save_json('sample_data.json',    sample_data)
    save_json('sample_annotation.json', sample_ann)
    save_json('instance.json',       inst)
    save_json('calibrated_sensor.json', calib)
    save_json('sensor.json',         sen)
    save_json('ego_pose.json',       ego)
    save_json('visibility.json',     vis)
    save_json('attribute.json',      [])  # none
    save_json('map.json',            [{
        "token":"v2xset_map_0",
        "filename":"",
        "log_tokens":[s['log_token'] for s in scene]
    }])

    # Write splits/val.json
    splits_dir = Path(OUT_ROOT) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    val_tokens = [s["token"] for s in sample]
    with open(splits_dir / "val.json", "w") as f:
        json.dump(val_tokens, f, indent=2)
    print(f"✅ Wrote {len(val_tokens)} tokens to {splits_dir/'val.json'}")

    print(f"\n🎉 JSONs for VAL written under:\n    {OUT_ROOT}\n")

def merge_nuscenes_tables(dataroot: str, version: str = "v1.0-trainval"):
    """
    Reads per-table NuScenes JSONs from `dataroot/version/` and writes a single
    `dataroot/version.json` with all of them merged under the standard keys.
    """
    table_dir = Path(dataroot) / version
    output_path = Path(dataroot) / f"{version}.json"

    # Map filenames to top‐level JSON keys
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

# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    auto_split_pkls()