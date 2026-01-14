

import mmcv
import os
from nuscenes.nuscenes import NuScenes
from PIL import Image
from typing import List, Dict, Any
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from tools.analysis_tools.box import Boxnew
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from copy import deepcopy
import sys
import json
from matplotlib.colors import ListedColormap

cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK',
 'CAM_FRONT_LEFT']

palette = np.array([
    [191, 216, 109],   # 0 background – green
    [211, 211, 211],   # 1 driving    – light gray
    [240, 230, 180],   # 2 sidewalk   – beige
    [229, 204, 255],   # 3 crosswalk  – purple
    [128,  128,  128],   # 4 shoulder   – dark gray
    [169, 169, 169],   # 5 border     – dark gray
    [  0, 188, 212],   # 6 parking    – teal
], dtype=np.uint8)


def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     with_map: bool = False,
                     nsweeps: int = 1,
                     conf_th: float = 0.2,
                     eval_range: float = 50,
                     verbose: bool = True,
                     display_legend: bool = False,
                     savepath: str = None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param display_legend: Whether to display GT and EST boxes legend on plot.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    
    # ─── underlay your BEV‐map.npy ────────────────────────────────────
    if with_map == 'True':
        scene_tok = nusc.get('sample', sample_token)['scene_token']
        map_file  = SCENE2MAP.get(scene_tok)
    else:
        map_file = None
    if map_file is not None and os.path.exists(map_file):
        bev_map = np.load(map_file)  # shape (200, 200)
        # Flip in Y direction
        # bev_map = np.flipud(bev_map)
        extent = (-51.2, 51.2, -51.2, 51.2)
        plt.figure(figsize=(9, 9))
        ax = plt.gca()
        custom_cmap = ListedColormap(palette / 255.0)
        ax.imshow(bev_map, cmap=custom_cmap, origin='lower', extent=extent, vmin=0, vmax=len(palette)-1)
    else:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.detection_score

    # # Get point cloud in lidar frame.
    # pc, _ = LidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    # _, ax = plt.subplots(1, 1, figsize=(9, 9))
    if 'ax' not in locals():
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # # Show point cloud.
    # points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range)
    # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(3), colors=('#3399FF',), linewidth=2, alpha=0.3)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            box.render(ax, view=np.eye(3), colors=('#FF6666',), linewidth=2, alpha=0.3)

    # Add legend.
    if display_legend:
        ax.legend(['GT', 'EST'], loc='upper right', labels=['GT', 'EST'],
                handles=[mpatches.Patch(color='g'), mpatches.Patch(color='b')])

        
    # Style to match draw_transparent_boxes visual
    ax.set_xlim(-eval_range, eval_range)
    ax.set_ylim(-eval_range, eval_range)

    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Grid + inward ticks
    ax.grid(True, color='gray', linestyle='--', linewidth=1)
    ax.tick_params(labelsize=10, direction='in', width=2, length=4, color='black')

    # Thicker plot frame
    for spine in ax.spines.values():
        spine.set_linewidth(3.0)

    # Adjust layout to show full frame (not tight on image)
    plt.tight_layout(pad=2)

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    plt.title(sample_token)
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, dpi=500, bbox_inches=None, pad_inches=0.1)
        plt.close()
    else:
        plt.tight_layout(pad=0)
        plt.show()
    

def boxes_to_sensor_bev(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box =  Boxnew(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out

def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box =  Boxnew(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out

def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    LidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)



def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose ac
    sd_record = nusc.get('sample_data', sample_data_token)
    if sd_record['sensor_modality'] != 'camera':
        print(f"Skipping non-camera sample_data {sample_data_token} with modality {sd_record['sensor_modality']}")
        return None, None, None
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        # print(f"GT Box [{box.name}]: {box.center}")
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic



def get_predicted_data(sample_data_token: str,
                       box_vis_level: BoxVisibility = BoxVisibility.ANY,
                       selected_anntokens=None,
                       use_flat_vehicle_coordinates: bool = False,
                       pred_anns=None
                       ):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    if sd_record['sensor_modality'] != 'camera':
        print(f"Skipping non-camera sample_data {sample_data_token} with modality {sd_record['sensor_modality']}")
        return None, None, None
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #    boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #    boxes = nusc.get_boxes(sample_data_token)
    boxes = pred_anns
    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        # print(f"PRED Box [{box.name}]: {box.center}")
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def lidiar_render(sample_token, with_map, data,out_path=None, conf_thresh=0.2):
    bbox_gt_list = []
    bbox_pred_list = []
    try:
        anns = nusc.get('sample', sample_token)['anns']
    except KeyError:
        print(f"Sample token {sample_token} not found in dataset, skipping...")
        return
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        #print(f"content {content}")
        try:
            bbox_gt_list.append(DetectionBox(
                sample_token=content['sample_token'],
                translation=tuple(content['translation']),
                size=tuple(content['size']),
                rotation=tuple(content['rotation']),
                velocity=nusc.box_velocity(content['token'])[:2],
                ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                else tuple(content['ego_translation']),
                num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                detection_name=content['category_name'],
                detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                attribute_name=''))
        except Exception as e:
            print('❌  GT box dropped:', content['token'], e),
    #print(f"lidiar_render bbox_gt_list {bbox_gt_list} ")

    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(DetectionBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            detection_name=content['detection_name'],
            detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
            attribute_name=content['attribute_name']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    bev_out = os.path.join(out_path, 'bev.png')
    visualize_sample(
        nusc,
        sample_token,
        gt_annotations,
        pred_annotations,
        with_map=with_map,
        conf_th= conf_thresh,
        savepath=bev_out,
        verbose=False,
        display_legend=False
    )



def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]

def render_sample_data(with_map, sample_token: str, pred_data=None, out_path: str = None, conf_thresh: float = 0.2):
    try:
        sample = nusc.get('sample', sample_token)
    except KeyError:
        print(f"Sample token {sample_token} not found in dataset, skipping...")
        return
    
    lidiar_render(sample_token, with_map, pred_data, out_path=out_path, conf_thresh=conf_thresh)
    cams = [c for c in ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK','CAM_FRONT_LEFT']
            if c in sample['data']]

    # prep folders
    for sub in ('pred','gt'):
        os.makedirs(os.path.join(out_path, sub), exist_ok=True)

    for cam in cams:
        token = sample['data'][cam]

        # ← build a list of nuScenes Box objects for your preds
        pred_boxes = [
            Box(r['translation'],
                r['size'],
                Quaternion(r['rotation']),
                name=r['detection_name'])
            for r in pred_data['results'][sample_token]
            if r['detection_score'] > conf_thresh
        ]

        # now call get_predicted_data with those Box instances
        data_path, boxes_pred, cam_intr = get_predicted_data(
            token,
            box_vis_level=BoxVisibility.ANY,
            pred_anns=pred_boxes
        )
        _, boxes_gt, _ = get_sample_data(
            token,
            box_vis_level=BoxVisibility.ANY,
            selected_anntokens=sample['anns']
        )

        img = Image.open(data_path)
        W, H = img.size
        
        # compute figsize so that figsize* dpi = (W, H)
        dpi     = 100
        figsize = (W / dpi, H / dpi)

        # ——— prediction overlay ———
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.imshow(img)
        for box in boxes_pred:
            c = np.array(get_color(box.name)) / 255.0
            box.render(ax, view=cam_intr, normalize=True, colors=(c, c, c))

        ax.axis('off')
        # remove all margins
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            os.path.join(out_path, 'pred', f'{cam}.png'),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

        # ——— ground-truth overlay ———
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.imshow(img)
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0
            box.render(ax, view=cam_intr, normalize=True, colors=(c, c, c))

        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(
            os.path.join(out_path,'gt',f'{cam}.png'),
            dpi=dpi,
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close(fig)

if __name__ == '__main__':
    nusc = NuScenes(version='/data3/yun/M2I_dataset/M2I_json/v1.0-trainval', dataroot='/data3/yun/M2I_dataset/', verbose=True)
    # render_annotation('7603b030b42a4b1caa8c443ccc1a7d52')
    #bevformer_results = mmcv.load('test/bevformer_base_inf_higherlr/Tue_May_13_14_03_22_2025/pts_bbox/results_nusc.json')

    if len(sys.argv) < 4:
        print("Usage: python script.py <result_json_path> <save_dir> <with_map>")
        print("  result_json_path: path to results_nusc.json file")
        print("  save_dir: directory to save visualizations")
        print("  with_map: True or False to include map rendering")
        sys.exit(1)

    result_path = sys.argv[1]
    save_dir = sys.argv[2]
    with_map = sys.argv[3]
    # result_path = 'test/bevformer_base_inf_betternorm/Wed_May_14_10_30_04_2025/pts_bbox/results_nusc.json'

    bevformer_results = mmcv.load(result_path)

    sample_token_list = list(bevformer_results['results'].keys())

    if with_map == 'True':
        #——— load map.json once ———
        MAP_JSON = "/data4/multi-cam-json-weather/v1.0-trainval/map.json"   # adjust path
        with open(MAP_JSON, 'r') as f:
            _map_entries = json.load(f)
        # build scene_token → filename lookup
        SCENE2MAP = {
            entry['log_tokens'][0]: entry['filename']
            for entry in _map_entries
        }
        
    for id in range(0, 1300):
    # for id in range(0, 500):
        sample_token = sample_token_list[id]
        out_path = os.path.join(save_dir, sample_token)
        render_sample_data(with_map, sample_token, pred_data=bevformer_results, out_path=out_path, conf_thresh=0.2)
