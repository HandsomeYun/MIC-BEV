from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion
import numpy as np
import mmcv
from functools import partial

# Load NuScenes
nusc = NuScenes(
    version='unseen_scenarios_json',
    dataroot='/data3/',
    verbose=True
)

# Save the original method before monkey-patching
original_get_sample_data = nusc.get_sample_data
sample = nusc.sample[50]

available_cams = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
valid_cams = [cam for cam in available_cams if cam in sample['data']]

for cam in valid_cams:
    cam_token = sample['data'][cam]
    _, boxes, _ = original_get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)

    # Apply inverse yaw rotation
    for box in boxes:
        yaw_nusc = box.orientation.yaw_pitch_roll[0]

    # Patch only for this render
    def _patched_get_sample_data(token, box_vis_level=BoxVisibility.ANY, selected_boxes=None):
        data_path, _, camera_intrinsic = original_get_sample_data(token, box_vis_level)
        return data_path, boxes, camera_intrinsic

    # Set custom color map
    nusc.colormap['car'] = (255, 158, 0)
    nusc.colormap['bicycle'] = (0, 158, 255)
    nusc.colormap['truck'] = (255, 0, 0)
    nusc.colormap['pedestrian'] = (100, 158, 0)

    # Monkey-patch and render
    nusc.get_sample_data = partial(_patched_get_sample_data, selected_boxes=boxes)
    nusc.render_sample_data(
        cam_token,
        with_anns=True,
        out_path=f"/home/yun/MIC-BEV_Official/data_dumping/gt_render/{cam}.png"
    )

# Restore original method
nusc.get_sample_data = original_get_sample_data
