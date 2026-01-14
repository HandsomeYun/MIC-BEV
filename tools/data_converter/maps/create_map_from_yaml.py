from pathlib import Path
import numpy as np
import yaml
from yaml import CLoader as Loader
import cv2

from tools.data_converter.maps.xodr_parser.parse_and_visualize import compute_filtered_pickles
from tools.data_converter.maps.map_converter import generate_bev_map

def rotate_full_bev(bev_200: np.ndarray, ego_yaw_rad: float) -> np.ndarray:
    """Rotate the 200×200 map so that +X points along the sensor heading."""
    M = cv2.getRotationMatrix2D((bev_200.shape[1] / 2, bev_200.shape[0] / 2),
                                -np.degrees(ego_yaw_rad), 1.0)
    return cv2.warpAffine(bev_200, M, bev_200.shape[:2][::-1], flags=cv2.INTER_NEAREST)


def _find_first_frame_yaml(scenario_dir: Path) -> Path:
    """
    Walk into any subdirectory (e.g. -1, -2, etc) and return the first *.yaml found.
    """
    for sub in sorted(scenario_dir.iterdir()):
        if sub.is_dir():
            yamls = sorted(sub.glob("*.yaml"))
            if yamls:
                return yamls[0]
    raise FileNotFoundError(f"No frame-level YAML found under {scenario_dir}")


def create_bev_from_yaml(data_protocol: Path,
                         xodr_root: Path,
                         out_root: Path,
                         range_x: float = 51.2,
                         range_y: float = 51.2,
                         downsample_size: int = 200) -> Path:
    """
    Read CARLA scenario `data_protocol.yaml`, find an ego pose, generate a north-up BEV map,
    rotate & flip to ego heading, and save under
      out_root/<town>/<x>,<y>/bev_label_map_{downsample_size}.npy

    Returns the final .npy path.
    """
    data = yaml.load(data_protocol.open('r'), Loader=Loader)
    town = data['world']['town']

    # 1) try to get lidar_pose directly
    pose = data.get('lidar_pose') or data.get('lidar_pose0')

    # 2) if missing, fall back to first per-frame yaml in the scenario folder
    if pose is None:
        scenario_dir = data_protocol.parent
        first_frame = _find_first_frame_yaml(scenario_dir)
        frame_data = yaml.load(first_frame.open('r'), Loader=Loader)
        pose = frame_data.get('lidar_pose') or frame_data.get('lidar_pose0')
        if pose is None:
            raise KeyError(f"No 'lidar_pose' or 'lidar_pose0' in {first_frame}")

    # extract center XY and yaw
    center_xy   = [pose[0], -pose[1]]
    ego_yaw_rad = -np.deg2rad(pose[4])

    # prepare output
    subdir  = out_root / town / f"{center_xy[0]:.2f},{center_xy[1]:.2f}"
    subdir.mkdir(parents=True, exist_ok=True)
    bev_file = subdir / f"bev_label_map_{downsample_size}.npy"
    if bev_file.exists():
        return bev_file

    # load xodr
    xodr = xodr_root / f"{town}.xodr"
    if not xodr.exists():
        raise FileNotFoundError(f"{xodr} not found")

    # filter & parse opendrive → pickles
    total_areas, crosswalks = compute_filtered_pickles(
        xodr,
        center_pose=center_xy,
        range_x=range_x, range_y=range_y,
        town_name=town,
        out_dir=None
    )

    # build raw north-up BEV
    generate_bev_map(
        total_areas, crosswalks,
        center_xy=center_xy,
        town_name=town,
        save_dir=str(subdir),
        downsample_size=downsample_size
    )

    # rotate & flip
    bev = np.load(bev_file)
    bev = rotate_full_bev(bev, ego_yaw_rad)
    bev = np.flipud(bev)
    np.save(bev_file, bev)

    return bev_file


if __name__ == "__main__":
    data_protocol = Path("/data4/multi-cam-new/3cam/"
                          "town07_intersection1_3cam_t_c_day_s47/"
                          "data_protocol.yaml")
    xodr_root  = Path("/data4/multi-cam-new/CARLA_map")
    out_root   = Path("/home/yun/MIC-BEV_Official/data/my_maps")

    npy_path = create_bev_from_yaml(data_protocol, xodr_root, out_root)
    print("Saved BEV map to:", npy_path)
