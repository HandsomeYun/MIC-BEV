import argparse
import hashlib
import json
import os
import pickle
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.color_map import get_colormap

from tools.analysis_tools.box import Boxnew

MAP_PALETTE = np.array([
    [191, 216, 109], #Background
    [211, 211, 211], #Road
    [240, 230, 180], #Sidewalk
    [229, 204, 255], #Crosswalk
    [128, 128, 128], #Shoulder
    [33, 64, 43], #Sidewalk
    [0, 188, 212], #Parking
], dtype=np.uint8)
def format_bev_axes(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    x_center = (x_min + x_max) * 0.5
    y_center = (y_min + y_max) * 0.5
    x_extent = max(x_center - x_min, x_max - x_center)
    y_extent = max(y_center - y_min, y_max - y_center)
    x_offsets = np.arange(
        np.ceil(-x_extent / 10.0) * 10.0,
        x_extent + 1e-6,
        10.0)
    y_offsets = np.arange(
        np.ceil(-y_extent / 10.0) * 10.0,
        y_extent + 1e-6,
        10.0)
    ax.set_xticks(x_center + x_offsets)
    ax.set_yticks(y_center + y_offsets)
    ax.set_xticklabels([f'{value:g}' for value in x_offsets])
    ax.set_yticklabels([f'{value:g}' for value in y_offsets])
    x_minor_offsets = np.arange(
        np.ceil(-x_extent / 0.5) * 0.5,
        x_extent + 1e-6,
        0.5)
    y_minor_offsets = np.arange(
        np.ceil(-y_extent / 0.5) * 0.5,
        y_extent + 1e-6,
        0.5)
    ax.set_xticks(x_center + x_minor_offsets, minor=True)
    ax.set_yticks(y_center + y_minor_offsets, minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.25, alpha=0.55)
    ax.grid(which='major', color='lightgray', linewidth=0.45, alpha=0.75)
    ax.tick_params(axis='both', which='major', labelsize=8, length=2)
    ax.tick_params(axis='both', which='minor', length=0)


def detection_color_rgb(category_name: str):
    cmap = getattr(detection_color_rgb, '_cmap', None)
    if cmap is None:
        cmap = get_colormap()
        detection_color_rgb._cmap = cmap

    if category_name == 'bicycle':
        return cmap['vehicle.bicycle']
    if category_name == 'construction_vehicle':
        return cmap['vehicle.construction']
    if category_name == 'traffic_cone':
        return cmap['movable_object.trafficcone']
    for key in cmap.keys():
        if category_name in key:
            return cmap[key]
    return (0, 0, 0)


def mpl_color(rgb):
    """Matplotlib accepts 0–1 floats; matches visual.py’s c=np.array(...) / 255."""
    return tuple(c / 255.0 for c in rgb)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fast pkl-based visualization for M2I/MIC-BEV results.')
    parser.add_argument('result_json', help='Path to results_nusc.json.')
    parser.add_argument('save_dir', help='Directory where images will be saved.')
    parser.add_argument('--config',
                        help='Model config. Visualization corruption follows its test_pipeline.')
    parser.add_argument('--ann-file',
                        default='/path/to/M2I/M2I_pkl/v2xset_infos_temporal_test.pkl')
    parser.add_argument('--data-root',
                        default='/path/to/M2I/M2I_split_dataset')
    parser.add_argument('--start', type=int, default=0,
                        help='Optional index of the first sample to render '
                        '(default: 0, i.e. start from the beginning).')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--conf-thresh', type=float, default=0.2)
    parser.add_argument('--bev-conf-thresh', type=float, default=0.4,
                        help='Confidence threshold for BEV map predicted boxes.')
    parser.add_argument('--single-image', action='store_true',
                        help='Save one 2x2 image per sample instead of pred/gt folders.')
    parser.add_argument('--show-map', action='store_true',
                        help='Save BEV map images with GT/pred boxes overlaid.')
    return parser.parse_args()


def get_cfg_value(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_random_mask_cfg(config_path):
    if not config_path:
        return None
    try:
        from mmcv import Config
    except ImportError as exc:
        raise ImportError(
            '--config requires mmcv so the visualizer can read test_pipeline.'
        ) from exc

    cfg = Config.fromfile(config_path)
    test_cfg = get_cfg_value(get_cfg_value(cfg, 'data', {}), 'test', {})
    pipeline = get_cfg_value(test_cfg, 'pipeline', [])
    for step in pipeline:
        if get_cfg_value(step, 'type') == 'RandomMaskMultiView':
            return step
    return None


def resolve_image_path(data_path, data_root):
    if os.path.isabs(data_path):
        return data_path
    return os.path.join(data_root, data_path)


def load_infos_by_token(ann_file):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    return {info['token']: info for info in infos}


def build_lidar2img(cam_info):
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

    lidar2cam_rt = np.eye(4, dtype=np.float32)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    lidar2cam = lidar2cam_rt.T

    intrinsic = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)
    viewpad = np.eye(4, dtype=np.float32)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    return viewpad @ lidar2cam


def build_gt_boxes(info):
    boxes = []
    gt_boxes = np.asarray(info.get('gt_boxes', []))
    gt_names = info.get('gt_names', ['gt'] * len(gt_boxes))
    valid_flag = info.get('valid_flag', np.ones(len(gt_boxes), dtype=bool))
    for box, name, valid in zip(gt_boxes, gt_names, valid_flag):
        if not valid:
            continue
        boxes.append(Boxnew(
            center=box[:3],
            size=box[3:6],
            orientation=Quaternion(axis=[0, 0, 1], radians=box[6]),
            name=str(name)))
    return [box_global_to_lidar(box, info) for box in boxes]


def box_global_to_lidar(box, info):
    box.translate(-np.asarray(info['ego2global_translation']))
    box.rotate(Quaternion(info['ego2global_rotation']).inverse)
    box.translate(-np.asarray(info['lidar2ego_translation']))
    box.rotate(Quaternion(info['lidar2ego_rotation']).inverse)
    return box


def build_pred_boxes(preds, conf_thresh, info):
    boxes = []
    for pred in preds:
        if pred.get('detection_score', 0.0) < conf_thresh:
            continue
        box = Boxnew(
            center=pred['translation'],
            size=pred['size'],
            orientation=Quaternion(pred['rotation']),
            score=pred.get('detection_score', np.nan),
            name=pred.get('detection_name', 'pred'))
        boxes.append(box_global_to_lidar(box, info))
    return boxes


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def clip_segment_rect(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang–Barsky: clip segment [(x0,y0),(x1,y1)] to axis-aligned rectangle."""
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-14:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            if t > u1:
                return None
            u0 = max(u0, t)
        else:
            if t < u0:
                return None
            u1 = min(u1, t)
    if u0 > u1:
        return None
    return (
        x0 + u0 * dx, y0 + u0 * dy,
        x0 + u1 * dx, y0 + u1 * dy,
    )


def _lock_image_axes(ax):
    """
    After ``imshow``, lock x/y limits to the image extent so line artists from
    near-camera projections (very large uv) do not trigger autoscale and shrink
    the photo or widen ``bbox_inches='tight'`` exports.
    """
    if not ax.images:
        return
    left, right, bottom, top = ax.images[0].get_extent()
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)


def _normalize_odd_kernel(kernel_size):
    kernel_size = int(kernel_size)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _sample_blur_from_cfg(rng, random_mask_cfg):
    blur_levels = get_cfg_value(random_mask_cfg, 'blur_levels')
    if blur_levels:
        level = rng.choice(list(blur_levels))
        if isinstance(level, str):
            sigma_text, kernel_text = level.split(':', 1)
            return float(sigma_text), _normalize_odd_kernel(kernel_text)
        return float(level[0]), _normalize_odd_kernel(level[1])

    blur_sigma = get_cfg_value(random_mask_cfg, 'blur_sigma', 10.0)
    if isinstance(blur_sigma, (tuple, list)):
        sigma = rng.uniform(float(blur_sigma[0]), float(blur_sigma[1]))
    else:
        sigma = float(blur_sigma)

    kernel_range = get_cfg_value(random_mask_cfg, 'blur_kernel_size_range')
    if kernel_range is not None:
        low = _normalize_odd_kernel(kernel_range[0])
        high = _normalize_odd_kernel(kernel_range[1])
        if low > high:
            low, high = high, low
        kernel_size = rng.randrange(low, high + 1, 2)
    else:
        kernel_size = _normalize_odd_kernel(
            get_cfg_value(random_mask_cfg, 'blur_kernel_size', 11)
        )
    return sigma, kernel_size


def deterministic_corruption(info, num_views, random_mask_cfg):
    if random_mask_cfg is None:
        return None
    prob = float(get_cfg_value(random_mask_cfg, 'mask_prob', 0.0))
    if num_views <= 1 or prob <= 0:
        return None
    token = info.get('token') or info.get('sample_idx') or info.get('scene_token')
    deterministic = bool(get_cfg_value(random_mask_cfg, 'deterministic', False))
    seed = int(hashlib.sha1(str(token).encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed) if deterministic else random
    if rng.random() >= prob:
        return None
    cfg_view_idx = get_cfg_value(random_mask_cfg, 'view_idx')
    view_idx = int(cfg_view_idx) if cfg_view_idx is not None else rng.choice(list(range(num_views)))
    if view_idx < 0 or view_idx >= num_views:
        return None
    mask_ratio = float(get_cfg_value(random_mask_cfg, 'mask_ratio', 0.5))
    blur_ratio = float(get_cfg_value(random_mask_cfg, 'blur_ratio', 0.5))
    total_ratio = mask_ratio + blur_ratio
    if total_ratio <= 0:
        return None
    alpha = float(get_cfg_value(random_mask_cfg, 'alpha', 1.0))
    mask_threshold = mask_ratio / total_ratio
    if rng.random() < mask_threshold:
        return dict(view_idx=view_idx, action='mask', alpha=float(alpha),
                    sigma=None, seed=seed)
    sigma, kernel_size = _sample_blur_from_cfg(rng, random_mask_cfg)
    return dict(view_idx=view_idx, action='blur', alpha=float(alpha),
                sigma=float(sigma), kernel_size=int(kernel_size), seed=seed)


def corruption_tag(corruption, view_idx):
    if corruption is None or view_idx != corruption['view_idx']:
        return 'clean'
    alpha = float(corruption.get('alpha', 1.0))
    if corruption['action'] == 'mask':
        return f"mask_alpha{alpha:.2f}"
    sigma = float(corruption.get('sigma') or 0.0)
    kernel_size = int(corruption.get('kernel_size') or 11)
    return f"blur_sigma{sigma:.2f}_kernel{kernel_size}_alpha{alpha:.2f}"


def apply_corruption(img, corruption, view_idx):
    if corruption is None or view_idx != corruption['view_idx']:
        return img
    arr = np.asarray(img).copy()
    alpha = float(corruption.get('alpha', 1.0))
    if corruption['action'] == 'mask':
        corrupted = np.zeros_like(arr)
    elif corruption['action'] == 'blur':
        sigma = float(corruption.get('sigma') or 10.0)
        k = _normalize_odd_kernel(corruption.get('kernel_size') or 11)
        corrupted = cv2.GaussianBlur(arr, (k, k), sigmaX=sigma, sigmaY=sigma)
    else:
        return img

    if alpha < 1.0:
        corrupted = (
            corrupted.astype(np.float32) * alpha
            + arr.astype(np.float32) * (1.0 - alpha)
        ).clip(0, 255).astype(np.uint8)
    return Image.fromarray(corrupted)


def resolve_map_path(info, data_root):
    map_path = info.get('map_path')
    if not map_path:
        return None
    if os.path.isabs(map_path):
        return map_path
    return os.path.join(data_root, map_path)

def render_bev_boxes(ax, boxes, color, label,
                     border_linewidth=0.5,
                     center_linewidth=2.4,
                     fill_alpha=0.3,
                     border_alpha=0.3,
                     show_heading=True,
                     color_by_class=False):
    first = True
    for box in boxes:
        box_color = mpl_color(detection_color_rgb(box.name)) if color_by_class else color
        corners = box.bottom_corners()[:2].T
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=box_color,
                linewidth=border_linewidth, alpha=border_alpha,
                label=label if first else None)
        ax.fill(corners[:, 0], corners[:, 1], color=box_color, alpha=fill_alpha)
        if show_heading:
            center = corners.mean(axis=0)
            front = (corners[0] + corners[1]) * 0.5
            ax.plot([center[0], front[0]], [center[1], front[1]],
                    color=box_color, linewidth=center_linewidth, alpha=1.0)
        first = False


def render_bev_map(info, gt_boxes, pred_boxes, save_dir, data_root):
    map_path = resolve_map_path(info, data_root)
    os.makedirs(save_dir, exist_ok=True)
    map_arr = None
    if map_path is not None and os.path.exists(map_path):
        map_arr = np.load(map_path)

    for name, boxes, color in (
        ('gt', gt_boxes, '#3399FF'),
        ('pred', pred_boxes, '#FF3333'),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=160)
        if map_arr is not None:
            labels = np.clip(map_arr.astype(np.int64), 0, len(MAP_PALETTE) - 1)
            rgb = MAP_PALETTE[labels]
            ax.imshow(rgb, origin='lower', extent=(-51.2, 51.2, -51.2, 51.2))
        else:
            ax.set_facecolor('#f5f5f5')
            ax.text(0.5, 0.98, 'No map_path in pkl',
                    transform=ax.transAxes, ha='center', va='top')
        render_bev_boxes(ax, boxes, color=color, label=name.upper())
        ax.scatter([0], [0], c='black', marker='x', s=40, label='Ego')
        ax.set_xlim(-51.2, 51.2)
        ax.set_ylim(-51.2, 51.2)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'bev_map_{name}.png'), dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=160)
    if map_arr is not None:
        labels = np.clip(map_arr.astype(np.int64), 0, len(MAP_PALETTE) - 1)
        rgb = MAP_PALETTE[labels]
        ax.imshow(rgb, origin='lower', extent=(-51.2, 51.2, -51.2, 51.2))
    else:
        ax.set_facecolor('#f5f5f5')
        ax.text(0.5, 0.98, 'No map_path in pkl',
                transform=ax.transAxes, ha='center', va='top')
    render_bev_boxes(ax, gt_boxes, color='#3399FF', label='GT',
                     fill_alpha=0.3)
    render_bev_boxes(ax, pred_boxes, color='#FF3333', label='PRED',
                     fill_alpha=0.3)
    ax.scatter([0], [0], c='black', marker='x', s=40, label='Ego')
    ax.set_xlim(-51.2, 51.2)
    ax.set_ylim(-51.2, 51.2)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'bev_map_overlay.png'), dpi=160)
    plt.close(fig)


def project_box_corners(box, lidar2img):
    corners = box.corners()
    corners = np.vstack([corners, np.ones((1, corners.shape[1]))])
    points = lidar2img @ corners
    depth = points[2]
    xy = np.zeros((2, corners.shape[1]), dtype=np.float32)
    valid_depth = depth > 1e-5
    xy[:, valid_depth] = points[:2, valid_depth] / depth[valid_depth]
    return xy, valid_depth


def box_visible_in_image(box, lidar2img, width, height):
    """
    Keep boxes that NuScenes would consider plausibly on-image: at least one
    corner projects in front, and the 2D AABB of valid projections intersects
    the image rectangle (filters boxes fully outside the FoV).
    """
    xy, valid_depth = project_box_corners(box, lidar2img)
    if not np.any(valid_depth):
        return False

    x = xy[0, valid_depth]
    y = xy[1, valid_depth]
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    ix0 = max(0.0, min(float(width), xmin))
    ix1 = max(0.0, min(float(width), xmax))
    iy0 = max(0.0, min(float(height), ymin))
    iy1 = max(0.0, min(float(height), ymax))
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    return True


def draw_projected_box(ax, box, lidar2img, width, height, color, linewidth=1.2):
    """Draw box edges clipped to image bounds (near-camera boxes stay on-canvas)."""
    xy, valid_depth = project_box_corners(box, lidar2img)
    w, h = float(width), float(height)
    for start, end in BOX_EDGES:
        if not (valid_depth[start] and valid_depth[end]):
            continue
        x0, y0 = float(xy[0, start]), float(xy[1, start])
        x1, y1 = float(xy[0, end]), float(xy[1, end])
        clip = clip_segment_rect(x0, y0, x1, y1, 0.0, w, 0.0, h)
        if clip is None:
            continue
        xa, ya, xb, yb = clip
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=linewidth, clip_on=True)


def draw_boxes_per_class(ax, img, lidar2img, boxes):
    width, height = img.size
    ax.imshow(img)
    for box in boxes:
        if not box_visible_in_image(box, lidar2img, width, height):
            continue
        rgb = detection_color_rgb(box.name)
        draw_projected_box(ax, box, lidar2img, width, height, mpl_color(rgb))
    _lock_image_axes(ax)
    ax.axis('off')


def render_grid(info, pred_boxes, gt_boxes, save_path, data_root, args, random_mask_cfg):
    cams = list(info['cams'].keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
    axes = axes.ravel()
    corruption = deterministic_corruption(info, len(cams), random_mask_cfg)

    for view_idx, (ax, cam_name) in enumerate(zip(axes, cams)):
        cam_info = info['cams'][cam_name]
        img_path = resolve_image_path(cam_info['data_path'], data_root)
        lidar2img = build_lidar2img(cam_info)
        img = Image.open(img_path)
        tag = corruption_tag(corruption, view_idx)
        img = apply_corruption(img, corruption, view_idx)
        width, height = img.size
        ax.imshow(img)
        for box in gt_boxes:
            if not box_visible_in_image(box, lidar2img, width, height):
                continue
            rgb = detection_color_rgb(box.name)
            draw_projected_box(ax, box, lidar2img, width, height, mpl_color(rgb),
                               linewidth=0.5)
        for box in pred_boxes:
            if not box_visible_in_image(box, lidar2img, width, height):
                continue
            rgb = detection_color_rgb(box.name)
            draw_projected_box(ax, box, lidar2img, width, height, mpl_color(rgb),
                               linewidth=0.5)
        _lock_image_axes(ax)
        ax.set_title(f'{cam_name}: {tag}', fontsize=7, pad=2)
        ax.axis('off')

    for ax in axes[len(cams):]:
        ax.axis('off')

    fig.suptitle('Colors match nuScenes colormap per class', fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01,
                        wspace=0.01, hspace=0.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def render_split(info, pred_boxes, gt_boxes, save_dir, data_root, args, random_mask_cfg):
    cams = list(info['cams'].items())
    corruption = deterministic_corruption(info, len(cams), random_mask_cfg)

    for subdir, boxes in (
        ('gt', gt_boxes),
        ('pred', pred_boxes),
    ):
        os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
        for view_idx, (cam_name, cam_info) in enumerate(cams):
            img_path = resolve_image_path(cam_info['data_path'], data_root)
            lidar2img = build_lidar2img(cam_info)
            img = Image.open(img_path)
            tag = corruption_tag(corruption, view_idx)
            img = apply_corruption(img, corruption, view_idx)
            width, height = img.size
            dpi = 100
            fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
            draw_boxes_per_class(ax, img, lidar2img, boxes)
            ax.set_title(f'{cam_name}: {tag}', fontsize=12, pad=4)
            fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
            fig.savefig(os.path.join(save_dir, subdir, f'{cam_name}_{tag}.png'),
                        dpi=dpi)
            plt.close(fig)


def main():
    args = parse_args()
    with open(args.result_json) as f:
        result_data = json.load(f)

    random_mask_cfg = load_random_mask_cfg(args.config)
    infos_by_token = load_infos_by_token(args.ann_file)
    tokens = list(result_data['results'].keys())
    tokens = tokens[args.start:args.start + args.num_samples]

    index_width = max(2, len(str(args.start + len(tokens))))
    rendered = 0
    for token_offset, token in enumerate(tokens):
        info = infos_by_token.get(token)
        if info is None:
            print(f'[WARN] Token not found in pkl, skipping: {token}')
            continue

        gt_boxes = build_gt_boxes(info)
        pred_boxes = build_pred_boxes(
            result_data['results'].get(token, []), args.conf_thresh, info)
        bev_pred_boxes = build_pred_boxes(
            result_data['results'].get(token, []), args.bev_conf_thresh, info)
        absolute_index = args.start + token_offset + 1
        sample_dir = os.path.join(
            args.save_dir, f'{absolute_index:0{index_width}d}_{token}')
        if args.single_image:
            render_grid(info, pred_boxes, gt_boxes,
                        os.path.join(sample_dir, 'overlay.png'), args.data_root,
                        args, random_mask_cfg)
        else:
            render_split(info, pred_boxes, gt_boxes, sample_dir, args.data_root,
                         args, random_mask_cfg)
        if args.show_map:
            render_bev_map(info, gt_boxes, bev_pred_boxes, sample_dir, args.data_root)
        rendered += 1
        print(f'[{rendered}/{len(tokens)}] saved {sample_dir}')

    print(f'Done. Rendered {rendered} samples to {args.save_dir}')


if __name__ == '__main__':
    main()
