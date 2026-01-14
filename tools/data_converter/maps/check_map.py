#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  Quick BEV sanity-check for the generated semantic map & GT boxes
#  Author: open-mmlab / BEVFormer community – adapted by ChatGPT
# ─────────────────────────────────────────────────────────────────────────────
"""
USAGE
-----
python tools/data_converter/maps/check_map.py \
       --pkl ./data/v2x_all_4cam_map/v2xset_infos_temporal_train.pkl \
       --index 200 \
       --out   ./vis/

"""

import argparse, mmcv, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from math import cos, sin
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox   # ← concrete!

# ───────────── helper ─────────────────────────────────────────────────────────
def box_corners_xy(cx, cy, w, l, yaw):
    """Return the 4 × 2 XY corners of a box (ego-frame)."""
    c, s = cos(yaw), sin(yaw)
    R = np.array([[c, -s],
                  [s,  c]])
    dx = np.array([[ l/2,  w/2],
                   [ l/2, -w/2],
                   [-l/2, -w/2],
                   [-l/2,  w/2]]).T                # (2,4)
    return (R @ dx + np.array([[cx], [cy]])).T      # (4,2)

# ───────────── main ───────────────────────────────────────────────────────────
def main(args):
    # ── load PKL & pick sample ------------------------------------------------
    pkl  = mmcv.load(args.pkl)
    info = pkl['infos'][args.index]
    print(f'visualising sample #{args.index}   token = {info["token"]}')

    # ── load the semantic raster ---------------------------------------------
    bev_map = np.load(info['map_path'])           # 200 × 200, uint8
    assert bev_map.ndim == 2, bev_map.shape

    # ── plot ------------------------------------------------------------------
    PC_RANGE = [-51.2, -51.2, -7, 51.2, 51.2, 1]          # adapt if needed
    h, w = bev_map.shape
    extent = [PC_RANGE[1], PC_RANGE[4], PC_RANGE[0], PC_RANGE[3]]  # left→right, bottom→top

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(bev_map,
                   origin='lower',
                   cmap='tab20',
                   extent=extent,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='semantic id')

    # draw ego
    ax.scatter([0], [0], c='white', s=40, zorder=5)
    ax.arrow(0, 0, 10, 0, width=0.3, color='white', zorder=5)
    ax.text(12, 1, '+X forward', color='white')

    # ── convert GT boxes to DetectionBox for completeness ---------------------
    boxes   = []
    for bb, cls in zip(info['gt_boxes'], info['gt_names']):
        cx, cy, cz, w, l, h, yaw = bb
        boxes.append(DetectionBox(
            sample_token   = info['token'],
            translation    = (cx, cy, cz),
            size           = (w, l, h),
            rotation       = (0, 0, np.sin(yaw / 2), np.cos(yaw / 2)),  # (x,y,z,w) quaternion
            velocity       = (0.0, 0.0),  # unknown
            detection_name = cls,
            detection_score= 1.0,
            attribute_name = ''
        ))

        # 2-D footprint
        pts = box_corners_xy(cx, cy, w, l, yaw)
        ax.add_patch(plt.Polygon(pts, fill=False, ec='red', lw=1.0))

    ax.set_xlabel('Y  (left  ←   → right)  [m]')
    ax.set_ylabel('X  (rear  ↓   ↑  front) [m]')
    ax.set_title('Semantic raster + GT boxes (ego frame)')
    ax.set_aspect('equal')
    plt.tight_layout()

    # ── save ------------------------------------------------------------------
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'{info["token"]}_bev.png'
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        print('✅ saved:', out_file)
    else:
        plt.show()


# ───────────── CLI ───────────────────────────────────────────────────────────
# ─── at the very end of check_map.py ─────────────────────────────────────────
if __name__ == '__main__':
    # instead of parsing argv, just bake in your values:
    import argparse
    args = argparse.Namespace(
        pkl   = "/home/handsomeyun/BEVFormer/data/v2x_all_4cam_map/v2xset_infos_temporal_train.pkl",
        index = 0,
        out   = "/home/handsomeyun/BEVFormer/data_dumping"
    )
    main(args)

