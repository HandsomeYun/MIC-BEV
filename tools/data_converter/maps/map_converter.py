import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# Class mapping
CLASS_ID = {
    "background": 0,
    "driving":    1,
    "sidewalk":   2,
    "crosswalk":  3,
    "shoulder":   4,
    "border":     5,
    "parking":    6,
}

ROAD_TYPES = {
    "driving", "entry", "exit", "onramp", "offRamp",
    "connectingRamp", "onRamp", "bidirectional"
}

def generate_bev_map(
    total_areas, crosswalks,
    center_xy,
    town_name,
    save_dir = None,
    world_range=51.2,
    base_size=512,
    downsample_size=200,
):
    cx, cy = center_xy
    x_min, x_max = cx - world_range, cx + world_range
    y_min, y_max = -cy - world_range, -cy + world_range
    res = (x_max - x_min) / base_size

    def world_to_pixel(x, y):
        px = int((x - x_min) / res)
        py = int((y_max - y) / res)
        return px, py

    # Initialize blank BEV map
    bev = np.zeros((base_size, base_size), dtype=np.uint8)

    # Draw roads
    for area in total_areas.values():
        for lane_dict in (area["left_lanes_area"], area["right_lanes_area"]):
            for lid, la in lane_dict.items():
                lt = area["types"][lid]
                if lt in ROAD_TYPES:
                    cid = CLASS_ID["driving"]
                elif lt in CLASS_ID:
                    cid = CLASS_ID[lt]
                else:
                    continue
                poly = la["inner"] + la["outer"][::-1]
                pts = np.array([world_to_pixel(x, y) for x, y in poly], np.int32)[None]
                if pts.shape[1] >= 3:
                    cv2.fillPoly(bev, pts, cid)

    # Draw crosswalks
    for cw in crosswalks:
        pts = np.array([world_to_pixel(pt["X"], pt["Y"]) for pt in cw["outline"]], np.int32)[None]
        if pts.shape[1] >= 3:
            cv2.fillPoly(bev, pts, CLASS_ID["crosswalk"])

    # Downsample
    bev_small = cv2.resize(bev, (downsample_size, downsample_size), interpolation=cv2.INTER_NEAREST)

    # Save
    if not save_dir:
        save_dir = os.path.join("/home/handsomeyun/BEVFormer/data/map/bev_semantic_output", town_name, str(int(cx)) + '_' + str(int(cy)))
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "bev_label_map_200.npy"), bev_small)

    return bev_small


def show_bev(bev, title="BEV", class_id=CLASS_ID):
    inv = {v: k for k, v in class_id.items()}
    colors = [
        "#e0e0e0", "#1f77b4", "#ff7f0e",
        "#2ca02c", "#d62728", "#9467bd", "#8c564b"
    ]
    cmap = ListedColormap(colors[:len(class_id)])
    plt.figure(figsize=(6, 6))
    plt.imshow(bev, cmap=cmap, vmin=0, vmax=len(class_id)-1)
    plt.title(title)
    patches = [Patch(color=colors[i], label=inv[i]) for i in range(len(class_id))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# ───────────── Example Usage ─────────────
if __name__ == "__main__":
    total_areas = pickle.load(open(
        "/home/handsomeyun/BEVFormer/data/map/Town10HD_Opt/-62_20/Town10HD_Opt_filtered_total_areas.pkl", "rb"))
    crosswalks = pickle.load(open(
        "/home/handsomeyun/BEVFormer/data/map/Town10HD_Opt/-62_20/Town10HD_Opt_filtered_crosswalks.pkl", "rb"))

    center = (-62.34999847, 20.18999863)

    bev200 = generate_bev_map(
        total_areas, crosswalks,
        center_xy=center,
        downsample_size=200,
        town_name = "Town10HD")

    show_bev(bev200, title="BEV Semantic Map 200×200")
