import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------------------------------------------------------------
# 1) IMPORT NuScenes SDK AND SET UP YOUR DATAROOT + SAMPLE TOKEN
# ------------------------------------------------------------
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# ←── EDIT THIS TO YOUR LOCAL NUSCENES “dataroot” ─────────────
DATAROOT = "/data2/mcbev-testdata"  
VERSION  = "v1.0-trainval"  # or “v1.0-test” if you’re using test split
# ←────────────────────────────────────────────────────────────

# The exact sample token you posted:
SAMPLE_TOKEN = "fourway_town10_v2xset_inf_test_c_c_day_s11_-1_000031"

# Load NuScenes
nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

# ------------------------------------------------------------
# 2) PULL ALL GROUND-TRUTH BOXES (with correct size/orientation)
# ------------------------------------------------------------
def load_gt_boxes_for_sample(sample_token):
    """
    Returns a list of (Box, category_name) for every annotation in the sample.
    The Box is in WORLD coordinates by default.
    """
    sample = nusc.get("sample", sample_token)
    gt_list = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        # ann['category_name'] is e.g. "vehicle.car", "vehicle.bicycle", etc.
        cat = ann["category_name"].split('.')[-1]  # just "car", "truck", "bicycle", …
        # Build a Box in WORLD frame:
        b = Box( 
            center          = np.array(ann["translation"]),       # (x,y,z)
            size            = np.array(ann["size"]),              # (w, l, h)
            orientation     = Quaternion(ann["rotation"]),         # quaternion (w, x, y, z)
            name            = ann["category_name"],                # full category
            token           = ann["token"]
        )
        gt_list.append((b, cat))
    return gt_list

gt_boxes = load_gt_boxes_for_sample(SAMPLE_TOKEN)

# ------------------------------------------------------------
# 3) YOUR PREDICTION CENTERS (as given)
# ------------------------------------------------------------
# centers + class
predicted_data = [
    ([-75.11174774, -12.971241  ,  0.77426338], "car"),
    ([-69.10861206, -13.03873444,  0.76847339], "truck"),
    ([-32.73389053, -12.75227356,  0.699651  ], "car"),
    ([-52.52011108, -20.66088486,  0.61195683], "bicycle"),
    ([-42.05620956,   8.59814453,  0.87082553], "car"),
    ([-48.92546463,  -0.88460541,  0.65660095], "car"),
    ([-56.09838104, -24.52594757,  0.90631318], "truck"),
    ([-38.25670242, -22.87346268,  0.70912814], "car"),
    ([-23.34896469, -28.29297638,  0.81826973], "truck"),
    ([-20.12604141, -13.20388412,  0.79976225], "car"),
    ([-20.12604141, -13.20388412,  0.79976225], "truck"),  # same center, two labels
]

pred_coords = np.array([x for x, cls in predicted_data])
pred_labels = [cls for x, cls in predicted_data]

# ------------------------------------------------------------
# 4) SET UP COLORS FOR EACH CLASS
# ------------------------------------------------------------
color_map = {
    "car":        "tab:blue",
    "truck":      "tab:purple",
    "pedestrian": "tab:green",
    "bicycle":    "tab:orange"
}

# Utility to draw a Box wireframe in 3D
def draw_wireframe_box(ax: Axes3D, box: Box, color: str, linewidth: float = 1.5):
    """
    Given a NuScenes Box (3×8 corners), plot its edges in 3D.
    """
    corners = box.corners()  # shape (3, 8)
    # define edges by corner indices:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # vertical edges
    ]
    for (i, j) in edges:
        xs = [corners[0, i], corners[0, j]]
        ys = [corners[1, i], corners[1, j]]
        zs = [corners[2, i], corners[2, j]]
        ax.plot(xs, ys, zs, color=color, linewidth=linewidth)

# ------------------------------------------------------------
# 5) PLOT EVERYTHING
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(f"Sample {SAMPLE_TOKEN}  ▶  GT Cuboids vs. Predicted Centers")

# 5a) Plot each GT box as a wireframe
for box, cls in gt_boxes:
    c = color_map.get(cls, "black")
    draw_wireframe_box(ax, box, color=c, linewidth=1.2)

# 5b) Scatter the predicted centers as ▲
for cls in np.unique(pred_labels):
    idxs = [i for i, lab in enumerate(pred_labels) if lab == cls]
    pts = pred_coords[idxs]
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        color=color_map.get(cls, "black"),
        marker="^", s=80, edgecolor="k", linewidth=0.8,
        label=f"PRED {cls}"
    )

# 5c) Optionally, mark the LiDAR ego at (0,0,0) in red
ax.scatter([0], [0], [0], color="red", s=40, label="LiDAR Ego (0,0,0)")

# 5d) Label axes
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")

# 5e) Legend (move outside so boxes don’t overlap)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

# 5f) Equal aspect ratio: ensure X, Y, Z scale look correct
#     We keep Z starting at 0 (ground plane).
all_pts = np.vstack([b.center for b, _ in gt_boxes] + [pred_coords])
max_range = np.ptp(all_pts, axis=0).max() / 2.0
mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2
mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2
mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(0,       mid_z + max_range)  # floor Z at 0

plt.tight_layout()
plt.show()
