#!/usr/bin/env python3

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Class definitions
CLASS_ID = {
    "background": 0,
    "driving":    1,
    "sidewalk":   2,
    "crosswalk":  3,
    "shoulder":   4,
    "border":     5,
    "parking":    6,
}

# Prettier palette indexed by class ID
palette = np.array([
    [255, 255, 255],   # 0 background – white
    [ 99, 165, 112],   # 1 driving   – green
    [193, 182, 255],   # 2 sidewalk  – dark pink
    [212, 154, 158],   # 3 crosswalk – pink
    [116,  60,  56],   # 4 shoulder  – coral red
    [ 33,  64,  43],   # 5 border    – orchid purple
    [  0, 188, 212],   # 6 parking   – teal
], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(
        description="Load, display, and save a semantic BEV map stored as a NumPy .npy file."
    )
    parser.add_argument(
        "path",
        help="Path to the .npy BEV map file",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/yun/MIC-BEV_Official/data_dumping/map",
        help="Directory to save the PNG (default: %(default)s)"
    )
    args = parser.parse_args()

    # Load the BEV map
    bev_map = np.load(args.path)

    # Print some info
    print(f"Loaded BEV map from: {args.path}")
    print("Shape:", bev_map.shape)
    print("Unique values (class IDs):", np.unique(bev_map))

    # Make sure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Map class IDs to RGB image for display
    # Normalize palette to [0,1] for matplotlib
    norm_palette = palette / 255.0
    rgb_map = norm_palette[bev_map]

    # Prepare figure
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(rgb_map)
    ax.set_title('Semantic BEV Map')
    ax.axis('off')

    # Create legend handles
    handles = [
        Patch(facecolor=norm_palette[class_id],
              edgecolor='k',
              label=class_name)
        for class_name, class_id in CLASS_ID.items()
    ]
    # Place legend outside to the right
    ax.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        title="Classes",
        frameon=True
    )
    plt.tight_layout()

    # Show on screen
    plt.show()

    # Save to PNG
    base = os.path.splitext(os.path.basename(args.path))[0]
    out_path = os.path.join(args.out_dir, base + ".png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    main()
