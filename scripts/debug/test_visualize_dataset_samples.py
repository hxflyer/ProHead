"""
Visualize landmark and mesh points from GeometryDataset on RGB images.
Usage:
    python scripts/debug/test_visualize_dataset_samples.py --data_root G:/CapturedFrames_final_processed --num_samples 4
"""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _candidate in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if (_candidate / "data_utils").exists():
        _PROJECT_ROOT = _candidate
        break
else:
    _PROJECT_ROOT = _THIS_FILE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import sys
import argparse
import random
import numpy as np
import cv2


from geometry_dataset import GeometryDataset
from train_visualize_helper import load_landmark_topology, load_mesh_topology, create_combined_overlay


def draw_points_on_image(image, points_norm, image_size, color=(0, 255, 0), radius=2):
    """Draw normalized [0,1] 2D points on image."""
    h, w = image.shape[:2]
    out = image.copy()
    for i in range(len(points_norm)):
        x = int(points_norm[i, 0] * w)
        y = int(points_norm[i, 1] * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset samples with landmark/mesh points")
    parser.add_argument("--data_root", type=str, default="G:/CapturedFrames_final_processed",
                        help="Path to dataset directory")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="artifacts/debug/test_previews_dataset", help="Output directory")
    parser.add_argument("--image_size", type=int, default=512, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="both", choices=["landmark", "mesh", "both", "wireframe"],
                        help="What to draw: landmark points, mesh points, both, or wireframe overlay")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"Loading dataset from: {args.data_root}")
    dataset = GeometryDataset(
        data_roots=args.data_root,
        split='val',
        image_size=args.image_size,
        train_ratio=0.95,
        augment=False,
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No samples found!")
        return

    num = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num)
    print(f"Visualizing indices: {indices}")

    # Load topologies for wireframe mode
    lm_topo = None
    mesh_topo = None
    if args.mode == "wireframe":
        try:
            lm_topo = load_landmark_topology()
            mesh_topo = load_mesh_topology()
        except Exception as e:
            print(f"Warning: could not load topology: {e}")

    for idx in indices:
        sample = dataset[idx]
        rgb = sample['rgb'].permute(1, 2, 0).numpy()  # [H, W, 3] float [0,1]
        img_u8 = (rgb * 255).astype(np.uint8)

        landmarks = sample['landmarks'].numpy()  # [N, 6] (x,y,z,u,v,depth)
        mesh = sample['mesh'].numpy()            # [M, 6]
        lm_weights = sample['landmark_weights'].numpy()
        mesh_weights = sample['mesh_weights'].numpy()

        # UV coords are columns 3,4 (normalized to image space)
        lm_uv = landmarks[:, 3:5]
        mesh_uv = mesh[:, 3:5]

        h, w = img_u8.shape[:2]
        vis = img_u8.copy()

        if args.mode == "wireframe":
            # Wireframe overlay using topology
            if lm_topo is not None:
                lm_px = lm_uv.copy()
                lm_px[:, 0] *= w
                lm_px[:, 1] *= h
                vis = create_combined_overlay(vis, lm_px, lm_topo)

            if mesh_topo is not None:
                mesh_px = mesh_uv.copy()
                mesh_px[:, 0] *= w
                mesh_px[:, 1] *= h
                vis = create_combined_overlay(vis, mesh_px, mesh_topo)
        else:
            if args.mode in ("mesh", "both"):
                # Mesh points in blue (draw first so landmarks are on top)
                for i in range(len(mesh_uv)):
                    if mesh_weights[i] < 0.5:
                        continue
                    x = int(mesh_uv[i, 0] * w)
                    y = int(mesh_uv[i, 1] * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(vis, (x, y), 1, (100, 100, 255), -1, cv2.LINE_AA)

            if args.mode in ("landmark", "both"):
                # Landmark points in green
                for i in range(len(lm_uv)):
                    if lm_weights[i] < 0.5:
                        continue
                    x = int(lm_uv[i, 0] * w)
                    y = int(lm_uv[i, 1] * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)

        # Save
        path = sample.get('image_path', f'sample_{idx}')
        basename = os.path.splitext(os.path.basename(path))[0] if isinstance(path, str) else f"sample_{idx}"
        out_path = os.path.join(args.output_dir, f"{basename}_{args.mode}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}  (lm={len(lm_uv)}, mesh={len(mesh_uv)}, "
              f"lm_valid={int(lm_weights.sum())}, mesh_valid={int(mesh_weights.sum())})")


if __name__ == "__main__":
    main()
