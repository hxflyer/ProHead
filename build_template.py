import argparse
import os

import cv2
import numpy as np
import torch

from train_visualize_helper import load_topology, create_combined_overlay


def load_landmarks_txt(path: str) -> np.ndarray | None:
    lines = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()

                if len(parts) >= 5:
                    vals = [float(x) for x in parts[:5]]
                    lines.append(vals)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    if not lines:
        return None

    lm = np.asarray(lines, dtype=np.float32)
    # Normalize 2D (last 2 cols) same as MetahumanDataset2 (assumes 1024x1024 source)
    lm[:, 3:5] /= 1024.0
    return lm


def load_mesh_txt(path: str) -> np.ndarray | None:
    """Load mesh vertices from txt file. Format: x y z [u v]"""
    lines = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()

                # Mesh might have 3 (xyz) or 5 (xyz + uv) values
                if len(parts) >= 3:
                    vals = [float(x) for x in parts[:min(5, len(parts))]]
                    # Pad to 5 if needed
                    while len(vals) < 5:
                        vals.append(0.0)
                    lines.append(vals)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

    if not lines:
        return None

    mesh = np.asarray(lines, dtype=np.float32)
    # Normalize 2D (last 2 cols) if present
    if mesh.shape[1] >= 5:
        mesh[:, 3:5] /= 1024.0
    return mesh


def iter_landmark_files(roots):
    for root in roots:
        if not os.path.exists(root):
            print(f"Root not found: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("landmark") and name.lower().endswith(".txt"):
                    yield os.path.join(dirpath, name)


def iter_mesh_files(roots):
    for root in roots:
        if not os.path.exists(root):
            print(f"Root not found: {root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("mesh") and name.lower().endswith(".txt"):
                    yield os.path.join(dirpath, name)


def find_mesh_landmark_pairs(landmark_roots, mesh_roots):
    """Find matching mesh and landmark file pairs based on filename patterns."""
    # Collect all landmark files
    landmark_files = {}
    for path in iter_landmark_files(landmark_roots):
        filename = os.path.basename(path)
        # Extract pattern: landmark_X_Y.txt -> X_Y
        if filename.startswith("landmark_") and filename.endswith(".txt"):
            key = filename[len("landmark_"):-4]  # Remove prefix and .txt
            landmark_files[key] = path
    
    # Collect all mesh files
    mesh_files = {}
    for path in iter_mesh_files(mesh_roots):
        filename = os.path.basename(path)
        # Extract pattern: mesh_X_Y.txt -> X_Y
        if filename.startswith("mesh_") and filename.endswith(".txt"):
            key = filename[len("mesh_"):-4]  # Remove prefix and .txt
            mesh_files[key] = path
    
    # Find pairs
    pairs = []
    for key in landmark_files:
        if key in mesh_files:
            pairs.append((landmark_files[key], mesh_files[key]))
        else:
            print(f"Warning: No matching mesh file for landmark key: {key}")
    
    print(f"Found {len(pairs)} matching mesh-landmark pairs")
    return pairs


def compute_templates_paired(
    landmark_roots,
    mesh_roots,
    max_samples: int | None = None,
):
    """
    Compute both mesh and landmark templates using paired files.
    Normalization parameters are derived from landmark and applied to both.
    """
    # Find paired files
    pairs = find_mesh_landmark_pairs(landmark_roots, mesh_roots)
    
    if not pairs:
        print("No paired files found!")
        return
    
    landmark_acc = None
    mesh_acc = None
    count = 0
    expected_landmark_n = None
    expected_mesh_n = None

    for landmark_path, mesh_path in pairs:
        # Load both files
        lm = load_landmarks_txt(landmark_path)
        mesh = load_mesh_txt(mesh_path)
        
        if lm is None or mesh is None:
            continue
        
        # Initialize accumulators on first valid pair
        if expected_landmark_n is None:
            expected_landmark_n = lm.shape[0]
            expected_mesh_n = mesh.shape[0]
            landmark_acc = np.zeros_like(lm, dtype=np.float64)
            mesh_acc = np.zeros_like(mesh, dtype=np.float64)
            print(f"Template sizes: landmark={expected_landmark_n}, mesh={expected_mesh_n}")
            print(f"First pair: {os.path.basename(landmark_path)} + {os.path.basename(mesh_path)}")
        else:
            # Check consistency
            if lm.shape[0] != expected_landmark_n:
                print(f"Skip {landmark_path}: landmark count {lm.shape[0]} != {expected_landmark_n}")
                continue
            if mesh.shape[0] != expected_mesh_n:
                print(f"Skip {mesh_path}: mesh count {mesh.shape[0]} != {expected_mesh_n}")
                continue
        
        # Compute normalization parameters FROM LANDMARK ONLY
        lm_xyz = lm[:, 0:3]
        min_xyz = np.min(lm_xyz, axis=0)
        max_xyz = np.max(lm_xyz, axis=0)
        center = (min_xyz + max_xyz) / 2
        scale = np.max(max_xyz - min_xyz) / 2  # Uniform scale
        
        if scale <= 0:
            print(f"Skip pair: invalid scale {scale}")
            continue
        
        # Apply landmark normalization to landmark
        lm[:, 0:3] = (lm_xyz - center) / scale
        
        # Apply SAME normalization to mesh
        mesh_xyz = mesh[:, 0:3]
        mesh[:, 0:3] = (mesh_xyz - center) / scale
        
        # Accumulate
        landmark_acc += lm
        mesh_acc += mesh
        count += 1
        
        if max_samples is not None and count >= max_samples:
            break
        
        if count % 500 == 0:
            print(f"Processed {count} pairs...")
    
    if landmark_acc is None or count == 0:
        print("No valid pairs processed.")
        return
    
    # Compute templates
    template_landmark = (landmark_acc / count).astype(np.float32)
    template_mesh = (mesh_acc / count).astype(np.float32)
    
    # Final refinement: ensure landmark template is perfectly centered/scaled
    lm_xyz = template_landmark[:, 0:3]
    min_xyz = np.min(lm_xyz, axis=0)
    max_xyz = np.max(lm_xyz, axis=0)
    center = (min_xyz + max_xyz) / 2
    scale = np.max(max_xyz - min_xyz) / 2
    
    if scale > 0:
        # Apply final refinement to both
        template_landmark[:, 0:3] = (lm_xyz - center) / scale
        mesh_xyz = template_mesh[:, 0:3]
        template_mesh[:, 0:3] = (mesh_xyz - center) / scale
    
    # Save templates
    np.save("model/landmark_template.npy", template_landmark)
    np.save("model/mesh_template.npy", template_mesh)
    
    print(f"\nSaved templates using {count} pairs (Aligned Normalization):")
    print(f"  - model/landmark_template.npy")
    print(f"  - model/mesh_template.npy")
    
    # Visualization for landmark
    try:
        topology = load_topology()
        h = w = 1024
        black = np.zeros((h, w, 3), dtype=np.uint8)
        
        lm_px = template_landmark[:, 3:5].copy()
        lm_px[:, 0] *= w
        lm_px[:, 1] *= h
        
        overlay = create_combined_overlay(black, lm_px, topology)
        out_path = "template_landmark_overlay.png"
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  - {out_path}")
    except Exception as e:
        print(f"Could not create visualization: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute mesh and landmark templates with aligned normalization"
    )
    parser.add_argument(
        "--landmark-roots",
        nargs="+",
        default=[
            "G:/CapturedFrames_final1_processed/landmark",
        ],
        help="Landmark data root directories",
    )
    parser.add_argument(
        "--mesh-roots",
        nargs="+",
        default=[
            "G:/CapturedFrames_final1_processed/mesh",
        ],
        help="Mesh data root directories",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max number of samples to process")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Building Mesh & Landmark Templates (Aligned)")
    print("=" * 60)
    print("Using landmark-derived normalization for both mesh and landmark")
    print("This ensures they remain in the same normalized space\n")
    
    compute_templates_paired(
        landmark_roots=args.landmark_roots,
        mesh_roots=args.mesh_roots,
        max_samples=args.max_samples,
    )
    
    print("\n" + "=" * 60)
    print("Template Building Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
