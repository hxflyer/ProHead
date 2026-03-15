"""
Precompute average normalized depth for mesh and landmark templates.

Iterates all samples in the specified data folders, computes per-sample
normalized depth via compute_vertex_depth, and averages across all valid
samples. Writes the result into column 5 of the template .npy files.

Usage:
    python precompute_template_depth.py --data_roots G:/CapturedFrames_final8_processed
"""

import argparse
import glob
import os

import numpy as np

from mat_load_helper import load_matrix_data, compute_vertex_depth


def load_geometry(filepath: str) -> np.ndarray | None:
    """Load geometry from txt file (5 cols: x3d y3d z3d x2d y2d)."""
    try:
        geom = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 5:
                    vals = [float(x) for x in parts[:5]]
                    geom.append(vals)
                elif len(parts) >= 3:
                    geom.append([float(parts[0]), float(parts[1]), float(parts[2]), 0.0, 0.0])
        geom = np.array(geom, dtype=np.float32)
        return geom if geom.shape[0] > 0 else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Precompute average depth for templates")
    parser.add_argument("--data_roots", nargs="+", required=True,
                        help="Data folders with mesh/landmark/mat subfolders")
    parser.add_argument("--model_dir", type=str, default="model",
                        help="Directory containing template .npy files")
    args = parser.parse_args()

    model_dir = args.model_dir
    mesh_template_path = os.path.join(model_dir, "mesh_template.npy")
    landmark_template_path = os.path.join(model_dir, "landmark_template.npy")

    mesh_template = np.load(mesh_template_path).astype(np.float32)
    landmark_template = np.load(landmark_template_path).astype(np.float32)

    # Load indices if they exist
    mesh_indices = None
    landmark_indices = None
    if os.path.exists(os.path.join(model_dir, "mesh_indices.npy")):
        mesh_indices = np.load(os.path.join(model_dir, "mesh_indices.npy"))
    if os.path.exists(os.path.join(model_dir, "landmark_indices.npy")):
        landmark_indices = np.load(os.path.join(model_dir, "landmark_indices.npy"))

    # Determine expected vertex counts (after index filtering)
    if mesh_indices is not None and mesh_indices.max() < mesh_template.shape[0]:
        n_mesh = len(mesh_indices)
    else:
        n_mesh = mesh_template.shape[0]
        mesh_indices = None

    if landmark_indices is not None and landmark_indices.max() < landmark_template.shape[0]:
        n_landmark = len(landmark_indices)
    else:
        n_landmark = landmark_template.shape[0]
        landmark_indices = None

    print(f"Mesh template: {mesh_template.shape}, expected vertices after indexing: {n_mesh}")
    print(f"Landmark template: {landmark_template.shape}, expected vertices after indexing: {n_landmark}")

    # Collect all samples
    mesh_depth_accum = np.zeros(n_mesh, dtype=np.float64)
    mesh_depth_count = 0
    landmark_depth_accum = np.zeros(n_landmark, dtype=np.float64)
    landmark_depth_count = 0

    for data_root in args.data_roots:
        if not os.path.exists(data_root):
            print(f"Skipping missing folder: {data_root}")
            continue

        color_files = glob.glob(os.path.join(data_root, "Color_*"))
        color_files = [f for f in color_files
                       if os.path.basename(f).startswith("Color_") and not f.endswith("_mask.png")]

        sample_ids = set()
        for cf in color_files:
            fn = os.path.basename(cf)
            full_id = os.path.splitext(fn[6:])[0]  # strip "Color_" prefix
            base_id = full_id
            for s in ['_gemini', '_flux', '_seedream']:
                if full_id.endswith(s):
                    base_id = full_id[:-len(s)]
                    break
            sample_ids.add(base_id)

        print(f"\n{data_root}: {len(sample_ids)} unique sample IDs")

        for sample_id in sorted(sample_ids):
            mat_path = os.path.join(data_root, "mat", f"Mats_{sample_id}.txt")
            mesh_path = os.path.join(data_root, "mesh", f"mesh_{sample_id}.txt")
            landmark_path = os.path.join(data_root, "landmark", f"landmark_{sample_id}.txt")

            if not os.path.exists(mat_path):
                continue

            try:
                matrix_data = load_matrix_data(mat_path)
            except Exception as e:
                print(f"  Failed to load matrix for {sample_id}: {e}")
                continue

            # Process mesh
            if os.path.exists(mesh_path):
                mesh_geom = load_geometry(mesh_path)
                if mesh_geom is not None:
                    xyz = mesh_geom[:, 0:3]
                    depth_raw = compute_vertex_depth(xyz, matrix_data)
                    if np.all(np.isfinite(depth_raw)):
                        d_min, d_max = depth_raw.min(), depth_raw.max()
                        d_range = d_max - d_min
                        if d_range > 1e-6:
                            depth_norm = np.clip((depth_raw - d_min) / (d_range + 1e-8), 0.0, 1.0)
                            # Apply index filtering to match template vertex count
                            if mesh_indices is not None:
                                if len(depth_norm) > mesh_indices.max():
                                    depth_norm = depth_norm[mesh_indices]
                                else:
                                    continue
                            if len(depth_norm) == n_mesh:
                                mesh_depth_accum += depth_norm.astype(np.float64)
                                mesh_depth_count += 1

            # Process landmarks
            if os.path.exists(landmark_path):
                lm_geom = load_geometry(landmark_path)
                if lm_geom is not None:
                    xyz = lm_geom[:, 0:3]
                    depth_raw = compute_vertex_depth(xyz, matrix_data)
                    if np.all(np.isfinite(depth_raw)):
                        d_min, d_max = depth_raw.min(), depth_raw.max()
                        d_range = d_max - d_min
                        if d_range > 1e-6:
                            depth_norm = np.clip((depth_raw - d_min) / (d_range + 1e-8), 0.0, 1.0)
                            if landmark_indices is not None:
                                if len(depth_norm) > landmark_indices.max():
                                    depth_norm = depth_norm[landmark_indices]
                                else:
                                    continue
                            if len(depth_norm) == n_landmark:
                                landmark_depth_accum += depth_norm.astype(np.float64)
                                landmark_depth_count += 1

    print(f"\nValid mesh samples: {mesh_depth_count}")
    print(f"Valid landmark samples: {landmark_depth_count}")

    if mesh_depth_count == 0 or landmark_depth_count == 0:
        print("ERROR: No valid samples found. Check data paths.")
        return

    mesh_avg_depth = (mesh_depth_accum / mesh_depth_count).astype(np.float32)
    landmark_avg_depth = (landmark_depth_accum / landmark_depth_count).astype(np.float32)

    print(f"\nMesh avg depth: min={mesh_avg_depth.min():.4f}, max={mesh_avg_depth.max():.4f}, mean={mesh_avg_depth.mean():.4f}")
    print(f"Landmark avg depth: min={landmark_avg_depth.min():.4f}, max={landmark_avg_depth.max():.4f}, mean={landmark_avg_depth.mean():.4f}")

    # Write into template column 5
    # For mesh_template: apply to indexed rows if indices exist, otherwise all rows
    if mesh_indices is not None:
        mesh_template[mesh_indices, 5] = mesh_avg_depth
    else:
        mesh_template[:, 5] = mesh_avg_depth

    if landmark_indices is not None:
        landmark_template[landmark_indices, 5] = landmark_avg_depth
    else:
        landmark_template[:, 5] = landmark_avg_depth

    # Backup originals
    backup_mesh = mesh_template_path + ".bak"
    backup_landmark = landmark_template_path + ".bak"
    if not os.path.exists(backup_mesh):
        np.save(backup_mesh, np.load(mesh_template_path))
        print(f"Backed up to {backup_mesh}")
    if not os.path.exists(backup_landmark):
        np.save(backup_landmark, np.load(landmark_template_path))
        print(f"Backed up to {backup_landmark}")

    np.save(mesh_template_path, mesh_template)
    np.save(landmark_template_path, landmark_template)
    print(f"\nSaved updated templates:")
    print(f"  {mesh_template_path}: shape {mesh_template.shape}")
    print(f"  {landmark_template_path}: shape {landmark_template.shape}")


if __name__ == "__main__":
    main()
