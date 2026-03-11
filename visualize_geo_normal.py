"""
visualize_geo_normal.py
-----------------------
Load the trained GeometryTransformer and run inference on test images.
Saves predicted albedo, geometric normals, and geo UV map renders
plus a 4-panel grid (input | albedo | normal | geo) for visual inspection.

Usage:
    python visualize_geo_normal.py
    python visualize_geo_normal.py --model_path best_geometry_transformer_dim6.pth \
                                   --image_dir test --output_dir vis_output --n 5
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from geometry_transformer import GeometryTransformer
from geometry_train_core import render_mesh_texture_to_image, render_vertex_attrs_to_image
from obj_load_helper import load_uv_obj_file
from train_visualize_helper import load_combined_mesh_uv


# ---------------------------------------------------------------------------
# Helpers re-used from inference_geometry.py
# ---------------------------------------------------------------------------

def load_combined_mesh_triangle_faces(model_dir: str = "model") -> np.ndarray:
    part_files = ["mesh_head.obj", "mesh_eye_l.obj", "mesh_eye_r.obj", "mesh_mouth.obj"]
    tris, offset = [], 0
    for fname in part_files:
        obj_path = os.path.join(model_dir, fname)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing mesh OBJ: {obj_path}")
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(obj_path, triangulate=True)
        tris.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))
    return np.concatenate(tris, axis=0).astype(np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def remap_faces_after_filter(faces: np.ndarray, kept: np.ndarray, orig_n: int) -> np.ndarray:
    """Remap face indices from N_full to N_unique space. Drops faces touching filtered vertices.
    Used only for compute_vertex_normals on N_unique vertices."""
    remap = np.full((int(orig_n),), -1, dtype=np.int64)
    remap[np.asarray(kept, dtype=np.int64)] = np.arange(len(kept), dtype=np.int64)
    tri_new = remap[faces.astype(np.int64)]
    return tri_new[np.all(tri_new >= 0, axis=1)].astype(np.int32)


def load_aux_data():
    template_landmark = np.load("model/landmark_template.npy")
    template_mesh = np.load("model/mesh_template.npy")
    template_mesh_full_n = int(template_mesh.shape[0])

    template_mesh_uv = load_combined_mesh_uv(model_dir="model", copy=True).astype(np.float32)
    template_mesh_uv_full = template_mesh_uv.copy()
    # Full (N_full indexed) faces — used for seamless rendering via mesh_restore_indices expansion
    template_mesh_faces_full = load_combined_mesh_triangle_faces(model_dir="model")
    template_mesh_faces = template_mesh_faces_full.copy()  # will be remapped to N_unique

    landmark_restore_indices = None
    lm_idx_path = "model/landmark_indices.npy"
    if os.path.exists(lm_idx_path):
        lm_idx = np.load(lm_idx_path)
        if lm_idx.max() < template_landmark.shape[0]:
            template_landmark = template_landmark[lm_idx]
        lm_inv_path = "model/landmark_inverse.npy"
        if os.path.exists(lm_inv_path):
            landmark_restore_indices = np.load(lm_inv_path)

    mesh_restore_indices = None
    mesh_idx_path = "model/mesh_indices.npy"
    if os.path.exists(mesh_idx_path):
        mesh_idx = np.load(mesh_idx_path)
        mesh_inv_path = "model/mesh_inverse.npy"
        if os.path.exists(mesh_inv_path):
            mesh_restore_indices = np.load(mesh_inv_path)
        if mesh_idx.max() < template_mesh.shape[0]:
            template_mesh = template_mesh[mesh_idx]
            if mesh_idx.max() < template_mesh_uv.shape[0]:
                template_mesh_uv = template_mesh_uv[mesh_idx]
            template_mesh_faces = remap_faces_after_filter(
                template_mesh_faces_full, mesh_idx, template_mesh_full_n,
            )

    lm2kp_idx = np.load("model/landmark2keypoint_knn_indices.npy")
    lm2kp_w   = np.load("model/landmark2keypoint_knn_weights.npy")
    n_keypoint = int(lm2kp_idx.max()) + 1
    mesh2lm_idx = np.load("model/mesh2landmark_knn_indices.npy")
    mesh2lm_w   = np.load("model/mesh2landmark_knn_weights.npy")

    if template_mesh_uv.shape[0] != template_mesh.shape[0]:
        if template_mesh.shape[1] >= 5:
            template_mesh_uv = template_mesh[:, 3:5].astype(np.float32)
        else:
            template_mesh_uv = np.zeros((template_mesh.shape[0], 2), dtype=np.float32)

    return dict(
        num_landmarks=template_landmark.shape[0],
        num_mesh=template_mesh.shape[0],
        template_landmark=template_landmark,
        template_mesh=template_mesh,
        landmark2keypoint_knn_indices=lm2kp_idx,
        landmark2keypoint_knn_weights=lm2kp_w,
        mesh2landmark_knn_indices=mesh2lm_idx,
        mesh2landmark_knn_weights=mesh2lm_w,
        n_keypoint=n_keypoint,
        template_mesh_uv=template_mesh_uv,
        template_mesh_uv_full=template_mesh_uv_full,
        template_mesh_faces=template_mesh_faces,
        template_mesh_faces_full=template_mesh_faces_full,
        mesh_restore_indices=mesh_restore_indices,
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_and_preprocess(img_path: str, size: int = 512):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (size, size))
    tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return img_rgb, tensor


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float [0,1] → [H, W, 3] uint8 BGR."""
    arr = t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_grid(images: list, labels: list | None = None) -> np.ndarray:
    """Stack images side-by-side, optionally adding a label bar."""
    h, w = images[0].shape[:2]
    canvas = np.concatenate(images, axis=1)
    if labels:
        bar_h = 28
        bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
        for i, lbl in enumerate(labels):
            cv2.putText(bar, lbl, (i * w + 8, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        canvas = np.concatenate([bar, canvas], axis=0)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load auxiliary data
    print("Loading auxiliary model data...")
    aux = load_aux_data()

    # Build model
    print(f"Loading model: {args.model_path}")
    model = GeometryTransformer(
        num_landmarks=aux["num_landmarks"],
        num_mesh=aux["num_mesh"],
        template_landmark=aux["template_landmark"],
        template_mesh=aux["template_mesh"],
        landmark2keypoint_knn_indices=aux["landmark2keypoint_knn_indices"],
        landmark2keypoint_knn_weights=aux["landmark2keypoint_knn_weights"],
        mesh2landmark_knn_indices=aux["mesh2landmark_knn_indices"],
        mesh2landmark_knn_weights=aux["mesh2landmark_knn_weights"],
        n_keypoint=aux["n_keypoint"],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        backbone_weights="imagenet",
        model_type=args.model_type,
        k_bins=args.k_bins,
        simdr_range_3d=(args.simdr_min_3d, args.simdr_max_3d),
        simdr_range_2d=(args.simdr_min_2d, args.simdr_max_2d),
        use_deformable_attention=(args.model_type == "simdr"),
        num_deformable_points=16,
        template_mesh_uv=aux["template_mesh_uv"],
        template_mesh_uv_full=aux["template_mesh_uv_full"],
        template_mesh_faces=aux["template_mesh_faces"],
        template_mesh_faces_full=aux.get("template_mesh_faces_full"),
        mesh_restore_indices=aux.get("mesh_restore_indices"),
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  geo_texture buffer: {model.geo_texture.shape}")

    # Collect images
    image_dir = args.image_dir
    if os.path.isdir(image_dir):
        paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])[:args.n]
    else:
        paths = [image_dir]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(paths)} image(s) → {args.output_dir}")

    decode_mask = [False] * (args.num_layers - 1) + [True]

    for img_path in paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"  {img_path}")

        img_rgb, tensor = load_and_preprocess(img_path, size=512)
        rgb_in = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(rgb_in, decode_texture_mask=decode_mask)

        last = outputs[-1]
        mesh_pred   = last["mesh"]          # [1, N_mesh, output_dim]
        mesh_texture = last["mesh_texture"] # [1, 3, H, W] or None

        # ---- Albedo render ------------------------------------------------
        albedo_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        if mesh_texture is not None:
            render_albedo, _ = render_mesh_texture_to_image(
                model, mesh_pred, mesh_texture, out_h=512, out_w=512, use_pred_depth=True
            )
            if render_albedo is not None:
                albedo_bgr = tensor_to_uint8(render_albedo[0])

        # ---- Normal render ------------------------------------------------
        vertex_normals_01 = model.compute_vertex_normals(mesh_pred[..., :3])  # [1, N, 3]
        render_normal, _ = render_vertex_attrs_to_image(
            model, mesh_pred, vertex_normals_01, out_h=512, out_w=512, use_pred_depth=True
        )
        normal_bgr = tensor_to_uint8(render_normal[0]) if render_normal is not None else np.zeros((512, 512, 3), dtype=np.uint8)

        # ---- Geo render ---------------------------------------------------
        geo_texture = model.geo_texture.unsqueeze(0).expand(1, -1, -1, -1).contiguous()  # [1, 3, 256, 256]
        geo_texture_512 = F.interpolate(geo_texture, size=(512, 512), mode="bilinear", align_corners=False)
        render_geo, _ = render_mesh_texture_to_image(
            model, mesh_pred, geo_texture_512, out_h=512, out_w=512, use_pred_depth=True
        )
        geo_bgr = tensor_to_uint8(render_geo[0]) if render_geo is not None else np.zeros((512, 512, 3), dtype=np.uint8)

        # ---- Input image (BGR) -------------------------------------------
        input_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ---- Save individual images ----------------------------------------
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_input.png"), input_bgr)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_albedo.png"), albedo_bgr)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_normal.png"), normal_bgr)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_geo.png"), geo_bgr)

        # ---- 4-panel grid --------------------------------------------------
        grid = make_grid(
            [input_bgr, albedo_bgr, normal_bgr, geo_bgr],
            labels=["input", "albedo", "normal", "geo"],
        )
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_grid.png"), grid)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_geometry_transformer_dim6.pth")
    parser.add_argument("--image_dir", default="test")
    parser.add_argument("--output_dir", default="vis_output")
    parser.add_argument("--n", type=int, default=5, help="Max number of images to process")
    # Model architecture (must match checkpoint)
    parser.add_argument("--model_type", default="simdr")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--k_bins", type=int, default=256)
    parser.add_argument("--simdr_min_3d", type=float, default=-0.5)
    parser.add_argument("--simdr_max_3d", type=float, default=0.5)
    parser.add_argument("--simdr_min_2d", type=float, default=-0.5)
    parser.add_argument("--simdr_max_2d", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
