"""
test_geo_feature_render.py
--------------------------
Test script: packs per-part EXR geo textures into the combined atlas, then renders
geo / normal / basecolor to image space with nvdiffrast (same pipeline as training).
Saves a 5-panel grid to test_geo_output/.

Usage:
    python test_geo_feature_render.py --data_root G:/CapturedFrames_final8_processed
    python test_geo_feature_render.py --data_root G:/CapturedFrames_final8_processed \
        --sample_idx 3 --output_dir test_geo_output --model_dir model
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import nvdiffrast.torch as dr
    _NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    _NVDIFFRAST_AVAILABLE = False

try:
    import OpenEXR
    import Imath
    _EXR_AVAILABLE = True
except ImportError:
    _EXR_AVAILABLE = False
    print("Warning: OpenEXR not available — pip install openexr")

from geometry_transformer import GeometryTransformer
from geometry_train_core import render_mesh_texture_to_image, render_vertex_attrs_to_image
from metahuman_geometry_dataset import FastGeometryDataset
from obj_load_helper import load_uv_obj_file
from tex_pack_helper import TexturePackHelper


# ---------------------------------------------------------------------------
# EXR helpers
# ---------------------------------------------------------------------------

def load_exr_rgb(path: str) -> np.ndarray:
    """Load an EXR file and return [H, W, 3] float32 RGB."""
    if not _EXR_AVAILABLE:
        raise RuntimeError("OpenEXR is required. Install with: pip install openexr")
    f = OpenEXR.InputFile(path)
    hdr = f.header()
    dw = hdr["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    R = np.frombuffer(f.channel("R", pt), dtype=np.float32).reshape(H, W)
    G = np.frombuffer(f.channel("G", pt), dtype=np.float32).reshape(H, W)
    B = np.frombuffer(f.channel("B", pt), dtype=np.float32).reshape(H, W)
    return np.stack([R, G, B], axis=-1)


def sample_texture_at_uv(tex: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """
    Bilinear-sample a [H, W, 3] float32 texture at UV coordinates [N, 2].
    UV convention: u=0 left, u=1 right, v=0 bottom, v=1 top (OpenGL).
    Returns [N, 3].
    """
    tex_t = torch.from_numpy(tex).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    uvs_t = torch.from_numpy(uvs).float()                                  # [N, 2]

    # grid_sample coords: x=width in [-1,1], y=height in [-1,1]
    # y=-1 is top row; since v=0 is bottom in UV, flip v → grid_y = (1-v)*2-1
    grid_x = uvs_t[:, 0] * 2.0 - 1.0           # [N]
    grid_y = (1.0 - uvs_t[:, 1]) * 2.0 - 1.0  # [N]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [N, 2]
    grid = grid.unsqueeze(0).unsqueeze(0)           # [1, 1, N, 2]

    sampled = F.grid_sample(
        tex_t, grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )  # [1, 3, 1, N]
    return sampled[0, :, 0, :].permute(1, 0).numpy()  # [N, 3]


# ---------------------------------------------------------------------------
# Geo feature loading: EXR → per-vertex
# ---------------------------------------------------------------------------

# Part order must match the OBJ concatenation order used everywhere.
_PART_FILES = [
    ("mesh_head.obj",   "MID_MI_Face_Skin_Baked_LOD0_0_vtx_uv.exr"),
    ("mesh_eye_l.obj",  "MI_EyeL_Baked_vtx_uv.exr"),
    ("mesh_eye_r.obj",  "MI_EyeR_Baked_vtx_uv.exr"),
    ("mesh_mouth.obj",  "MI_Teeth_Baked_vtx_uv.exr"),
]


def load_geo_vertex_features(model_dir: str = "model", mesh_indices=None) -> np.ndarray:
    """
    For each mesh part:
      1. Load per-part vertex UVs from OBJ (local [0,1] UV space).
      2. Load the corresponding EXR geo texture.
      3. Sample EXR at each vertex UV to get per-vertex geo feature [3].
    Concatenates all parts in order, then sub-samples by mesh_indices.
    Returns [N, 3] float32.
    """
    all_feats = []
    for obj_name, exr_name in _PART_FILES:
        obj_path = os.path.join(model_dir, obj_name)
        exr_path = os.path.join(model_dir, exr_name)

        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing OBJ: {obj_path}")
        if not os.path.exists(exr_path):
            raise FileNotFoundError(f"Missing EXR: {exr_path}")

        _, uvs, _, _, _, _ = load_uv_obj_file(obj_path, triangulate=False)
        uvs = np.array(uvs, dtype=np.float32)[:, :2]   # [N_part, 2]

        exr_img = load_exr_rgb(exr_path)                # [H, W, 3]
        feats = sample_texture_at_uv(exr_img, uvs)      # [N_part, 3]
        all_feats.append(feats)
        print(f"  {obj_name}: {len(uvs)} verts, EXR {exr_img.shape}, "
              f"feat range [{feats.min():.3f}, {feats.max():.3f}]")

    geo_all = np.concatenate(all_feats, axis=0).astype(np.float32)  # [N_total, 3]

    if mesh_indices is not None:
        geo_all = geo_all[mesh_indices]

    return geo_all


def load_exr_rgb_alpha(path: str, force_opaque: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.clip(load_exr_rgb(path).astype(np.float32), 0.0, 1.0)
    if force_opaque:
        alpha = np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    else:
        alpha = TexturePackHelper._derive_alpha_from_rgb(rgb)
    return rgb, alpha


def pack_geo_exrs_to_atlas_tex_pack(model_dir: str = "model", atlas_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack source EXRs into the combined atlas using the same crop/resize/paste strategy
    as TexturePackHelper. Returns:
        atlas_rgb: [H, W, 3] float32 in [0, 1]
        coverage:  [H, W, 1] float32 mask where packed atlas has non-empty content
    """
    helper = TexturePackHelper(texture_root=None, mesh_texture_size=atlas_size)

    head_rgb, head_alpha = load_exr_rgb_alpha(os.path.join(model_dir, "MID_MI_Face_Skin_Baked_LOD0_0_vtx_uv.exr"))
    eye_l_rgb, eye_l_alpha = load_exr_rgb_alpha(os.path.join(model_dir, "MI_EyeL_Baked_vtx_uv.exr"), force_opaque=True)
    eye_r_rgb, eye_r_alpha = load_exr_rgb_alpha(os.path.join(model_dir, "MI_EyeR_Baked_vtx_uv.exr"), force_opaque=True)
    mouth_rgb, mouth_alpha = load_exr_rgb_alpha(os.path.join(model_dir, "MI_Teeth_Baked_vtx_uv.exr"))

    tex_size = int(atlas_size)
    head_canvas = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    overlay_canvas = np.zeros((tex_size, tex_size, 3), dtype=np.float32)
    overlay_mask_canvas = np.zeros((tex_size, tex_size, 3), dtype=np.float32)

    head_rgb_r, head_alpha_r = helper._resize_rgba(head_rgb, head_alpha, tex_size, tex_size)
    head_canvas = head_canvas * (1.0 - head_alpha_r) + head_rgb_r * head_alpha_r

    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_canvas,
        src_rgb=eye_l_rgb,
        src_alpha=eye_l_alpha,
        u0=helper.eye_l_u_start,
        v0=helper.bottom_margin,
        box_size=helper.eye_box_size,
        align_bottom=False,
    )
    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_mask_canvas,
        src_rgb=np.ones_like(eye_l_rgb, dtype=np.float32),
        src_alpha=eye_l_alpha,
        u0=helper.eye_l_u_start,
        v0=helper.bottom_margin,
        box_size=helper.eye_box_size,
        align_bottom=False,
    )

    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_canvas,
        src_rgb=eye_r_rgb,
        src_alpha=eye_r_alpha,
        u0=helper.eye_r_u_start,
        v0=helper.bottom_margin,
        box_size=helper.eye_box_size,
        align_bottom=False,
    )
    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_mask_canvas,
        src_rgb=np.ones_like(eye_r_rgb, dtype=np.float32),
        src_alpha=eye_r_alpha,
        u0=helper.eye_r_u_start,
        v0=helper.bottom_margin,
        box_size=helper.eye_box_size,
        align_bottom=False,
    )

    (mouth_high_rgb, mouth_high_alpha), (mouth_low_rgb, mouth_low_alpha) = helper._split_texture_by_v_threshold(
        mouth_rgb,
        mouth_alpha,
        v_threshold=helper.mouth_split_v,
    )

    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_canvas,
        src_rgb=mouth_high_rgb,
        src_alpha=mouth_high_alpha,
        u0=helper.mouth_v_gt_u_start,
        v0=helper.bottom_margin,
        box_size=helper.mouth_box_size,
        align_bottom=True,
    )
    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_mask_canvas,
        src_rgb=np.ones_like(mouth_high_rgb, dtype=np.float32) if mouth_high_rgb is not None else None,
        src_alpha=mouth_high_alpha,
        u0=helper.mouth_v_gt_u_start,
        v0=helper.bottom_margin,
        box_size=helper.mouth_box_size,
        align_bottom=True,
    )

    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_canvas,
        src_rgb=mouth_low_rgb,
        src_alpha=mouth_low_alpha,
        u0=helper.mouth_v_le_u_start,
        v0=helper.bottom_margin,
        box_size=helper.mouth_box_size,
        align_bottom=True,
    )
    helper._paste_texture_into_uv_box(
        canvas_rgb=overlay_mask_canvas,
        src_rgb=np.ones_like(mouth_low_rgb, dtype=np.float32) if mouth_low_rgb is not None else None,
        src_alpha=mouth_low_alpha,
        u0=helper.mouth_v_le_u_start,
        v0=helper.bottom_margin,
        box_size=helper.mouth_box_size,
        align_bottom=True,
    )

    uv_mask = helper._load_combined_uv_layout_mask(tex_size=tex_size)
    if uv_mask is None:
        uv_mask = np.clip(overlay_mask_canvas[..., :1], 0.0, 1.0)

    atlas = overlay_canvas * uv_mask + head_canvas * (1.0 - uv_mask)
    atlas = np.clip(atlas, 0.0, 1.0).astype(np.float32)
    coverage = (atlas.max(axis=2, keepdims=True) > (1.0 / 255.0)).astype(np.float32)
    return atlas, coverage


def load_geo_texture_parts(model_dir: str = "model") -> list[dict]:
    """
    Load per-part source EXR textures plus source/destination UV layouts for atlas baking.
    Source UVs come directly from each OBJ/EXR pair.
    Destination UVs come from the combined packed UV layout used by training/rendering.
    """
    from train_visualize_helper import load_combined_mesh_uv

    combined_uv = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32)
    parts: list[dict] = []
    uv_cursor = 0

    for obj_name, exr_name in _PART_FILES:
        obj_path = os.path.join(model_dir, obj_name)
        exr_path = os.path.join(model_dir, exr_name)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing OBJ: {obj_path}")
        if not os.path.exists(exr_path):
            raise FileNotFoundError(f"Missing EXR: {exr_path}")

        _, uvs, _, vertex_faces, uv_faces, _ = load_uv_obj_file(obj_path, triangulate=True)
        if uvs is None:
            raise ValueError(f"OBJ has no UVs: {obj_path}")
        if uv_faces is None and vertex_faces is None:
            raise ValueError(f"OBJ has no valid faces: {obj_path}")

        src_uv = np.asarray(uvs[:, :2], dtype=np.float32)
        uv_count = int(src_uv.shape[0])
        dst_uv = combined_uv[uv_cursor : uv_cursor + uv_count]
        if dst_uv.shape[0] != uv_count:
            raise ValueError(
                f"Combined UV slice mismatch for {obj_name}: expected {uv_count}, got {dst_uv.shape[0]}"
            )
        uv_cursor += uv_count

        tri = np.asarray(uv_faces if uv_faces is not None else vertex_faces, dtype=np.int32)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError(f"Triangulated UV faces required for {obj_path}, got {tri.shape}")

        exr_img = load_exr_rgb(exr_path).astype(np.float32)
        parts.append(
            {
                "obj_name": obj_name,
                "exr_name": exr_name,
                "src_uv": src_uv,
                "dst_uv": dst_uv,
                "tri": tri,
                "texture": exr_img,
            }
        )

    if uv_cursor != int(combined_uv.shape[0]):
        raise ValueError(f"Combined UV length mismatch: consumed {uv_cursor}, total {combined_uv.shape[0]}")

    return parts


def texture_map_to_packed_atlas(
    texture: np.ndarray,
    src_uv: np.ndarray,
    dst_uv: np.ndarray,
    tri: np.ndarray,
    out_size: int,
    device: torch.device,
    ctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Render a source EXR texture into the packed atlas using true UV texture mapping.
    Returns:
        color: [1, 3, H, W]
        coverage: [1, 1, H, W]
    """
    if not _NVDIFFRAST_AVAILABLE:
        raise RuntimeError("nvdiffrast is required for texture-mapped atlas baking.")
    if device.type != "cuda":
        raise RuntimeError(f"Texture-mapped atlas baking requires CUDA, got {device}.")

    tex = np.asarray(texture, dtype=np.float32)
    if tex.ndim != 3 or tex.shape[2] != 3:
        raise ValueError(f"texture must be [H, W, 3], got {tex.shape}")
    tex = np.nan_to_num(tex, nan=0.0, posinf=1.0, neginf=0.0)
    tex = np.clip(tex, 0.0, 1.0)

    src_uv = np.asarray(src_uv, dtype=np.float32)
    dst_uv = np.asarray(dst_uv, dtype=np.float32)
    tri = np.asarray(tri, dtype=np.int32)
    if src_uv.shape != dst_uv.shape or src_uv.ndim != 2 or src_uv.shape[1] != 2:
        raise ValueError(f"src_uv and dst_uv must both be [N, 2], got {src_uv.shape} and {dst_uv.shape}")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError(f"tri must be [F, 3], got {tri.shape}")

    x_clip = np.clip(dst_uv[:, 0], 0.0, 1.0) * 2.0 - 1.0
    y_clip = 1.0 - (np.clip(dst_uv[:, 1], 0.0, 1.0) * 2.0)
    clip_pos = np.stack(
        [
            np.nan_to_num(x_clip, nan=0.0, posinf=2.0, neginf=-2.0),
            np.nan_to_num(y_clip, nan=0.0, posinf=2.0, neginf=-2.0),
            np.zeros_like(x_clip, dtype=np.float32),
            np.ones_like(x_clip, dtype=np.float32),
        ],
        axis=-1,
    )

    pos_t = torch.from_numpy(clip_pos).to(device=device, dtype=torch.float32)[None, ...].contiguous()
    tri_t = torch.from_numpy(tri).to(device=device, dtype=torch.int32).contiguous()
    src_uv_t = torch.from_numpy(np.clip(src_uv, 0.0, 1.0)).to(device=device, dtype=torch.float32)[None, ...].contiguous()
    tex_t = torch.from_numpy(tex).to(device=device, dtype=torch.float32)[None, ...].contiguous()

    with torch.amp.autocast(device_type="cuda", enabled=False):
        rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[int(out_size), int(out_size)])
        uv_pix, _ = dr.interpolate(src_uv_t, rast, tri_t)
        uv_pix = torch.stack([uv_pix[..., 0], 1.0 - uv_pix[..., 1]], dim=-1).clamp(0.0, 1.0)
        color = dr.texture(tex_t, uv_pix, filter_mode="linear", boundary_mode="clamp")
        coverage = (rast[..., 3:4] > 0).to(dtype=color.dtype)
        color = color * coverage

    color = color.permute(0, 3, 1, 2).contiguous()
    coverage = coverage.permute(0, 3, 1, 2).contiguous()
    return color, coverage


# ---------------------------------------------------------------------------
# Aux data + model (same pattern as visualize_geo_normal.py)
# ---------------------------------------------------------------------------

def load_aux_and_model(model_dir: str = "model", device=None):
    from train_visualize_helper import load_combined_mesh_uv

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    template_landmark = np.load(os.path.join(model_dir, "landmark_template.npy"))
    template_mesh     = np.load(os.path.join(model_dir, "mesh_template.npy"))
    template_mesh_full_n = int(template_mesh.shape[0])
    template_mesh_uv  = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32)
    template_mesh_uv_full = template_mesh_uv.copy()

    # Load faces — keep full (N_full indexed) version for seamless rendering
    from visualize_geo_normal import load_combined_mesh_triangle_faces, remap_faces_after_filter
    template_mesh_faces_full = load_combined_mesh_triangle_faces(model_dir=model_dir)
    template_mesh_faces = template_mesh_faces_full.copy()  # will be remapped to N_unique

    mesh_indices = None
    mesh_restore_indices = None
    lm_idx_path = os.path.join(model_dir, "landmark_indices.npy")
    lm_idx = np.load(lm_idx_path) if os.path.exists(lm_idx_path) else None
    if lm_idx is not None and lm_idx.max() < template_landmark.shape[0]:
        template_landmark = template_landmark[lm_idx]

    mesh_idx_path = os.path.join(model_dir, "mesh_indices.npy")
    if os.path.exists(mesh_idx_path):
        mesh_idx = np.load(mesh_idx_path)
        mesh_indices = mesh_idx
        mesh_inv_path = os.path.join(model_dir, "mesh_inverse.npy")
        if os.path.exists(mesh_inv_path):
            mesh_restore_indices = np.load(mesh_inv_path)
        if mesh_idx.max() < template_mesh.shape[0]:
            template_mesh = template_mesh[mesh_idx]
            if mesh_idx.max() < template_mesh_uv.shape[0]:
                template_mesh_uv = template_mesh_uv[mesh_idx]
            template_mesh_faces = remap_faces_after_filter(
                template_mesh_faces_full, mesh_idx, template_mesh_full_n
            )

    lm2kp_idx = np.load(os.path.join(model_dir, "landmark2keypoint_knn_indices.npy"))
    lm2kp_w   = np.load(os.path.join(model_dir, "landmark2keypoint_knn_weights.npy"))
    n_keypoint = int(lm2kp_idx.max()) + 1
    mesh2lm_idx = np.load(os.path.join(model_dir, "mesh2landmark_knn_indices.npy"))
    mesh2lm_w   = np.load(os.path.join(model_dir, "mesh2landmark_knn_weights.npy"))

    if template_mesh_uv.shape[0] != template_mesh.shape[0]:
        if template_mesh.shape[1] >= 5:
            template_mesh_uv = template_mesh[:, 3:5].astype(np.float32)
        else:
            template_mesh_uv = np.zeros((template_mesh.shape[0], 2), dtype=np.float32)

    model = GeometryTransformer(
        num_landmarks=template_landmark.shape[0],
        num_mesh=template_mesh.shape[0],
        template_landmark=template_landmark,
        template_mesh=template_mesh,
        landmark2keypoint_knn_indices=lm2kp_idx,
        landmark2keypoint_knn_weights=lm2kp_w,
        mesh2landmark_knn_indices=mesh2lm_idx,
        mesh2landmark_knn_weights=mesh2lm_w,
        n_keypoint=n_keypoint,
        d_model=256,
        nhead=8,
        num_layers=4,
        backbone_weights="none",
        model_type="simdr",
        k_bins=256,
        template_mesh_uv=template_mesh_uv,
        template_mesh_uv_full=template_mesh_uv_full,
        template_mesh_faces=template_mesh_faces,
        template_mesh_faces_full=template_mesh_faces_full,
        mesh_restore_indices=mesh_restore_indices,
    ).to(device)
    model.eval()

    return model, mesh_indices, device


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

def floodfill_replace_black_pixels(img: torch.Tensor) -> torch.Tensor:
    """
    Replace near-black pixels with the average color of their non-black neighbors.
    img: [B, H, W, 3] float32
    Returns: [B, H, W, 3] float32
    """
    black_mask = (torch.abs(img) < 0.02).all(dim=-1)       # [B, H, W]

    padded_img  = torch.nn.functional.pad(img,        (0, 0, 1, 1, 1, 1, 0, 0)).permute(0, 3, 1, 2)
    black_mask  = torch.nn.functional.pad(black_mask, (1, 1, 1, 1, 0, 0))

    non_black_mask = ~black_mask
    kernel = torch.tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]], dtype=torch.float32).to(img.device)

    sum_neighbors   = torch.nn.functional.conv2d(padded_img,            kernel.repeat(3, 1, 1, 1), padding=0, groups=3)
    count_neighbors = torch.nn.functional.conv2d(non_black_mask.float().unsqueeze(1), kernel,              padding=0)

    count_neighbors[count_neighbors == 0] = 1
    avg_color = sum_neighbors / count_neighbors

    black_mask = black_mask[:, 1:-1, 1:-1].unsqueeze(1).repeat(1, 3, 1, 1)

    out_img = img.permute(0, 3, 1, 2).clone()
    out_img[black_mask] = avg_color[black_mask]
    return out_img.permute(0, 2, 3, 1)


def floodfill_n(img: torch.Tensor, n: int = 4) -> torch.Tensor:
    """Apply floodfill_replace_black_pixels n times."""
    for _ in range(n):
        img = floodfill_replace_black_pixels(img)
    return img


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def tensor_to_uint8_bgr(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float [0,1] → [H, W, 3] uint8 BGR."""
    arr = t.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
    return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def na_placeholder(h: int = 512, w: int = 512, text: str = "N/A") -> np.ndarray:
    """Dark grey tile with centred text, used when a GT image is unavailable."""
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    lines = text.split("\n")
    line_h = int(scale * 30)
    y0 = h // 2 - line_h * (len(lines) - 1) // 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, font, scale, thick)
        cv2.putText(img, line, ((w - tw) // 2, y0 + i * line_h + th // 2),
                    font, scale, (160, 160, 160), thick, cv2.LINE_AA)
    return img


def make_row(images: list, label: str | None = None, cell_w: int = 0) -> np.ndarray:
    """Concatenate images horizontally, prepend an optional row label cell."""
    h = images[0].shape[0]
    if cell_w <= 0:
        cell_w = images[0].shape[1]
    row = np.concatenate(images, axis=1)
    if label:
        tag = np.zeros((h, cell_w, 3), dtype=np.uint8)
        cv2.putText(tag, label, (6, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
        row = np.concatenate([tag, row], axis=1)
    return row


def make_grid(rows: list[np.ndarray], col_labels: list[str] | None = None,
              cell_w: int = 512) -> np.ndarray:
    """
    Stack rows vertically.  Optionally add a column-header bar above the first row.
    col_labels should include an entry for the row-label column if rows have one.
    """
    if col_labels:
        bar_h = 28
        bar = np.zeros((bar_h, rows[0].shape[1], 3), dtype=np.uint8)
        for i, lbl in enumerate(col_labels):
            cv2.putText(bar, lbl, (i * cell_w + 8, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        rows = [bar] + rows
    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Aux data + model (no checkpoint needed, just uses rasterization utils)
    print("Building model for rasterization utilities...")
    model, mesh_indices, device = load_aux_and_model(args.model_dir, device)
    print(f"  num_mesh={model.num_mesh}, texture_size={model.texture_feature_map_size}")

    # --- Build geo atlas from source EXRs via direct tex-pack style composition
    print("Building geo atlas from EXR files via tex-pack composition...")
    GEO_ATLAS_SIZE = 2048
    geo_uv_map_np, geo_cov_map_np = pack_geo_exrs_to_atlas_tex_pack(args.model_dir, atlas_size=GEO_ATLAS_SIZE)
    geo_uv_map = torch.from_numpy(geo_uv_map_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    geo_cov_map = torch.from_numpy(geo_cov_map_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    print(f"  geo_uv_map (raw):  {tuple(geo_uv_map.shape)}, "
          f"range [{geo_uv_map.min():.3f}, {geo_uv_map.max():.3f}]")

    # Swap R and B channels to match GT convention
    geo_uv_packed_bchw = torch.stack(
        [geo_uv_map[:, 2], geo_uv_map[:, 1], geo_uv_map[:, 0]], dim=1
    )

    # Mark empty atlas background red for u<0.5 and green for u>=0.5 for debugging.
    red_bg_mask = (geo_cov_map <= 0).bool() & (torch.abs(geo_uv_packed_bchw) < 0.02).all(dim=1, keepdim=True)
    if bool(red_bg_mask.any().item()):
        _, _, _, atlas_w = geo_uv_packed_bchw.shape
        u_mask = torch.linspace(
            0.0,
            1.0,
            atlas_w,
            device=geo_uv_packed_bchw.device,
            dtype=geo_uv_packed_bchw.dtype,
        ).view(1, 1, 1, atlas_w)
        debug_bg = torch.zeros_like(geo_uv_packed_bchw)
        debug_bg[:, 0, :, :] = (u_mask < 0.5).to(dtype=geo_uv_packed_bchw.dtype)
        debug_bg[:, 1, :, :] = (u_mask >= 0.5).to(dtype=geo_uv_packed_bchw.dtype)
        geo_uv_packed_bchw = torch.where(red_bg_mask, debug_bg, geo_uv_packed_bchw)

    # Keep the packed atlas at full resolution for both rendering and saving.
    atlas_np = geo_uv_packed_bchw[0].cpu().float().numpy()  # [3, 2048, 2048]
    if atlas_np.shape != (3, GEO_ATLAS_SIZE, GEO_ATLAS_SIZE):
        raise RuntimeError(
            f"Unexpected atlas shape {atlas_np.shape}; expected (3, {GEO_ATLAS_SIZE}, {GEO_ATLAS_SIZE})"
        )
    npy_path = os.path.join(args.model_dir, "geo_feature_atlas.npy")
    np.save(npy_path, atlas_np)
    print(f"  Saved geo_feature_atlas.npy → {npy_path}  shape={atlas_np.shape}")

    # Also save as EXR for visual inspection in DCC tools
    if _EXR_AVAILABLE:
        exr_path = os.path.join(args.model_dir, "geo_feature_atlas.exr")
        H_exr, W_exr = atlas_np.shape[1], atlas_np.shape[2]
        exr_out = OpenEXR.OutputFile(
            exr_path,
            OpenEXR.Header(W_exr, H_exr),
        )
        exr_out.writePixels({
            "R": atlas_np[0].astype(np.float32).tobytes(),
            "G": atlas_np[1].astype(np.float32).tobytes(),
            "B": atlas_np[2].astype(np.float32).tobytes(),
        })
        exr_out.close()
        print(f"  Saved geo_feature_atlas.exr  → {exr_path}")
    else:
        print("  (skipping EXR save — OpenEXR not available)")

    # Save PNG previews for quick inspection
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(args.output_dir, "geo_uv_atlas_raw.png"),
        tensor_to_uint8_bgr(geo_uv_map[0]),
    )
    cv2.imwrite(
        os.path.join(args.output_dir, "geo_uv_atlas_filled.png"),
        tensor_to_uint8_bgr(geo_uv_packed_bchw[0]),
    )
    print(f"  Saved PNG previews → {args.output_dir}/geo_uv_atlas_{{raw,filled}}.png")

    # --- Dataset
    print(f"Loading dataset from: {args.data_root}")
    dataset = FastGeometryDataset(
        data_roots=[args.data_root],
        split="val",
        image_size=512,
        augment=False,
    )
    print(f"  {len(dataset)} samples in val split")
    if len(dataset) == 0:
        dataset = FastGeometryDataset(
            data_roots=[args.data_root],
            split="train",
            image_size=512,
            augment=False,
        )
        print(f"  (switched to train split: {len(dataset)} samples)")

    total = len(dataset)
    if args.num_samples > 1:
        step = max(1, total // args.num_samples)
        indices = [min(args.sample_idx + i * step, total - 1) for i in range(args.num_samples)]
    else:
        indices = [min(args.sample_idx, total - 1)]
    print(f"  Rendering {len(indices)} sample(s): indices {indices}")

    def _to_bgr(t: torch.Tensor) -> np.ndarray:
        arr = t.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
        return cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _diff_vis_bgr(gt_t: torch.Tensor | None, pred_t: torch.Tensor | None, scale: float) -> np.ndarray:
        if gt_t is None or pred_t is None:
            return na_placeholder(text="diff\nN/A")
        if gt_t.ndim == 4:
            gt_t = gt_t[0]
        if pred_t.ndim == 4:
            pred_t = pred_t[0]
        diff_t = (gt_t.detach().cpu().float() - pred_t.detach().cpu().float()).abs() * float(scale)
        return _to_bgr(diff_t.clamp(0.0, 1.0))

    # Pre-fetch full face topology from model buffers (expanded via restore mapping)
    restore_idx_np  = model.mesh_restore_indices.cpu().numpy()        # [N_full]
    faces_full_np   = model.template_mesh_faces_full.cpu().numpy()    # [F, 3]

    def _save_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
        """Write a simple OBJ file. verts: [N,3], faces: [F,3] 0-based."""
        with open(path, "w") as f:
            f.write("# ProHead mesh export\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in faces:
                f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    for idx in indices:
        sample = dataset[idx]
        print(f"\n--- Sample #{idx}: {sample['image_path']}")

        # 1. Color (aligned RGB input)
        input_bgr = _to_bgr(sample["rgb"])

        # 2. Basecolor GT
        bc_valid  = float(sample["basecolor_valid"].item())
        bc_gt_t = sample["basecolor"]
        bc_gt_bgr = _to_bgr(sample["basecolor"])
        normal_valid_tensor = sample.get("normal_valid")
        normal_gt_tensor = sample.get("normal_gt")
        normal_valid = float(normal_valid_tensor.item()) if normal_valid_tensor is not None else 0.0
        print(f"  basecolor_valid={bc_valid:.0f}  normal_valid={normal_valid:.0f}  geo_valid={float(sample['geo_valid'].item()):.0f}")

        # 3. Normal GT
        normal_gt_bgr = (
            _to_bgr(normal_gt_tensor)
            if normal_gt_tensor is not None and normal_valid > 0.5
            else na_placeholder(text="normal GT\ndisabled")
        )

        # 4. Geo GT
        geo_valid  = float(sample["geo_valid"].item())
        geo_gt_t = sample["geo_gt"]
        geo_gt_bgr = (_to_bgr(geo_gt_t)
                      if geo_valid > 0.5 else na_placeholder(text="geo GT\nnot found"))

        # Mesh + texture
        mesh_gt   = sample["mesh"].unsqueeze(0).to(device)
        mesh_tex  = sample["mesh_texture"].unsqueeze(0).to(device)
        tex_valid = float(sample["mesh_texture_valid"].item())

        # --- Render basecolor
        with torch.no_grad():
            if tex_valid > 0.5:
                mesh_tex_512 = F.interpolate(mesh_tex, size=(512, 512),
                                             mode="bilinear", align_corners=False)
                render_bc, _ = render_mesh_texture_to_image(
                    model, mesh_gt, mesh_tex_512, out_h=512, out_w=512, use_pred_depth=True
                )
                bc_bgr = tensor_to_uint8_bgr(render_bc[0]) if render_bc is not None else \
                    np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                bc_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
                render_bc = None

        # --- Render normal
        with torch.no_grad():
            vertex_normals = model.compute_vertex_normals(mesh_gt[..., :3])
            render_normal, _ = render_vertex_attrs_to_image(
                model, mesh_gt, vertex_normals, out_h=512, out_w=512, use_pred_depth=True
            )
            normal_bgr = tensor_to_uint8_bgr(render_normal[0]) if render_normal is not None else \
                np.zeros((512, 512, 3), dtype=np.uint8)

        # --- Render geo
        with torch.no_grad():
            render_geo, _ = render_mesh_texture_to_image(
                model, mesh_gt, geo_uv_packed_bchw, out_h=512, out_w=512, use_pred_depth=True
            )
            geo_bgr = tensor_to_uint8_bgr(render_geo[0]) if render_geo is not None else \
                np.zeros((512, 512, 3), dtype=np.uint8)

        bc_diff_bgr = (
            _diff_vis_bgr(bc_gt_t, render_bc, args.diff_scale)
            if bc_valid > 0.5 and render_bc is not None
            else na_placeholder(text="base diff\nN/A")
        )
        normal_diff_bgr = (
            _diff_vis_bgr(normal_gt_tensor, render_normal, args.diff_scale)
            if normal_gt_tensor is not None and normal_valid > 0.5 and render_normal is not None
            else na_placeholder(text="normal diff\nN/A")
        )
        geo_diff_bgr = (
            _diff_vis_bgr(geo_gt_t, render_geo, args.diff_scale)
            if geo_valid > 0.5 and render_geo is not None
            else na_placeholder(text="geo diff\nN/A")
        )

        stem = f"sample_{idx:04d}"

        # --- Save OBJ meshes (full topology, expanded via restore mapping)
        mesh_np = mesh_gt[0].cpu().numpy()                          # [N_unique, 6]
        mesh_full_np = mesh_np[restore_idx_np]                      # [N_full, 6]

        # 3D mesh: x, y, z from columns 0:3
        _save_obj(
            os.path.join(args.output_dir, f"{stem}_mesh3d.obj"),
            mesh_full_np[:, :3],
            faces_full_np,
        )
        # 2D mesh: u, v from columns 3:5, z=depth from column 5 (for z-ordering)
        verts_2d = np.stack([
            mesh_full_np[:, 3],          # u  → x
            1.0 - mesh_full_np[:, 4],    # 1-v → y  (flip to match image space)
            -mesh_full_np[:, 5],         # depth → z (negative = forward)
        ], axis=-1)
        _save_obj(
            os.path.join(args.output_dir, f"{stem}_mesh2d.obj"),
            verts_2d,
            faces_full_np,
        )
        print(f"  → {stem}_mesh3d.obj  {stem}_mesh2d.obj  (N_full={len(mesh_full_np)}, F={len(faces_full_np)})")

        # --- Grid: render row + gt row + abs-diff row
        W = 512
        row_render = make_row([input_bgr, bc_bgr,    normal_bgr,    geo_bgr],
                              label="render", cell_w=W)
        row_gt     = make_row([input_bgr, bc_gt_bgr, normal_gt_bgr, geo_gt_bgr],
                              label="gt",     cell_w=W)
        row_diff   = make_row(
            [na_placeholder(text="abs diff"), bc_diff_bgr, normal_diff_bgr, geo_diff_bgr],
            label=f"diff x{args.diff_scale:g}",
            cell_w=W,
        )
        grid = make_grid([row_render, row_gt, row_diff],
                         col_labels=["", "input", "basecolor", "normal", "geo"], cell_w=W)
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_grid.png"), grid)
        print(f"  → {stem}_grid.png")

    print(f"\nDone. All outputs in {args.output_dir}/")


DATA_ROOT = "G:/CapturedFrames_final8_processed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help="Path to dataset folder")
    parser.add_argument("--sample_idx", type=int, default=0,
                        help="Starting sample index")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of samples to render (spread evenly from sample_idx)")
    parser.add_argument("--output_dir", default="test_geo_output")
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--diff_scale", type=float, default=10.0,
                        help="Scale factor applied to abs(gt-render) before visualization")
    args = parser.parse_args()
    main(args)
