import argparse
import os
from pathlib import Path
import subprocess
from types import MethodType
from types import ModuleType

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import nvdiffrast.torch as dr
    _NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    _NVDIFFRAST_AVAILABLE = False

from align_5pt_helper import Align5PtHelper
from dense2geometry import Dense2Geometry, DenseStageConfig, _load_filtered_template_mesh, _load_matching_state_dict
from train_visualize_helper import create_combined_overlay, load_mesh_topology

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_DR_CONTEXT_CACHE: dict[str, object] = {}


def preprocess_aligned_rgb(img_rgb: np.ndarray, image_size: int) -> np.ndarray:
    x = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    # Dense2Geometry normalizes to ImageNet stats inside _run_dense_stage().
    return x.transpose(2, 0, 1)


def transform_geometry_2d(
    geom: np.ndarray,
    m_inv: np.ndarray,
    aligned_size: int,
    original_w: int,
    original_h: int,
) -> np.ndarray:
    result = geom.copy()
    uv_norm = geom[:, 3:5]
    uv_px = uv_norm * float(aligned_size)
    ones = np.ones((len(uv_px), 1), dtype=np.float32)
    uv_h = np.concatenate([uv_px, ones], axis=1)
    uv_orig_px = (m_inv @ uv_h.T).T
    result[:, 3] = uv_orig_px[:, 0] / float(max(original_w, 1))
    result[:, 4] = uv_orig_px[:, 1] / float(max(original_h, 1))
    return result


def compose_affine(second: np.ndarray, first: np.ndarray) -> np.ndarray:
    second_h = np.eye(3, dtype=np.float32)
    first_h = np.eye(3, dtype=np.float32)
    second_h[:2, :] = np.asarray(second, dtype=np.float32)
    first_h[:2, :] = np.asarray(first, dtype=np.float32)
    composed = second_h @ first_h
    return composed[:2, :].astype(np.float32)


def maybe_resize_for_alignment(
    img_rgb: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img_rgb.shape[:2]
    if h == target_size and w == target_size:
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return img_rgb, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), identity

    resized_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    pre_align_affine = np.array(
        [
            [float(target_size) / float(max(w, 1)), 0.0, 0.0],
            [0.0, float(target_size) / float(max(h, 1)), 0.0],
        ],
        dtype=np.float32,
    )
    return resized_rgb, cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR), pre_align_affine


def save_mesh_obj(
    path: str,
    vertices: np.ndarray,
    topology: dict,
    restore_indices: np.ndarray | None = None,
) -> None:
    verts = np.asarray(vertices, dtype=np.float32)
    if restore_indices is not None:
        verts = verts[np.asarray(restore_indices, dtype=np.int64)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Dense2Geometry prediction\n")
        for v in verts:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for name, data in topology.items():
            start_idx = int(data["start_idx"])
            f.write(f"g {name}\n")
            for face in data["faces"]:
                indices_str = " ".join(str(int(idx) + start_idx + 1) for idx in face)
                f.write(f"f {indices_str}\n")


def build_full_mesh_faces(topology: dict) -> np.ndarray:
    faces_full: list[np.ndarray] = []
    for _, data in topology.items():
        faces = np.asarray(data["faces"], dtype=np.int32)
        if faces.size <= 0:
            continue
        start_idx = int(data["start_idx"])
        faces_full.append(faces + start_idx)
    if not faces_full:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(faces_full, axis=0).astype(np.int32, copy=False)


def restore_full_vertices(vertices: np.ndarray, restore_indices: np.ndarray | None = None) -> np.ndarray:
    verts = np.asarray(vertices)
    if restore_indices is None:
        return verts.copy()
    return verts[np.asarray(restore_indices, dtype=np.int64)]


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


_5PT_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 150, 255),
    (255, 200, 50),
    (220, 80, 220),
]
_5PT_LABELS = ["R_Eye", "L_Eye", "Nose", "Mouth_R", "Mouth_L"]


def draw_5pt(image_rgb: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    for pt, color, label in zip(pts_px, _5PT_COLORS, _5PT_LABELS):
        if not np.isfinite(pt).all():
            continue
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        cv2.circle(out, (x, y), 8, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (x + 12, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
    )
    return out


def compose_panel_grid(panels: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not panels:
        raise ValueError("Expected at least one panel to compose.")
    cols = max(1, int(cols))
    panel_h, panel_w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    grid = np.zeros((rows * panel_h, cols * panel_w, 3), dtype=panels[0].dtype)
    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        y0 = row * panel_h
        x0 = col * panel_w
        grid[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
    return grid


def draw_vertex_points(
    image_rgb: np.ndarray,
    pts_px: np.ndarray,
    color: tuple[int, int, int] = (80, 255, 255),
) -> np.ndarray:
    out = image_rgb.copy()
    pts = np.asarray(pts_px, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return out
    valid = np.isfinite(pts).all(axis=1)
    if not np.any(valid):
        return out
    xy = np.rint(pts[valid]).astype(np.int32, copy=False)
    h, w = out.shape[:2]
    xy[:, 0] = np.clip(xy[:, 0], 0, max(w - 1, 0))
    xy[:, 1] = np.clip(xy[:, 1], 0, max(h - 1, 0))
    for x, y in xy:
        cv2.circle(out, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)
    return out


def draw_refine_displacement_overlay(
    image_rgb: np.ndarray,
    pre_pts_px: np.ndarray,
    post_pts_px: np.ndarray,
    max_color_px: float,
    max_arrows: int = 300,
) -> np.ndarray:
    out = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    pre = np.asarray(pre_pts_px, dtype=np.float32)
    post = np.asarray(post_pts_px, dtype=np.float32)
    if pre.ndim != 2 or post.ndim != 2 or pre.shape != post.shape or pre.shape[1] != 2:
        return image_rgb.copy()

    valid = np.isfinite(pre).all(axis=1) & np.isfinite(post).all(axis=1)
    if not np.any(valid):
        return image_rgb.copy()

    delta = post - pre
    mag = np.linalg.norm(delta, axis=1)
    all_valid_idx = np.flatnonzero(valid)
    if all_valid_idx.size == 0:
        return image_rgb.copy()

    denom = float(max(max_color_px, 1e-6))
    h, w = out.shape[:2]

    # Dense point heat: color every valid vertex by displacement magnitude.
    for idx in all_valid_idx:
        px1, py1 = post[idx]
        if not np.isfinite(px1 + py1):
            continue
        x1 = int(np.clip(round(float(px1)), 0, max(w - 1, 0)))
        y1 = int(np.clip(round(float(py1)), 0, max(h - 1, 0)))
        t = float(np.clip(mag[idx] / denom, 0.0, 1.0))
        color = tuple(
            int(v)
            for v in cv2.applyColorMap(
                np.array([[int(round(255.0 * t))]], dtype=np.uint8),
                cv2.COLORMAP_JET,
            )[0, 0]
        )
        cv2.circle(out, (x1, y1), 1, color, -1, cv2.LINE_AA)

    # Sparse arrows for the largest movers so direction stays readable.
    valid_idx = all_valid_idx
    if valid_idx.size > int(max_arrows):
        order = np.argsort(mag[valid_idx])[::-1]
        valid_idx = valid_idx[order[: int(max_arrows)]]

    for idx in valid_idx:
        px0, py0 = pre[idx]
        px1, py1 = post[idx]
        if not np.isfinite(px0 + py0 + px1 + py1):
            continue
        x0 = int(np.clip(round(float(px0)), 0, max(w - 1, 0)))
        y0 = int(np.clip(round(float(py0)), 0, max(h - 1, 0)))
        x1 = int(np.clip(round(float(px1)), 0, max(w - 1, 0)))
        y1 = int(np.clip(round(float(py1)), 0, max(h - 1, 0)))
        t = float(np.clip(mag[idx] / denom, 0.0, 1.0))
        color = tuple(int(v) for v in cv2.applyColorMap(np.array([[int(round(255.0 * t))]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0])
        cv2.arrowedLine(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(out, (x1, y1), 1, color, -1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def compute_vertex_normals_np(mesh_xyz: np.ndarray, faces_full: np.ndarray) -> np.ndarray:
    xyz = np.asarray(mesh_xyz, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces_full, dtype=np.int32).reshape(-1, 3)
    if xyz.shape[0] <= 0 or faces.shape[0] <= 0:
        return np.zeros_like(xyz, dtype=np.float32)

    valid_faces = (
        (faces[:, 0] >= 0)
        & (faces[:, 1] >= 0)
        & (faces[:, 2] >= 0)
        & (faces[:, 0] < xyz.shape[0])
        & (faces[:, 1] < xyz.shape[0])
        & (faces[:, 2] < xyz.shape[0])
    )
    faces = faces[valid_faces]
    if faces.shape[0] <= 0:
        return np.zeros_like(xyz, dtype=np.float32)

    v0 = xyz[faces[:, 0]]
    v1 = xyz[faces[:, 1]]
    v2 = xyz[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32, copy=False)

    vertex_normals = np.zeros_like(xyz, dtype=np.float32)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norm = np.where(norm < 1e-6, 1.0, norm)
    return (vertex_normals / norm).astype(np.float32, copy=False)


def draw_normal_points_fallback(
    projected_px: np.ndarray,
    normals_rgb01: np.ndarray,
    image_size: int,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    canvas = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
    pts = np.asarray(projected_px, dtype=np.float32)
    colors = np.asarray(normals_rgb01, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or colors.ndim != 2 or colors.shape[0] != pts.shape[0]:
        return canvas

    scale_x = float(max(out_w, 1)) / float(max(int(image_size), 1))
    scale_y = float(max(out_h, 1)) / float(max(int(image_size), 1))
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(colors).all(axis=1)
    for idx in np.flatnonzero(valid):
        x = int(np.clip(round(float(pts[idx, 0] * scale_x)), 0, max(out_w - 1, 0)))
        y = int(np.clip(round(float(pts[idx, 1] * scale_y)), 0, max(out_h - 1, 0)))
        rgb = np.clip(colors[idx] * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.circle(canvas, (x, y), 1, tuple(int(v) for v in rgb.tolist()), -1, cv2.LINE_AA)
    return canvas


def render_mesh_screen_normals(
    mesh_xyz_full: np.ndarray,
    projected_px_full: np.ndarray,
    faces_full: np.ndarray,
    image_size: int,
    out_h: int,
    out_w: int,
    device: torch.device,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> np.ndarray:
    xyz = np.asarray(mesh_xyz_full, dtype=np.float32).reshape(-1, 3)
    proj_px = np.asarray(projected_px_full, dtype=np.float32).reshape(-1, 2)
    faces = np.asarray(faces_full, dtype=np.int32).reshape(-1, 3)
    if xyz.shape[0] <= 0 or proj_px.shape[0] != xyz.shape[0] or faces.shape[0] <= 0:
        return np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)

    normals_world = compute_vertex_normals_np(xyz, faces)
    cam_xyz = xyz.copy()
    normals_cam = normals_world.copy()
    if rvec is not None:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
        cam_xyz = xyz @ R.T
        normals_cam = normals_world @ R.T
        if tvec is not None:
            cam_xyz = cam_xyz + np.asarray(tvec, dtype=np.float32).reshape(1, 3)

    normals_vis = normals_cam.copy()
    normals_vis[:, 1] *= -1.0
    normals_vis[:, 2] *= -1.0
    normals_rgb01 = np.clip(normals_vis * 0.5 + 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    normals_rgb01[:, 0] = 1.0 - normals_rgb01[:, 0]
    normals_rgb01[:, 1] = 1.0 - normals_rgb01[:, 1]

    if (not _NVDIFFRAST_AVAILABLE) or device.type != "cuda":
        return draw_normal_points_fallback(proj_px, normals_rgb01, image_size=image_size, out_h=out_h, out_w=out_w)

    try:
        ctx = _get_raster_context(device)
    except Exception:
        return draw_normal_points_fallback(proj_px, normals_rgb01, image_size=image_size, out_h=out_h, out_w=out_w)

    uv_norm = proj_px / float(max(int(image_size), 1))
    x_clip = np.nan_to_num(uv_norm[:, 0] * 2.0 - 1.0, nan=0.0, posinf=2.0, neginf=-2.0)
    y_clip = np.nan_to_num(1.0 - uv_norm[:, 1] * 2.0, nan=0.0, posinf=2.0, neginf=-2.0)
    depth = cam_xyz[:, 2].astype(np.float32, copy=False)
    if depth.size <= 0 or float(np.max(depth) - np.min(depth)) < 1e-8:
        depth_norm = np.full_like(depth, 0.5, dtype=np.float32)
    else:
        depth_norm = (depth - float(np.min(depth))) / float(np.max(depth) - np.min(depth))
    z_clip = -(depth_norm * 0.8 + 0.1).astype(np.float32, copy=False)

    clip_pos = torch.from_numpy(
        np.stack([x_clip, y_clip, z_clip, np.ones_like(x_clip, dtype=np.float32)], axis=-1)[None, ...]
    ).to(device=device, dtype=torch.float32)
    attrs = torch.from_numpy(normals_rgb01[None, ...]).to(device=device, dtype=torch.float32)
    tri = torch.from_numpy(faces.astype(np.int32, copy=False)).to(device=device, dtype=torch.int32)

    render_h = int(out_h) * 2
    render_w = int(out_w) * 2
    try:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            rast, _ = dr.rasterize(ctx, clip_pos, tri, resolution=[render_h, render_w])
            color, _ = dr.interpolate(attrs, rast, tri)
            color = dr.antialias(color, rast, clip_pos, tri)
            cov = (rast[..., 3:4] > 0).to(dtype=color.dtype)
            color = torch.nan_to_num(color * cov, nan=0.0, posinf=0.0, neginf=0.0)

        color = color.permute(0, 3, 1, 2).contiguous()
        color = F.interpolate(color, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
        color = torch.flip(color, dims=[2])
        normal_rgb = color[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.clip(normal_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    except Exception:
        return draw_normal_points_fallback(proj_px, normals_rgb01, image_size=image_size, out_h=out_h, out_w=out_w)


def _build_clip_positions_from_screen_uv_and_depth(
    screen_uv_norm: np.ndarray,
    depth_values: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(screen_uv_norm, dtype=np.float32).reshape(-1, 2)
    depth = np.asarray(depth_values, dtype=np.float32).reshape(-1)
    clip = np.zeros((uv.shape[0], 4), dtype=np.float32)
    clip[:, 0] = np.nan_to_num(uv[:, 0] * 2.0 - 1.0, nan=0.0, posinf=2.0, neginf=-2.0)
    clip[:, 1] = np.nan_to_num(1.0 - uv[:, 1] * 2.0, nan=0.0, posinf=2.0, neginf=-2.0)
    if depth.size <= 0 or float(np.max(depth) - np.min(depth)) < 1e-8:
        depth_norm = np.full_like(depth, 0.5, dtype=np.float32)
    else:
        depth_norm = (depth - float(np.min(depth))) / float(np.max(depth) - np.min(depth))
    clip[:, 2] = -(depth_norm * 0.8 + 0.1).astype(np.float32, copy=False)
    clip[:, 3] = 1.0
    return clip


def compute_visible_face_mask_from_aligned_view(
    mesh_xyz: np.ndarray,
    screen_uv_norm: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
    rvec: np.ndarray | None = None,
    tvec: np.ndarray | None = None,
) -> np.ndarray:
    tri = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    xyz = np.asarray(mesh_xyz, dtype=np.float32).reshape(-1, 3)
    uv = np.asarray(screen_uv_norm, dtype=np.float32).reshape(-1, 2)
    if tri.shape[0] <= 0 or xyz.shape[0] != uv.shape[0]:
        return np.zeros((tri.shape[0],), dtype=bool)

    cam_xyz = xyz.copy()
    if rvec is not None:
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
        cam_xyz = xyz @ R.T
        if tvec is not None:
            cam_xyz = cam_xyz + np.asarray(tvec, dtype=np.float32).reshape(1, 3)

    v0 = cam_xyz[tri[:, 0]]
    v1 = cam_xyz[tri[:, 1]]
    v2 = cam_xyz[tri[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32, copy=False)
    normal_z = face_normals[:, 2]
    valid_sign_sets = [
        np.isfinite(normal_z) & (normal_z < -1e-8),
        np.isfinite(normal_z) & (normal_z > 1e-8),
    ]

    if (not _NVDIFFRAST_AVAILABLE) or device.type != "cuda":
        count0 = int(valid_sign_sets[0].sum())
        count1 = int(valid_sign_sets[1].sum())
        return valid_sign_sets[0] if count0 >= count1 else valid_sign_sets[1]

    try:
        ctx = _get_raster_context(device)
    except Exception:
        count0 = int(valid_sign_sets[0].sum())
        count1 = int(valid_sign_sets[1].sum())
        return valid_sign_sets[0] if count0 >= count1 else valid_sign_sets[1]

    clip_pos = _build_clip_positions_from_screen_uv_and_depth(uv, cam_xyz[:, 2])
    pos_t = torch.from_numpy(clip_pos[None, ...]).to(device=device, dtype=torch.float32).contiguous()

    best_cover = -1
    best_mask = np.zeros((tri.shape[0],), dtype=bool)
    for candidate_mask in valid_sign_sets:
        candidate_idx = np.flatnonzero(candidate_mask)
        if candidate_idx.size <= 0:
            continue
        tri_t = torch.from_numpy(tri[candidate_idx].astype(np.int32, copy=False)).to(device=device, dtype=torch.int32).contiguous()
        try:
            with torch.amp.autocast(device_type=device.type, enabled=False):
                rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[512, 512])
            tri_ids = rast[0, ..., 3].detach().cpu().numpy()
            visible_local = tri_ids > 0
            cover = int(np.count_nonzero(visible_local))
            if cover > best_cover:
                best_cover = cover
                visible_face_ids = np.unique(tri_ids[visible_local].astype(np.int32) - 1)
                candidate_visible = np.zeros((tri.shape[0],), dtype=bool)
                valid_face_ids = visible_face_ids[(visible_face_ids >= 0) & (visible_face_ids < candidate_idx.size)]
                candidate_visible[candidate_idx[valid_face_ids]] = True
                best_mask = candidate_visible
        except Exception:
            continue

    if int(best_mask.sum()) <= 0:
        count0 = int(valid_sign_sets[0].sum())
        count1 = int(valid_sign_sets[1].sum())
        return valid_sign_sets[0] if count0 >= count1 else valid_sign_sets[1]
    return best_mask


def bake_screen_rgb_to_uv_atlas(
    source_rgb: np.ndarray,
    mesh_uv_norm: np.ndarray,
    template_mesh_uv: np.ndarray,
    template_mesh_faces: np.ndarray,
    out_h: int,
    out_w: int,
    device: torch.device,
    visible_face_mask: np.ndarray | None = None,
) -> np.ndarray:
    atlas_rgb = np.zeros((int(out_h), int(out_w), 3), dtype=np.uint8)
    uv_src = np.asarray(mesh_uv_norm, dtype=np.float32).reshape(-1, 2)
    uv_dst = np.asarray(template_mesh_uv, dtype=np.float32).reshape(-1, 2)
    tri = np.asarray(template_mesh_faces, dtype=np.int32).reshape(-1, 3)
    if uv_src.shape[0] <= 0 or uv_dst.shape[0] != uv_src.shape[0] or tri.shape[0] <= 0:
        return atlas_rgb

    if visible_face_mask is not None:
        face_mask = np.asarray(visible_face_mask, dtype=bool).reshape(-1)
        if face_mask.shape[0] == tri.shape[0]:
            tri = tri[face_mask]
        if tri.shape[0] <= 0:
            return atlas_rgb

    src_uv = np.clip(uv_src, 0.0, 1.0).astype(np.float32, copy=False)
    dst_uv = np.clip(uv_dst, 0.0, 1.0).astype(np.float32, copy=False)

    if (not _NVDIFFRAST_AVAILABLE) or device.type != "cuda":
        img = np.asarray(source_rgb, dtype=np.uint8)
        img_h, img_w = img.shape[:2]
        src_px = np.empty_like(src_uv)
        src_px[:, 0] = src_uv[:, 0] * float(max(img_w - 1, 1))
        src_px[:, 1] = src_uv[:, 1] * float(max(img_h - 1, 1))
        x0 = np.floor(src_px[:, 0]).astype(np.int32)
        y0 = np.floor(src_px[:, 1]).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, max(img_w - 1, 0))
        y1 = np.clip(y0 + 1, 0, max(img_h - 1, 0))
        x0 = np.clip(x0, 0, max(img_w - 1, 0))
        y0 = np.clip(y0, 0, max(img_h - 1, 0))
        tx = (src_px[:, 0] - x0).astype(np.float32)[:, None]
        ty = (src_px[:, 1] - y0).astype(np.float32)[:, None]
        c00 = img[y0, x0].astype(np.float32)
        c10 = img[y0, x1].astype(np.float32)
        c01 = img[y1, x0].astype(np.float32)
        c11 = img[y1, x1].astype(np.float32)
        vertex_rgb = (c00 * (1.0 - tx) * (1.0 - ty) + c10 * tx * (1.0 - ty) + c01 * (1.0 - tx) * ty + c11 * tx * ty)
        if visible_face_mask is not None:
            visible_vertex_mask = np.zeros((dst_uv.shape[0],), dtype=bool)
            visible_vertex_mask[tri.reshape(-1)] = True
        else:
            visible_vertex_mask = np.ones((dst_uv.shape[0],), dtype=bool)
        dst_px = np.empty_like(dst_uv)
        dst_px[:, 0] = dst_uv[:, 0] * float(max(int(out_w) - 1, 1))
        dst_px[:, 1] = (1.0 - dst_uv[:, 1]) * float(max(int(out_h) - 1, 1))
        for idx in range(dst_px.shape[0]):
            if not visible_vertex_mask[idx]:
                continue
            x = int(np.clip(round(float(dst_px[idx, 0])), 0, max(int(out_w) - 1, 0)))
            y = int(np.clip(round(float(dst_px[idx, 1])), 0, max(int(out_h) - 1, 0)))
            atlas_rgb[y, x] = np.clip(vertex_rgb[idx], 0.0, 255.0).astype(np.uint8)
        return atlas_rgb

    try:
        ctx = _get_raster_context(device)
    except Exception:
        return atlas_rgb

    clip_pos = np.zeros((dst_uv.shape[0], 4), dtype=np.float32)
    clip_pos[:, 0] = dst_uv[:, 0] * 2.0 - 1.0
    clip_pos[:, 1] = dst_uv[:, 1] * 2.0 - 1.0
    clip_pos[:, 2] = 0.0
    clip_pos[:, 3] = 1.0

    pos_t = torch.from_numpy(clip_pos[None, ...]).to(device=device, dtype=torch.float32).contiguous()
    tri_t = torch.from_numpy(tri.astype(np.int32, copy=False)).to(device=device, dtype=torch.int32).contiguous()
    src_uv_t = torch.from_numpy(src_uv[None, ...]).to(device=device, dtype=torch.float32).contiguous()
    tex_t = torch.from_numpy(np.asarray(source_rgb, dtype=np.float32) / 255.0).to(device=device, dtype=torch.float32)[None, ...].contiguous()

    render_h = int(out_h) * 2
    render_w = int(out_w) * 2
    try:
        with torch.amp.autocast(device_type=device.type, enabled=False):
            rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[render_h, render_w])
            sample_uv, _ = dr.interpolate(src_uv_t, rast, tri_t)
            sample_uv = torch.stack([sample_uv[..., 0],  sample_uv[..., 1]], dim=-1).clamp(0.0, 1.0)
            color = dr.texture(tex_t, sample_uv, filter_mode="linear", boundary_mode="clamp")
            color = dr.antialias(color, rast, pos_t, tri_t)
            cov = (rast[..., 3:4] > 0).to(dtype=color.dtype)
            color = torch.nan_to_num(color * cov, nan=0.0, posinf=0.0, neginf=0.0)

        color = color.permute(0, 3, 1, 2).contiguous()
        color = F.interpolate(color, size=(int(out_h), int(out_w)), mode="bilinear", align_corners=False)
        #color = torch.flip(color, dims=[2])
        atlas = color[0].permute(1, 2, 0).detach().cpu().numpy()
        return np.flipud(np.clip(atlas * 255.0, 0.0, 255.0).astype(np.uint8))
    except Exception:
        return atlas_rgb


def build_pnp_camera_matrix(image_size: int, focal_scale: float = 1.0) -> np.ndarray:
    f = float(max(int(image_size), 1)) * float(max(focal_scale, 1e-3))
    c = 0.5 * float(max(int(image_size), 1))
    return np.array(
        [
            [f, 0.0, c],
            [0.0, f, c],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _get_raster_context(device: torch.device):
    if not _NVDIFFRAST_AVAILABLE:
        raise RuntimeError("nvdiffrast is not available.")
    if device.type != "cuda":
        raise RuntimeError(f"nvdiffrast requires CUDA, got {device}.")
    key = str(device)
    if key not in _DR_CONTEXT_CACHE:
        _DR_CONTEXT_CACHE[key] = dr.RasterizeCudaContext(device=device)
    return _DR_CONTEXT_CACHE[key]


def identity_screen_affine() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def focal_scale_to_fov_deg(focal_scale: float) -> float:
    scale = float(max(focal_scale, 1e-6))
    return float(np.degrees(2.0 * np.arctan(1.0 / (2.0 * scale))))


def _solve_weighted_regularized_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
    prior: np.ndarray,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    w = np.clip(np.asarray(weights, dtype=np.float32).reshape(-1), 0.0, None)
    prior_arr = np.asarray(prior, dtype=np.float32).reshape(x.shape[1], -1)

    sqrt_w = np.sqrt(w + 1e-8).reshape(-1, 1)
    xw = x * sqrt_w
    yw = y * sqrt_w

    if float(ridge) > 0.0:
        reg = np.sqrt(float(ridge)) * np.eye(x.shape[1], dtype=np.float32)
        xw = np.concatenate([xw, reg], axis=0)
        yw = np.concatenate([yw, reg @ prior_arr], axis=0)

    solution, _, _, _ = np.linalg.lstsq(xw, yw, rcond=None)
    return solution.astype(np.float32, copy=False)


def fit_weighted_screen_scale_translate(
    src_px: np.ndarray,
    dst_px: np.ndarray,
    weights: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    src = np.asarray(src_px, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_px, dtype=np.float32).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & (w > 1e-5)
    if int(valid.sum()) < 2:
        return identity_screen_affine()

    src_v = src[valid]
    dst_v = dst[valid]
    w_v = w[valid]
    coord_scale = float(max(np.max(np.abs(np.concatenate([src_v, dst_v], axis=0))), 1.0))
    src_n = src_v / coord_scale
    dst_n = dst_v / coord_scale

    x_design = np.concatenate([src_n[:, 0:1], np.ones((src_n.shape[0], 1), dtype=np.float32)], axis=1)
    y_design = np.concatenate([src_n[:, 1:2], np.ones((src_n.shape[0], 1), dtype=np.float32)], axis=1)
    x_sol = _solve_weighted_regularized_lstsq(x_design, dst_n[:, 0:1], w_v, ridge, np.array([[1.0], [0.0]], dtype=np.float32))
    y_sol = _solve_weighted_regularized_lstsq(y_design, dst_n[:, 1:2], w_v, ridge, np.array([[1.0], [0.0]], dtype=np.float32))

    sx = float(max(x_sol[0, 0], 1e-3))
    sy = float(max(y_sol[0, 0], 1e-3))
    tx = float(x_sol[1, 0] * coord_scale)
    ty = float(y_sol[1, 0] * coord_scale)
    return np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
        ],
        dtype=np.float32,
    )


def fit_weighted_screen_affine(
    src_px: np.ndarray,
    dst_px: np.ndarray,
    weights: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    src = np.asarray(src_px, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_px, dtype=np.float32).reshape(-1, 2)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & (w > 1e-5)
    if int(valid.sum()) < 6:
        return fit_weighted_screen_scale_translate(src, dst, w, ridge=ridge)

    src_v = src[valid]
    dst_v = dst[valid]
    w_v = w[valid]
    coord_scale = float(max(np.max(np.abs(np.concatenate([src_v, dst_v], axis=0))), 1.0))
    src_n = src_v / coord_scale
    dst_n = dst_v / coord_scale
    design = np.concatenate([src_n, np.ones((src_n.shape[0], 1), dtype=np.float32)], axis=1)
    prior = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    theta = _solve_weighted_regularized_lstsq(design, dst_n, w_v, ridge, prior)
    affine = theta.T.astype(np.float32, copy=False)
    affine[:, 2] *= coord_scale

    linear = affine[:, :2]
    if not np.isfinite(affine).all():
        return fit_weighted_screen_scale_translate(src, dst, w, ridge=ridge)
    try:
        singular_values = np.linalg.svd(linear, compute_uv=False)
        det = float(np.linalg.det(linear))
    except np.linalg.LinAlgError:
        return fit_weighted_screen_scale_translate(src, dst, w, ridge=ridge)

    if det <= 1e-4 or float(singular_values.min()) < 0.20 or float(singular_values.max()) > 5.0:
        return fit_weighted_screen_scale_translate(src, dst, w, ridge=ridge)
    return affine


def apply_screen_affine(points_px: np.ndarray, affine_2x3: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 2)
    if affine_2x3 is None:
        return pts.astype(np.float32, copy=True)
    affine = np.asarray(affine_2x3, dtype=np.float32).reshape(2, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    return (pts_h @ affine.T).astype(np.float32, copy=False)


def invert_screen_affine(affine_2x3: np.ndarray | None) -> np.ndarray:
    if affine_2x3 is None:
        return identity_screen_affine()
    affine = np.asarray(affine_2x3, dtype=np.float32).reshape(2, 3)
    try:
        return cv2.invertAffineTransform(affine).astype(np.float32, copy=False)
    except Exception:
        return identity_screen_affine()


def describe_screen_affine(affine_2x3: np.ndarray | None) -> str:
    if affine_2x3 is None:
        return "warp=identity"
    affine = np.asarray(affine_2x3, dtype=np.float32).reshape(2, 3)
    linear = affine[:, :2]
    try:
        singular_values = np.linalg.svd(linear, compute_uv=False)
        det = float(np.linalg.det(linear))
        s0 = float(singular_values[0])
        s1 = float(singular_values[1])
    except Exception:
        det = float("nan")
        s0 = float("nan")
        s1 = float("nan")
    tx = float(affine[0, 2])
    ty = float(affine[1, 2])
    return f"warp(s1={s0:.3f},s2={s1:.3f},det={det:.3f},tx={tx:.1f},ty={ty:.1f})"


def _select_spread_pnp_indices(
    uv_px: np.ndarray,
    confidence: np.ndarray,
    max_points: int,
    grid_size: int = 24,
) -> np.ndarray:
    conf = np.asarray(confidence, dtype=np.float32)
    pts = np.asarray(uv_px, dtype=np.float32)
    if pts.shape[0] <= int(max_points):
        return np.arange(pts.shape[0], dtype=np.int64)

    grid_size = max(4, int(grid_size))
    cell_best: dict[tuple[int, int], int] = {}
    x_bin = np.clip(np.floor(pts[:, 0] * grid_size / max(float(pts[:, 0].max()) + 1e-6, 1.0)).astype(np.int32), 0, grid_size - 1)
    y_bin = np.clip(np.floor(pts[:, 1] * grid_size / max(float(pts[:, 1].max()) + 1e-6, 1.0)).astype(np.int32), 0, grid_size - 1)
    for idx in range(pts.shape[0]):
        key = (int(x_bin[idx]), int(y_bin[idx]))
        prev = cell_best.get(key, -1)
        if prev < 0 or conf[idx] > conf[prev]:
            cell_best[key] = idx

    chosen = list(cell_best.values())
    if len(chosen) < int(max_points):
        chosen_set = set(chosen)
        remaining = [idx for idx in np.argsort(conf)[::-1] if int(idx) not in chosen_set]
        chosen.extend(remaining[: int(max_points) - len(chosen)])
    elif len(chosen) > int(max_points):
        chosen = list(np.asarray(chosen, dtype=np.int64)[np.argsort(conf[chosen])[::-1][: int(max_points)]])

    return np.asarray(chosen, dtype=np.int64)


def solve_mesh_pnp(
    mesh_xyz: np.ndarray,
    target_uv_norm: np.ndarray,
    confidence: np.ndarray,
    image_size: int,
    focal_scale: float = 1.0,
    max_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    xyz = np.asarray(mesh_xyz, dtype=np.float32)
    uv = np.asarray(target_uv_norm, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1)
    valid &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    valid &= conf > 1e-5
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size < 16:
        return None

    uv_px_all = uv[valid_idx] * float(max(int(image_size), 1))
    conf_all = conf[valid_idx]
    chosen_local = _select_spread_pnp_indices(uv_px_all, conf_all, max_points=max_points)
    chosen_idx = valid_idx[chosen_local]
    obj = xyz[chosen_idx].astype(np.float32, copy=False)
    img = (uv[chosen_idx] * float(max(int(image_size), 1))).astype(np.float32, copy=False)
    obj = np.ascontiguousarray(obj)
    img = np.ascontiguousarray(img)

    camera_matrix = build_pnp_camera_matrix(image_size=image_size, focal_scale=focal_scale)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        iterationsCount=100,
        reprojectionError=4.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return None

    if inliers is not None and len(inliers) >= 6:
        inlier_idx = inliers.reshape(-1)
        try:
            success_refine, rvec, tvec = cv2.solvePnP(
                objectPoints=obj[inlier_idx],
                imagePoints=img[inlier_idx],
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success_refine:
                return None
        except Exception:
            pass
    else:
        inliers = np.arange(obj.shape[0], dtype=np.int32).reshape(-1, 1)

    return rvec.astype(np.float32), tvec.astype(np.float32), camera_matrix.astype(np.float32), inliers.reshape(-1)


def compute_pnp_reprojection_score(
    mesh_xyz: np.ndarray,
    target_uv_norm: np.ndarray,
    confidence: np.ndarray,
    image_size: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    perspective_blend: float = 0.0,
    screen_warp_ridge: float = 1e-3,
) -> float:
    score, _, _ = compute_pnp_reprojection_score_and_warp(
        mesh_xyz,
        target_uv_norm,
        confidence,
        image_size,
        rvec,
        tvec,
        camera_matrix,
        perspective_blend=perspective_blend,
        screen_warp_ridge=screen_warp_ridge,
    )
    return score


def solve_mesh_pnp_with_focal_search(
    mesh_xyz: np.ndarray,
    target_uv_norm: np.ndarray,
    confidence: np.ndarray,
    image_size: int,
    init_focal_scale: float = 3.0,
    perspective_blend: float = 0.0,
    focal_min_scale: float = 0.6,
    focal_max_scale: float = 12.0,
    focal_num_scales: int = 9,
    focal_refine_rounds: int = 2,
    max_points: int = 4000,
    screen_warp_ridge: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray] | None:
    scale_min = float(max(min(focal_min_scale, focal_max_scale), 1e-3))
    scale_max = float(max(focal_min_scale, focal_max_scale, scale_min + 1e-3))
    init_scale = float(np.clip(init_focal_scale, scale_min, scale_max))
    num_scales = int(max(focal_num_scales, 3))
    refine_rounds = int(max(focal_refine_rounds, 0))

    cache: dict[float, tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None, np.ndarray]] = {}

    def _evaluate(scale: float):
        key = float(scale)
        if key in cache:
            return cache[key]
        pnp = solve_mesh_pnp(
            mesh_xyz=mesh_xyz,
            target_uv_norm=target_uv_norm,
            confidence=confidence,
            image_size=image_size,
            focal_scale=key,
            max_points=max_points,
        )
        if pnp is None:
            cache[key] = (float("inf"), None, identity_screen_affine())
            return cache[key]
        score, screen_affine, _ = compute_pnp_reprojection_score_and_warp(
            mesh_xyz=mesh_xyz,
            target_uv_norm=target_uv_norm,
            confidence=confidence,
            image_size=image_size,
            rvec=pnp[0],
            tvec=pnp[1],
            camera_matrix=pnp[2],
            perspective_blend=perspective_blend,
            screen_warp_ridge=screen_warp_ridge,
        )
        cache[key] = (score, pnp, screen_affine)
        return cache[key]

    scales = np.geomspace(scale_min, scale_max, num=num_scales, dtype=np.float64)
    scales = np.unique(np.concatenate([scales, np.asarray([init_scale], dtype=np.float64)]))
    best_scale = init_scale
    best_score = float("inf")
    best_pnp = None
    best_screen_affine = identity_screen_affine()

    for scale in scales:
        score, pnp, screen_affine = _evaluate(float(scale))
        if score < best_score:
            best_score = score
            best_scale = float(scale)
            best_pnp = pnp
            best_screen_affine = screen_affine

    for _ in range(refine_rounds):
        log_best = float(np.log(best_scale))
        log_min = float(np.log(scale_min))
        log_max = float(np.log(scale_max))
        span = max((log_max - log_min) / max(num_scales - 1, 1), 1e-3)
        local_lo = float(np.exp(max(log_best - span, log_min)))
        local_hi = float(np.exp(min(log_best + span, log_max)))
        local_scales = np.geomspace(local_lo, local_hi, num=5, dtype=np.float64)
        for scale in local_scales:
            score, pnp, screen_affine = _evaluate(float(scale))
            if score < best_score:
                best_score = score
                best_scale = float(scale)
                best_pnp = pnp
                best_screen_affine = screen_affine
        span *= 0.5

    if best_pnp is None:
        return None
    return (
        best_pnp[0],
        best_pnp[1],
        best_pnp[2],
        best_pnp[3],
        float(best_scale),
        float(best_score),
        best_screen_affine.astype(np.float32, copy=False),
    )


def project_mesh_with_camera(
    mesh_xyz: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    perspective_blend: float = 0.0,
    screen_affine: np.ndarray | None = None,
) -> np.ndarray:
    xyz = np.ascontiguousarray(np.asarray(mesh_xyz, dtype=np.float32).reshape(-1, 3))
    rvec_f = np.ascontiguousarray(np.asarray(rvec, dtype=np.float32).reshape(3, 1))
    tvec_f = np.ascontiguousarray(np.asarray(tvec, dtype=np.float32).reshape(3, 1))
    K = np.ascontiguousarray(np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3))
    R, _ = cv2.Rodrigues(rvec_f)
    cam = xyz @ R.T + tvec_f.reshape(1, 3)
    z = cam[:, 2:3]
    z_ref = np.mean(z, axis=0, keepdims=True)
    blend = float(np.clip(perspective_blend, 0.0, 1.0))
    z_proj = z_ref * (1.0 - blend) + z * blend
    z_proj = np.where(np.abs(z_proj) < 1e-6, np.sign(z_proj) * 1e-6 + 1e-6, z_proj)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    u = fx * cam[:, 0:1] / z_proj + cx
    v = fy * cam[:, 1:2] / z_proj + cy
    proj = np.concatenate([u, v], axis=1).astype(np.float32, copy=False)
    if screen_affine is not None:
        proj = apply_screen_affine(proj, screen_affine)
    return proj


def compute_pnp_reprojection_score_and_warp(
    mesh_xyz: np.ndarray,
    target_uv_norm: np.ndarray,
    confidence: np.ndarray,
    image_size: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    perspective_blend: float = 0.0,
    screen_warp_ridge: float = 1e-3,
) -> tuple[float, np.ndarray, np.ndarray]:
    xyz = np.asarray(mesh_xyz, dtype=np.float32)
    uv = np.asarray(target_uv_norm, dtype=np.float32)
    conf = np.asarray(confidence, dtype=np.float32)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(uv).all(axis=1) & (conf > 1e-5)
    if int(valid.sum()) < 16:
        identity = identity_screen_affine()
        return float("inf"), identity, np.zeros((xyz.shape[0], 2), dtype=np.float32)

    proj_px = project_mesh_with_camera(
        xyz,
        rvec,
        tvec,
        camera_matrix,
        perspective_blend=perspective_blend,
    )
    target_px = uv * float(max(int(image_size), 1))
    screen_affine = fit_weighted_screen_affine(
        proj_px[valid],
        target_px[valid],
        conf[valid],
        ridge=float(max(screen_warp_ridge, 0.0)),
    )
    warped_proj_px = apply_screen_affine(proj_px, screen_affine)
    err = np.linalg.norm(warped_proj_px[valid] - target_px[valid], axis=1)
    w = conf[valid]
    score = float((err * w).sum() / (w.sum() + 1e-6))
    return score, screen_affine, warped_proj_px


def refine_mesh_xyz_with_projection(
    model: Dense2Geometry,
    mesh_xyz: torch.Tensor,
    target_uv_norm: torch.Tensor,
    match_mask: torch.Tensor,
    search_distance: torch.Tensor,
    image_size: int,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    confidence_power: float = 1.5,
    anchor_gain: float = 1.0,
    diffuse_iters: int = 6,
    diffuse_blend: float = 0.70,
    anchor_hold: float = 0.85,
    perspective_blend: float = 0.0,
    screen_affine: np.ndarray | None = None,
) -> torch.Tensor:
    xyz = mesh_xyz.float()
    device = xyz.device
    dtype = xyz.dtype

    match = match_mask.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    distance = torch.nan_to_num(
        search_distance.to(device=device, dtype=dtype),
        nan=float(model.search_distance_threshold),
        posinf=float(model.search_distance_threshold),
        neginf=0.0,
    )
    tau = float(max(model.search_distance_threshold, model.search_distance_floor, 1e-6))
    confidence = match * torch.exp(-distance / tau)
    if float(confidence_power) != 1.0:
        confidence = confidence.pow(float(max(confidence_power, 0.0)))
    if not bool((confidence > 0).any().item()):
        return xyz

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
    R_t = torch.from_numpy(R.astype(np.float32)).to(device=device, dtype=dtype)
    t_row = torch.from_numpy(np.asarray(tvec, dtype=np.float32).reshape(1, 3)).to(device=device, dtype=dtype)
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    screen_affine_inv = torch.from_numpy(invert_screen_affine(screen_affine)).to(device=device, dtype=dtype)

    cam_xyz = xyz @ R_t.transpose(0, 1) + t_row
    target_px = target_uv_norm.to(device=device, dtype=dtype).clamp(0.0, 1.0) * float(max(int(image_size), 1))
    target_px_h = torch.cat([target_px, torch.ones_like(target_px[:, :1])], dim=-1)
    target_px = target_px_h @ screen_affine_inv.transpose(0, 1)
    z = cam_xyz[:, 2:3]
    z_ref = z.mean(dim=0, keepdim=True)
    blend = float(np.clip(perspective_blend, 0.0, 1.0))
    z_proj = z_ref * (1.0 - blend) + z * blend
    x_new = (target_px[:, 0:1] - cx) * z_proj / max(fx, 1e-6)
    y_new = (target_px[:, 1:2] - cy) * z_proj / max(fy, 1e-6)
    cam_anchor = torch.cat([x_new, y_new, z], dim=-1)
    world_anchor = (cam_anchor - t_row) @ R_t

    anchor_delta = (world_anchor - xyz) * confidence.unsqueeze(-1) * float(max(anchor_gain, 0.0))
    anchor_delta = torch.where(match.unsqueeze(-1) > 0.5, anchor_delta, torch.zeros_like(anchor_delta))

    edge_src = model.mesh_edge_src.to(device=device)
    edge_dst = model.mesh_edge_dst.to(device=device)
    edge_degree = model.mesh_edge_degree.to(device=device, dtype=dtype)

    disp = anchor_delta
    support = confidence
    matched = match > 0.5
    hold = float(np.clip(anchor_hold, 0.0, 1.0))
    blend = float(np.clip(diffuse_blend, 0.0, 1.0))
    for _ in range(int(max(diffuse_iters, 0))):
        if edge_src.numel() <= 0:
            break
        neighbor_disp = torch.zeros_like(disp)
        neighbor_disp.index_add_(0, edge_dst, disp[edge_src])
        neighbor_disp = neighbor_disp / edge_degree.view(-1, 1).clamp_min(1.0)

        neighbor_support = torch.zeros_like(support)
        neighbor_support.index_add_(0, edge_dst, support[edge_src])
        neighbor_support = neighbor_support / edge_degree.clamp_min(1.0)

        mixed_disp = (1.0 - blend) * disp + blend * neighbor_disp
        disp = torch.where(
            matched.unsqueeze(-1),
            hold * anchor_delta + (1.0 - hold) * mixed_disp,
            mixed_disp * neighbor_support.unsqueeze(-1).clamp(0.0, 1.0),
        )
        support = torch.where(matched, confidence, neighbor_support.clamp(0.0, 1.0))

    return xyz + disp


def tensor_chw_to_masked_rgb(tensor_chw: torch.Tensor, mask_hw: torch.Tensor | None = None) -> np.ndarray:
    img = tensor_chw.detach().cpu().float().permute(1, 2, 0).numpy()
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    if mask_hw is not None:
        mask = mask_hw.detach().cpu().float().numpy()
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.clip(mask, 0.0, 1.0)
        img = img * mask[..., None]
    return (img * 255.0).astype(np.uint8)


def prepare_overlay_points(
    uv_norm: np.ndarray,
    vis_size: int,
    restore_indices: np.ndarray | None = None,
    invalid_mask: np.ndarray | None = None,
) -> np.ndarray:
    pts = np.asarray(uv_norm, dtype=np.float32).copy()
    pts = np.clip(pts, 0.0, 1.0)
    if restore_indices is not None:
        pts = pts[np.asarray(restore_indices, dtype=np.int64)]
        if invalid_mask is not None:
            invalid_mask = np.asarray(invalid_mask, dtype=bool)[np.asarray(restore_indices, dtype=np.int64)]
    elif invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
    pts[:, 0] *= float(vis_size)
    pts[:, 1] *= float(vis_size)
    if invalid_mask is not None and invalid_mask.shape[0] == pts.shape[0]:
        pts[invalid_mask] = np.nan
    return pts


def load_restore_indices(model_dir: str) -> np.ndarray | None:
    restore_path = os.path.join(model_dir, "mesh_inverse.npy")
    if not os.path.exists(restore_path):
        return None
    return np.load(restore_path).astype(np.int64, copy=False)


def collect_images(image_path: str) -> list[str]:
    path = Path(image_path)
    if path.is_file():
        return [str(path)]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {image_path}")
    files = []
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTS:
            files.append(str(file_path))
    return files


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_legacy_dense2geometry_module(commit: str = "1659b8f") -> ModuleType:
    result = subprocess.run(
        ["git", "show", f"{commit}:dense2geometry.py"],
        cwd=os.getcwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    module = ModuleType(f"dense2geometry_legacy_{commit}")
    module.__file__ = f"<git:{commit}:dense2geometry.py>"
    exec(result.stdout, module.__dict__)
    return module


def legacy_search_single_sample(
    self,
    pred_geo: torch.Tensor,
    pred_mask_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    search_geo = torch.nn.functional.interpolate(
        pred_geo.unsqueeze(0),
        size=(self.search_size, self.search_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    search_mask = torch.nn.functional.interpolate(
        pred_mask_logits.unsqueeze(0),
        size=(self.search_size, self.search_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    geo_flat = search_geo.permute(1, 2, 0).reshape(-1, 3)
    mask_prob = torch.sigmoid(search_mask[0]).reshape(-1)
    geo_mag = torch.linalg.norm(geo_flat, dim=1)
    valid = geo_mag > self.search_min_geo_magnitude
    valid = valid & (mask_prob > self.search_mask_threshold)
    if int(valid.sum().item()) < self.min_search_candidates:
        valid = geo_mag > self.search_min_geo_magnitude
    if int(valid.sum().item()) < self.min_search_candidates:
        topk = min(self.min_search_candidates, int(geo_mag.numel()))
        _, top_idx = torch.topk(geo_mag, k=max(topk, 1), largest=True, sorted=False)
        valid = torch.zeros_like(geo_mag, dtype=torch.bool)
        valid[top_idx] = True

    ys, xs = torch.meshgrid(
        torch.arange(self.search_size, device=pred_geo.device, dtype=torch.float32),
        torch.arange(self.search_size, device=pred_geo.device, dtype=torch.float32),
        indexing="ij",
    )
    uv_grid = torch.stack(
        [
            (xs + 0.5) / float(self.search_size),
            (ys + 0.5) / float(self.search_size),
        ],
        dim=-1,
    ).view(-1, 2)

    candidate_codes = geo_flat[valid]
    candidate_uv = uv_grid[valid]
    if candidate_codes.numel() == 0:
        return (
            torch.full((self.num_mesh, 2), -1.0, device=pred_geo.device, dtype=torch.float32),
            torch.zeros((self.num_mesh,), device=pred_geo.device, dtype=torch.float32),
            torch.full((self.num_mesh,), self.search_distance_threshold, device=pred_geo.device, dtype=torch.float32),
        )

    nearest_idx, distances = self._nearest_feature_search(self.mesh_geo_codes.to(pred_geo.device), candidate_codes)
    threshold = self._compute_adaptive_threshold(distances)
    accept = torch.isfinite(distances) & (distances <= threshold)
    searched_uv = candidate_uv[nearest_idx].to(dtype=torch.float32)
    searched_uv[~accept] = -1.0
    return searched_uv, accept.float(), distances


def legacy_search_mesh_positions(
    self,
    pred_geo: torch.Tensor,
    pred_mask_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    searched = []
    matched = []
    distances = []
    with torch.no_grad():
        for b in range(pred_geo.shape[0]):
            uv_b, match_b, dist_b = legacy_search_single_sample(self, pred_geo[b], pred_mask_logits[b])
            searched.append(uv_b)
            matched.append(match_b)
            distances.append(dist_b)
    return (
        torch.stack(searched, dim=0),
        torch.stack(matched, dim=0),
        torch.stack(distances, dim=0),
    )


def legacy_forward_compat(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    f4, f8, f16, dense_parts = self._run_dense_stage(rgb)
    searched_uv, match_mask, search_dist = legacy_search_mesh_positions(
        self,
        pred_geo=dense_parts["geo"],
        pred_mask_logits=dense_parts["mask_logits"],
    )

    sampled_f4 = self._sample_feature_map(f4, searched_uv, match_mask)
    sampled_f8 = self._sample_feature_map(f8, searched_uv, match_mask)
    sampled_f16 = self._sample_feature_map(f16, searched_uv, match_mask)
    sampled_feat = torch.cat([sampled_f4, sampled_f8, sampled_f16], dim=-1)

    semantic_token = self.vertex_semantic_proj(self.mesh_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1))
    coord_input = torch.cat([searched_uv.clamp_min(0.0), match_mask.unsqueeze(-1), search_dist.unsqueeze(-1)], dim=-1)
    vertex_tokens = (
        self.vertex_image_proj(sampled_feat)
        + semantic_token
        + self.vertex_coord_proj(coord_input)
        + self.vertex_embed.weight.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    )

    image_memory = self._build_image_memory(f4, f8, f16)
    landmark_tokens, landmark_uv, landmark_valid = self._pool_mesh_to_landmarks(
        vertex_tokens,
        searched_uv,
        match_mask,
    )
    landmark_tokens = landmark_tokens + self.landmark_embed.weight.unsqueeze(0)
    landmark_tokens = landmark_tokens + self.landmark_semantic_proj(
        self.landmark_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    )
    landmark_tokens = landmark_tokens + self.landmark_coord_proj(
        torch.cat([landmark_uv.clamp_min(0.0), landmark_valid.unsqueeze(-1)], dim=-1)
    )
    for block in self.landmark_blocks:
        landmark_tokens = block(landmark_tokens, image_memory)
    landmark_tokens = self.landmark_norm(landmark_tokens)

    mesh_context = self._broadcast_landmarks_to_mesh(landmark_tokens)
    mesh_tokens = vertex_tokens + self.mesh_context_proj(mesh_context)
    for block in self.mesh_refine_blocks[:2]:
        mesh_tokens = block(mesh_tokens)

    offsets = self.output_head(mesh_tokens)
    mesh_coords = self.template_mesh.unsqueeze(0).expand(rgb.shape[0], -1, -1).clone()
    mesh_coords[..., :3] = mesh_coords[..., :3] + offsets[..., :3]
    uv_base = torch.where(
        match_mask.unsqueeze(-1) > 0.5,
        searched_uv,
        mesh_coords[..., 3:5],
    )
    mesh_coords[..., 3:5] = (uv_base + offsets[..., 3:5]).clamp_(0.0, 1.0)
    mesh_coords[..., 5:] = mesh_coords[..., 5:] + offsets[..., 5:]
    return {
        "mesh": mesh_coords,
        "searched_uv": searched_uv,
        "match_mask": match_mask,
        "search_distance": search_dist,
        "pred_geo": dense_parts["geo"],
        "pred_normal": dense_parts["normal"],
        "pred_basecolor": dense_parts["basecolor"],
        "pred_mask_logits": dense_parts["mask_logits"],
    }


def pre_concat_local_search_forward_compat(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
    f4, f8, f16, dense_parts = self._run_dense_stage(rgb)
    if dense_parts["basecolor"] is None or dense_parts["geo"] is None or dense_parts["normal"] is None:
        raise ValueError(
            "Dense2Geometry inference requires dense-stage predictions for basecolor, geo, normal, and mask."
        )

    pred_basecolor = dense_parts["basecolor"]
    pred_geo = dense_parts["geo"]
    pred_normal = dense_parts["normal"]
    pred_mask_logits = dense_parts["mask_logits"]
    searched_uv, match_mask, search_dist = self._search_mesh_positions(
        pred_geo=pred_geo,
        pred_mask_logits=pred_mask_logits,
    )

    sampled_f4 = self._sample_feature_map(f4, searched_uv, match_mask)
    sampled_f8 = self._sample_feature_map(f8, searched_uv, match_mask)
    sampled_f16 = self._sample_feature_map(f16, searched_uv, match_mask)
    sampled_feat = torch.cat([sampled_f4, sampled_f8, sampled_f16], dim=-1)
    sampled_basecolor = self._sample_feature_map(pred_basecolor, searched_uv, match_mask)
    sampled_geo = self._sample_feature_map(pred_geo, searched_uv, match_mask)
    sampled_normal = self._sample_feature_map(pred_normal, searched_uv, match_mask)
    sampled_mask = self._sample_feature_map(torch.sigmoid(pred_mask_logits), searched_uv, match_mask)
    sampled_dense = torch.cat([sampled_geo, sampled_normal, sampled_basecolor, sampled_mask], dim=-1)

    semantic_token = self.vertex_semantic_proj(self.mesh_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1))
    coord_input = torch.cat([searched_uv.clamp_min(0.0), match_mask.unsqueeze(-1), search_dist.unsqueeze(-1)], dim=-1)
    vertex_tokens = (
        self.vertex_image_proj(sampled_feat)
        + self.vertex_dense_proj(sampled_dense)
        + semantic_token
        + self.vertex_coord_proj(coord_input)
        + self.vertex_embed.weight.unsqueeze(0).expand(rgb.shape[0], -1, -1)
    )

    image_memory = self._build_image_memory(f4, f8, f16)
    landmark_tokens, landmark_uv, landmark_valid = self._pool_mesh_to_landmarks(
        vertex_tokens,
        searched_uv,
        match_mask,
    )
    landmark_tokens = (
        landmark_tokens
        + self.landmark_semantic_proj(self.landmark_geo_codes.unsqueeze(0).expand(rgb.shape[0], -1, -1))
        + self.landmark_coord_proj(
            torch.cat([landmark_uv.clamp_min(0.0), landmark_valid.unsqueeze(-1)], dim=-1)
        )
        + self.landmark_embed.weight.unsqueeze(0)
    )
    for block in self.landmark_blocks:
        landmark_tokens = block(landmark_tokens, image_memory)
    landmark_tokens = self.landmark_norm(landmark_tokens)

    mesh_context = self._broadcast_landmarks_to_mesh(landmark_tokens)
    mesh_tokens = vertex_tokens + self.mesh_context_proj(mesh_context)
    for block in self.mesh_graph_blocks:
        mesh_tokens = block(
            mesh_tokens,
            self.mesh_edge_src,
            self.mesh_edge_dst,
            self.mesh_edge_degree,
        )
    for block in self.mesh_refine_blocks:
        mesh_tokens = block(mesh_tokens)

    offsets = self.output_head(mesh_tokens)
    mesh_coords = self.template_mesh.unsqueeze(0) + offsets
    return {
        "mesh": mesh_coords,
        "searched_uv": searched_uv,
        "match_mask": match_mask,
        "search_distance": search_dist,
        "pred_geo": pred_geo,
        "pred_normal": pred_normal,
        "pred_basecolor": pred_basecolor,
        "pred_mask_logits": pred_mask_logits,
    }


def build_model(args, device: torch.device) -> Dense2Geometry:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    use_pre_concat_graph_model = (
        "vertex_concat_fuse.1.weight" not in state_dict
        and "mesh_graph_blocks.0.self_norm.weight" in state_dict
        and "vertex_dense_proj.0.weight" in state_dict
    )
    use_older_legacy_compat = (
        "vertex_concat_fuse.1.weight" not in state_dict
        and "mesh_graph_blocks.0.self_norm.weight" not in state_dict
        and "vertex_image_proj.weight" in state_dict
    )
    dense_stage_cfg = DenseStageConfig(
        d_model=int(args.dense_d_model),
        nhead=int(args.dense_nhead),
        num_layers=int(args.dense_num_layers),
        output_size=int(args.dense_output_size),
        transformer_map_size=int(args.dense_transformer_map_size),
        backbone_weights=str(args.dense_backbone_weights),
        decoder_type=str(args.dense_decoder_type),
    )
    model = Dense2Geometry(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        dense_stage_cfg=dense_stage_cfg,
        dense_checkpoint="",
        freeze_dense_stage=True,
        image_memory_size=int(args.image_memory_size),
        search_size=int(args.search_size),
        landmark_search_size=int(args.landmark_search_size),
        local_search_radius=int(args.local_search_radius),
        local_min_candidates=int(args.local_min_candidates),
        search_chunk_size=int(args.search_chunk_size),
        search_distance_threshold=float(args.search_distance_threshold),
        search_distance_floor=float(args.search_distance_floor),
        search_mad_scale=float(args.search_mad_scale),
        search_mask_threshold=float(args.search_mask_threshold),
        search_min_geo_magnitude=float(args.search_min_geo_magnitude),
        min_search_candidates=int(args.min_search_candidates),
        model_dir=str(args.model_dir),
    ).to(device)
    skipped_keys, load_notes = _load_matching_state_dict(model, state_dict)
    ignored_static_buffers = {
        "landmark2keypoint_knn_indices",
        "landmark2keypoint_knn_weights",
    }
    load_notes = [name for name in load_notes if name not in ignored_static_buffers]
    if use_pre_concat_graph_model:
        model.forward = MethodType(pre_concat_local_search_forward_compat, model)
    if use_older_legacy_compat:
        if int(args.search_size) == 512:
            model.search_size = 128
        model.forward = MethodType(legacy_forward_compat, model)
    model.eval()

    print(f"[Info] Loaded checkpoint: {args.checkpoint}")
    if use_pre_concat_graph_model:
        print(
            "[Info] Enabled pre-concat local-search compatibility mode "
            f"(search_size={model.search_size}, landmark_search_size={model.landmark_search_size}, "
            f"dense_output_size={model.dense_stage.output_size})."
        )
        if int(args.image_size) != 1024:
            print(
                "[Warn] This checkpoint family was trained with 1024x1024 aligned inputs. "
                f"Using image_size={int(args.image_size)} can make the 2D mesh appear too small."
            )
    if use_older_legacy_compat:
        print(f"[Info] Enabled legacy checkpoint compatibility mode (search_size={model.search_size}).")
    if skipped_keys:
        print(f"[Warn] Skipped incompatible keys: {skipped_keys[:10]}")
    if load_notes:
        print(f"[Warn] Missing/unexpected keys: {load_notes[:10]}")
    return model


def run_inference(args) -> None:
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    image_files = collect_images(args.image_path)
    if not image_files:
        raise FileNotFoundError(f"No supported images found in: {args.image_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    align_helper = Align5PtHelper(
        image_size=int(args.image_size),
    )

    mesh_topology = load_mesh_topology()
    mesh_faces_full = build_full_mesh_faces(mesh_topology)
    _, uv_atlas_template_uv, uv_atlas_template_faces = _load_filtered_template_mesh(model_dir=str(args.model_dir))
    mesh_restore_indices = load_restore_indices(str(args.model_dir))
    model = build_model(args, device)
    if int(args.post_refine_iters) > 0:
        print(
            "[Info] Enabled inference UV snap "
            f"(iters={int(args.post_refine_iters)}, step={float(args.post_refine_step_size):.2f}, "
            f"max_step_px={float(args.post_refine_max_step_px):.2f}, "
            f"kp_gain={float(args.post_refine_keypoint_gain):.2f}, "
            f"kp_max_step_px={float(args.post_refine_keypoint_max_step_px):.2f}, "
            f"lm_refine={float(args.post_refine_landmark_refine_blend):.2f}, "
            f"kp_only={bool(args.post_refine_keypoint_only)}, "
            f"smooth={float(args.post_refine_smooth_blend):.2f}, "
            f"direct_gain={float(args.post_refine_direct_gain):.2f}, "
            f"direct_max_step_px={float(args.post_refine_direct_max_step_px):.2f}, "
            f"diffuse_iters={int(args.post_refine_diffuse_iters)})."
        )
    if bool(args.pnp_optimize_focal):
        focal_msg = (
            f"optimize[{float(args.pnp_focal_min_scale):.2f},{float(args.pnp_focal_max_scale):.2f}]"
            f"/fov[{focal_scale_to_fov_deg(float(args.pnp_focal_max_scale)):.1f},{focal_scale_to_fov_deg(float(args.pnp_focal_min_scale)):.1f}]deg"
        )
    else:
        focal_msg = (
            f"fixed={float(args.pnp_focal_scale):.2f}"
            f"/fov={focal_scale_to_fov_deg(float(args.pnp_focal_scale)):.1f}deg"
        )
    print(
        "[Info] Enabled PnP 3D align "
        f"(focal_scale={focal_msg}, max_points={int(args.pnp_max_points)}, "
        f"perspective_blend={float(args.pnp_perspective_blend):.2f}, "
        f"screen_warp_ridge={float(args.pnp_screen_warp_ridge):.4f}, "
        f"anchor_gain={float(args.post_refine_3d_anchor_gain):.2f}, "
        f"diffuse_iters={int(args.post_refine_3d_diffuse_iters)})."
    )

    vis_size = int(args.vis_size)
    amp_enabled = bool(args.amp) and device.type == "cuda"

    for img_path in tqdm(image_files, desc="Dense2Geometry"):
        basename = Path(img_path).stem
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[Warn] Failed to read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        align_rgb, _, pre_align_affine = maybe_resize_for_alignment(img_rgb, int(args.image_size))
        lm6, detection_source = align_helper.detect_landmarks(align_rgb, fallback_lm_px=None)
        m = align_helper.estimate_alignment_matrix(
            lm6,
            src_w=align_rgb.shape[1],
            src_h=align_rgb.shape[0],
            split="val",
        )

        pre_align_inv = cv2.invertAffineTransform(pre_align_affine)
        key5_px, key5_valid = align_helper.extract_key5_from_lm68(lm6)
        if bool(key5_valid.any()):
            five_pts_orig = Align5PtHelper.transform_points_px(key5_px, pre_align_inv)
        else:
            five_pts_orig = np.full((5, 2), np.nan, dtype=np.float32)
            if detection_source == "none":
                print(f"[Warn] Alignment landmarks not found for {img_path}; using fallback centering.")
        five_pt_rgb = draw_5pt(img_rgb, five_pts_orig)
        img_aligned = cv2.warpAffine(
            align_rgb,
            m,
            (int(args.image_size), int(args.image_size)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        rgb_tensor = torch.from_numpy(preprocess_aligned_rgb(img_aligned, int(args.image_size))).unsqueeze(0).to(device)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                outputs = model(rgb_tensor)
                pre_refine_mesh = outputs["mesh"].clone()
                if int(args.post_refine_iters) > 0:
                    outputs = dict(outputs)
                    outputs["mesh"] = model.refine_mesh_uv_with_search(
                        mesh_coords=outputs["mesh"],
                        searched_uv=outputs["searched_uv"],
                        match_mask=outputs["match_mask"],
                        search_distance=outputs["search_distance"],
                        image_size=int(args.image_size),
                        num_iters=int(args.post_refine_iters),
                        step_size=float(args.post_refine_step_size),
                        max_step_px=float(args.post_refine_max_step_px),
                        smooth_blend=float(args.post_refine_smooth_blend),
                        confidence_power=float(args.post_refine_confidence_power),
                        keypoint_gain=float(args.post_refine_keypoint_gain),
                        keypoint_max_step_px=float(args.post_refine_keypoint_max_step_px),
                        landmark_refine_blend=float(args.post_refine_landmark_refine_blend),
                        keypoint_only=bool(args.post_refine_keypoint_only),
                        direct_snap_gain=float(args.post_refine_direct_gain),
                        direct_snap_max_step_px=float(args.post_refine_direct_max_step_px),
                        direct_diffuse_iters=int(args.post_refine_diffuse_iters),
                        direct_diffuse_blend=float(args.post_refine_direct_diffuse_blend),
                    )

        search_distance_np = outputs["search_distance"][0].detach().cpu().float().numpy()
        pred_mask = torch.sigmoid(outputs["pred_mask_logits"][0, 0]).detach().cpu()
        pred_basecolor = tensor_chw_to_masked_rgb(outputs["pred_basecolor"][0], pred_mask)
        pred_geo = tensor_chw_to_masked_rgb(outputs["pred_geo"][0], pred_mask)
        pred_normal = tensor_chw_to_masked_rgb(outputs["pred_normal"][0], pred_mask)

        pre_refine_mesh_np = pre_refine_mesh[0].detach().cpu().float().numpy()
        mesh_pred = outputs["mesh"][0].detach().cpu().float().numpy()
        match_mask = outputs["match_mask"][0].detach().cpu().float().numpy()
        matched_point_overlay = prepare_overlay_points(
            outputs["searched_uv"][0].detach().cpu().float().numpy(),
            vis_size=vis_size,
            restore_indices=mesh_restore_indices,
            invalid_mask=match_mask <= 0.5,
        )
        pre_refine_overlay_points = prepare_overlay_points(
            pre_refine_mesh_np[:, 3:5],
            vis_size=vis_size,
            restore_indices=mesh_restore_indices,
        )
        aligned_overlay_points = prepare_overlay_points(
            mesh_pred[:, 3:5],
            vis_size=vis_size,
            restore_indices=mesh_restore_indices,
        )

        tau = float(max(getattr(model, "search_distance_threshold", 0.05), getattr(model, "search_distance_floor", 0.02), 1e-6))
        pnp_confidence = np.clip(match_mask, 0.0, 1.0) * np.exp(-np.clip(search_distance_np, 0.0, None) / tau)
        pnp_confidence = np.power(pnp_confidence, float(max(args.post_refine_confidence_power, 0.0)))
        if bool(args.pnp_optimize_focal):
            pnp_result = solve_mesh_pnp_with_focal_search(
                mesh_xyz=mesh_pred[:, :3],
                target_uv_norm=mesh_pred[:, 3:5],
                confidence=pnp_confidence,
                image_size=int(args.image_size),
                init_focal_scale=float(args.pnp_focal_scale),
                perspective_blend=float(args.pnp_perspective_blend),
                focal_min_scale=float(args.pnp_focal_min_scale),
                focal_max_scale=float(args.pnp_focal_max_scale),
                focal_num_scales=int(args.pnp_focal_num_scales),
                focal_refine_rounds=int(args.pnp_focal_refine_rounds),
                max_points=int(args.pnp_max_points),
                screen_warp_ridge=float(args.pnp_screen_warp_ridge),
            )
        else:
            pnp_fixed = solve_mesh_pnp(
                mesh_xyz=mesh_pred[:, :3],
                target_uv_norm=mesh_pred[:, 3:5],
                confidence=pnp_confidence,
                image_size=int(args.image_size),
                focal_scale=float(args.pnp_focal_scale),
                max_points=int(args.pnp_max_points),
            )
            if pnp_fixed is None:
                pnp_result = None
            else:
                fixed_score, fixed_screen_affine, _ = compute_pnp_reprojection_score_and_warp(
                    mesh_xyz=mesh_pred[:, :3],
                    target_uv_norm=mesh_pred[:, 3:5],
                    confidence=pnp_confidence,
                    image_size=int(args.image_size),
                    rvec=pnp_fixed[0],
                    tvec=pnp_fixed[1],
                    camera_matrix=pnp_fixed[2],
                    perspective_blend=float(args.pnp_perspective_blend),
                    screen_warp_ridge=float(args.pnp_screen_warp_ridge),
                )
                pnp_result = (
                    pnp_fixed[0],
                    pnp_fixed[1],
                    pnp_fixed[2],
                    pnp_fixed[3],
                    float(args.pnp_focal_scale),
                    float(fixed_score),
                    fixed_screen_affine.astype(np.float32, copy=False),
                )

        pred_3d_overlay_rgb = cv2.resize(img_aligned, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        refined_3d_overlay_rgb = cv2.resize(img_aligned, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        refined_3d_normal_rgb = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
        refined_xyz_np = mesh_pred[:, :3].copy()
        if pnp_result is not None:
            rvec, tvec, camera_matrix, inliers, focal_scale_used, pnp_score, screen_affine = pnp_result
            pred_proj_raw_px = project_mesh_with_camera(
                mesh_pred[:, :3],
                rvec,
                tvec,
                camera_matrix,
                perspective_blend=float(args.pnp_perspective_blend),
            )
            pred_proj_px = apply_screen_affine(pred_proj_raw_px, screen_affine)
            refined_xyz = refine_mesh_xyz_with_projection(
                model=model,
                mesh_xyz=outputs["mesh"][0, :, :3],
                target_uv_norm=outputs["mesh"][0, :, 3:5],
                match_mask=outputs["match_mask"][0],
                search_distance=outputs["search_distance"][0],
                image_size=int(args.image_size),
                rvec=rvec,
                tvec=tvec,
                camera_matrix=camera_matrix,
                confidence_power=float(args.post_refine_confidence_power),
                anchor_gain=float(args.post_refine_3d_anchor_gain),
                diffuse_iters=int(args.post_refine_3d_diffuse_iters),
                diffuse_blend=float(args.post_refine_3d_diffuse_blend),
                anchor_hold=float(args.post_refine_3d_anchor_hold),
                perspective_blend=float(args.pnp_perspective_blend),
                screen_affine=screen_affine,
            )
            refined_xyz_np = refined_xyz.detach().cpu().float().numpy()
            refined_proj_raw_px = project_mesh_with_camera(
                refined_xyz_np,
                rvec,
                tvec,
                camera_matrix,
                perspective_blend=float(args.pnp_perspective_blend),
            )
            refined_proj_px = apply_screen_affine(refined_proj_raw_px, screen_affine)

            target_px = mesh_pred[:, 3:5] * float(max(int(args.image_size), 1))
            valid_reproj = (match_mask > 0.5) & np.isfinite(target_px).all(axis=1)
            if np.any(valid_reproj):
                raw_pre_reproj = np.linalg.norm(pred_proj_raw_px[valid_reproj] - target_px[valid_reproj], axis=1)
                pre_reproj = np.linalg.norm(pred_proj_px[valid_reproj] - target_px[valid_reproj], axis=1)
                post_reproj = np.linalg.norm(refined_proj_px[valid_reproj] - target_px[valid_reproj], axis=1)
                print(
                    f"[PnP3D] {basename}: inliers={int(len(inliers))} "
                    f"focal={float(focal_scale_used):.2f}/fov={focal_scale_to_fov_deg(float(focal_scale_used)):.1f}deg "
                    f"score={float(pnp_score):.2f} {describe_screen_affine(screen_affine)} "
                    f"raw(mean={float(raw_pre_reproj.mean()):.2f}px,p95={float(np.percentile(raw_pre_reproj, 95)):.2f}px) "
                    f"warp(mean={float(pre_reproj.mean()):.2f}px,p95={float(np.percentile(pre_reproj, 95)):.2f}px) "
                    f"post(mean={float(post_reproj.mean()):.2f}px,p95={float(np.percentile(post_reproj, 95)):.2f}px)"
                )
            pred_proj_pts = pred_proj_px.copy()
            refined_proj_pts = refined_proj_px.copy()
            if mesh_restore_indices is not None:
                pred_proj_pts = pred_proj_pts[np.asarray(mesh_restore_indices, dtype=np.int64)]
                refined_proj_pts = refined_proj_pts[np.asarray(mesh_restore_indices, dtype=np.int64)]
            pred_3d_overlay_rgb = create_combined_overlay(img_aligned, pred_proj_pts, mesh_topology)
            refined_3d_overlay_rgb = create_combined_overlay(img_aligned, refined_proj_pts, mesh_topology)
            pred_3d_overlay_rgb = cv2.resize(pred_3d_overlay_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            refined_3d_overlay_rgb = cv2.resize(refined_3d_overlay_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            refined_xyz_full = restore_full_vertices(refined_xyz_np, mesh_restore_indices).astype(np.float32, copy=False)
            refined_proj_full = restore_full_vertices(refined_proj_px, mesh_restore_indices).astype(np.float32, copy=False)
            refined_3d_normal_rgb = render_mesh_screen_normals(
                mesh_xyz_full=refined_xyz_full,
                projected_px_full=refined_proj_full,
                faces_full=mesh_faces_full,
                image_size=int(args.image_size),
                out_h=vis_size,
                out_w=vis_size,
                device=device,
                rvec=rvec,
                tvec=tvec,
            )
        else:
            print(f"[PnP3D] {basename}: solvePnP failed")
            pred_3d_overlay_rgb = cv2.resize(img_aligned, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            refined_3d_overlay_rgb = cv2.resize(img_aligned, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            refined_3d_normal_rgb = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)

        m_total = compose_affine(m, pre_align_affine)
        m_inv = cv2.invertAffineTransform(m_total)
        mesh_pred_orig = transform_geometry_2d(
            mesh_pred,
            m_inv,
            aligned_size=int(args.image_size),
            original_w=w_orig,
            original_h=h_orig,
        )

        src_vis = cv2.resize(img_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        five_pt_vis = cv2.resize(five_pt_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        aligned_rgb = cv2.resize(img_aligned, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        aligned_matched_points_rgb = draw_vertex_points(aligned_rgb, matched_point_overlay)
        pre_refine_overlay_rgb = create_combined_overlay(aligned_rgb, pre_refine_overlay_points, mesh_topology)
        aligned_overlay_rgb = create_combined_overlay(aligned_rgb, aligned_overlay_points, mesh_topology)
        aligned_uv_atlas_rgb = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
        basecolor_uv_atlas_rgb = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
        normal_uv_atlas_rgb = np.zeros((vis_size, vis_size, 3), dtype=np.uint8)
        if (
            uv_atlas_template_uv is not None
            and uv_atlas_template_faces is not None
            and uv_atlas_template_uv.shape[0] == mesh_pred.shape[0]
        ):
            atlas_rvec = rvec if pnp_result is not None else None
            atlas_tvec = tvec if pnp_result is not None else None
            atlas_visible_face_mask = compute_visible_face_mask_from_aligned_view(
                mesh_xyz=refined_xyz_np,
                screen_uv_norm=np.clip(mesh_pred[:, 3:5], 0.0, 1.0).astype(np.float32, copy=False),
                device=device,
                rvec=atlas_rvec,
                tvec=atlas_tvec,
                faces=uv_atlas_template_faces,
            )
            aligned_uv_atlas_rgb = bake_screen_rgb_to_uv_atlas(
                source_rgb=img_aligned,
                mesh_uv_norm=np.clip(mesh_pred[:, 3:5], 0.0, 1.0).astype(np.float32, copy=False),
                template_mesh_uv=uv_atlas_template_uv.astype(np.float32, copy=False),
                template_mesh_faces=uv_atlas_template_faces,
                out_h=vis_size,
                out_w=vis_size,
                device=device,
                visible_face_mask=atlas_visible_face_mask,
            )
            basecolor_uv_atlas_rgb = bake_screen_rgb_to_uv_atlas(
                source_rgb=pred_basecolor,
                mesh_uv_norm=np.clip(mesh_pred[:, 3:5], 0.0, 1.0).astype(np.float32, copy=False),
                template_mesh_uv=uv_atlas_template_uv.astype(np.float32, copy=False),
                template_mesh_faces=uv_atlas_template_faces,
                out_h=vis_size,
                out_w=vis_size,
                device=device,
                visible_face_mask=atlas_visible_face_mask,
            )
            normal_uv_atlas_rgb = bake_screen_rgb_to_uv_atlas(
                source_rgb=pred_normal,
                mesh_uv_norm=np.clip(mesh_pred[:, 3:5], 0.0, 1.0).astype(np.float32, copy=False),
                template_mesh_uv=uv_atlas_template_uv.astype(np.float32, copy=False),
                template_mesh_faces=uv_atlas_template_faces,
                out_h=vis_size,
                out_w=vis_size,
                device=device,
                visible_face_mask=atlas_visible_face_mask,
            )
        delta_px = np.linalg.norm(aligned_overlay_points - pre_refine_overlay_points, axis=1)
        valid_delta = np.isfinite(delta_px)
        matched_delta = valid_delta & ((match_mask[mesh_restore_indices] if mesh_restore_indices is not None else match_mask) > 0.5)
        if np.any(valid_delta):
            mean_all = float(delta_px[valid_delta].mean())
            p95_all = float(np.percentile(delta_px[valid_delta], 95))
            max_all = float(delta_px[valid_delta].max())
        else:
            mean_all = p95_all = max_all = 0.0
        if np.any(matched_delta):
            mean_matched = float(delta_px[matched_delta].mean())
            p95_matched = float(np.percentile(delta_px[matched_delta], 95))
            max_matched = float(delta_px[matched_delta].max())
        else:
            mean_matched = p95_matched = max_matched = 0.0
        print(
            f"[RefineDiff] {basename}: "
            f"all(mean={mean_all:.2f}px,p95={p95_all:.2f}px,max={max_all:.2f}px) "
            f"matched(mean={mean_matched:.2f}px,p95={p95_matched:.2f}px,max={max_matched:.2f}px)"
        )
        theoretical_max_px = (
            float(max(args.post_refine_max_step_px, 0.0))
            * float(max(args.post_refine_step_size, 0.0))
            * float(max(int(args.post_refine_iters), 1))
        )
        refine_diff_rgb = draw_refine_displacement_overlay(
            aligned_rgb,
            pre_refine_overlay_points,
            aligned_overlay_points,
            max_color_px=max(theoretical_max_px, max_all, 1.0),
        )

        base_vis = cv2.resize(pred_basecolor, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        geo_vis = cv2.resize(pred_geo, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        normal_vis = cv2.resize(pred_normal, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        src_panel = add_panel_title(src_vis, "Src Img")
        five_pt_panel = add_panel_title(five_pt_vis, "Pred 5 Pt on Src Img")
        matched_points_panel = add_panel_title(aligned_matched_points_rgb, "Matched 2D Vertices")
        pre_refine_overlay_panel = add_panel_title(pre_refine_overlay_rgb, "Aligned Pre-Refine Mesh")
        overlay_title = "Aligned Refined Mesh Overlay" if int(args.post_refine_iters) > 0 else "Aligned Output Mesh Overlay"
        aligned_overlay_panel = add_panel_title(aligned_overlay_rgb, overlay_title)
        aligned_uv_atlas_panel = add_panel_title(aligned_uv_atlas_rgb, "Aligned Img -> UV Atlas")
        basecolor_uv_atlas_panel = add_panel_title(basecolor_uv_atlas_rgb, "Pred Basecolor -> UV Atlas")
        normal_uv_atlas_panel = add_panel_title(normal_uv_atlas_rgb, "Pred Normal -> UV Atlas")
        pred_3d_overlay_panel = add_panel_title(pred_3d_overlay_rgb, "PnP+Affine Pred 3D Mesh")
        refined_3d_overlay_panel = add_panel_title(refined_3d_overlay_rgb, "PnP+Affine Refined 3D Mesh")
        refined_3d_normal_panel = add_panel_title(refined_3d_normal_rgb, "Aligned Refined 3D Normal")
        diff_title = f"Refine Diff Heat (max {max_all:.2f}px)"
        refine_diff_panel = add_panel_title(refine_diff_rgb, diff_title)
        base_panel = add_panel_title(base_vis, "Pred Basecolor")
        geo_panel = add_panel_title(geo_vis, "Pred Geo")
        normal_panel = add_panel_title(normal_vis, "Pred Normal")
        combined = compose_panel_grid(
            [
                src_panel,
                five_pt_panel,
                matched_points_panel,
                pre_refine_overlay_panel,
                aligned_overlay_panel,
                aligned_uv_atlas_panel,
                basecolor_uv_atlas_panel,
                normal_uv_atlas_panel,
                pred_3d_overlay_panel,
                refined_3d_overlay_panel,
                refined_3d_normal_panel,
                refine_diff_panel,
                base_panel,
                geo_panel,
                normal_panel,
            ],
            cols=4,
        )

        cv2.imwrite(
            os.path.join(args.output_dir, f"{basename}_combined.png"),
            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR),
        )
        save_mesh_obj(
            os.path.join(args.output_dir, f"{basename}_mesh.obj"),
            mesh_pred_orig[:, :3],
            mesh_topology,
            restore_indices=mesh_restore_indices,
        )
        save_mesh_obj(
            os.path.join(args.output_dir, f"{basename}_mesh_3d_refined.obj"),
            refined_xyz_np,
            mesh_topology,
            restore_indices=mesh_restore_indices,
        )

    print(f"[Done] Saved outputs to {args.output_dir}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense2Geometry inference on images with combined visualization export.")
    parser.add_argument("--image_path", type=str, default="samples", help="Input image file or folder.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/best_dense2geometry.pth", help="Dense2Geometry checkpoint path.")
    parser.add_argument("--output_dir", type=str, default="artifacts/test_result_dense2geometry", help="Directory for saved outputs.")
    parser.add_argument("--model_dir", type=str, default="assets/topology", help="Directory containing mesh templates/topology.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable fp16 autocast on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable fp16 autocast.")

    parser.add_argument("--image_size", type=int, default=1024, help="Aligned input image size.")
    parser.add_argument("--vis_size", type=int, default=1024, help="Per-panel output size.")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--image_memory_size", type=int, default=16)
    parser.add_argument("--search_size", type=int, default=512)
    parser.add_argument("--landmark_search_size", type=int, default=256)
    parser.add_argument("--local_search_radius", type=int, default=12)
    parser.add_argument("--local_min_candidates", type=int, default=9)
    parser.add_argument("--search_chunk_size", type=int, default=1024)
    parser.add_argument("--search_distance_threshold", type=float, default=0.05)
    parser.add_argument("--search_distance_floor", type=float, default=0.02)
    parser.add_argument("--search_mad_scale", type=float, default=3.0)
    parser.add_argument("--search_mask_threshold", type=float, default=0.30)
    parser.add_argument("--search_min_geo_magnitude", type=float, default=0.02)
    parser.add_argument("--min_search_candidates", type=int, default=512)

    parser.add_argument("--dense_d_model", type=int, default=256)
    parser.add_argument("--dense_nhead", type=int, default=8)
    parser.add_argument("--dense_num_layers", type=int, default=4)
    parser.add_argument("--dense_output_size", type=int, default=1024)
    parser.add_argument("--dense_transformer_map_size", type=int, default=32)
    parser.add_argument("--dense_backbone_weights", type=str, default="imagenet", choices=["imagenet", "dinov3"])
    parser.add_argument("--dense_decoder_type", type=str, default="multitask", choices=["multitask", "shared"])
    parser.add_argument("--post_refine_iters", type=int, default=3, help="Inference-only landmark-space UV snap iterations. Set 0 to disable.")
    parser.add_argument("--post_refine_step_size", type=float, default=0.9, help="Per-iteration UV snap step size.")
    parser.add_argument("--post_refine_max_step_px", type=float, default=1.5, help="Maximum landmark step per iteration in aligned-image pixels.")
    parser.add_argument("--post_refine_keypoint_gain", type=float, default=1.0, help="Coarse keypoint-stage gain before landmark refinement.")
    parser.add_argument("--post_refine_keypoint_max_step_px", type=float, default=2.0, help="Maximum keypoint step per iteration in aligned-image pixels.")
    parser.add_argument("--post_refine_landmark_refine_blend", type=float, default=0.60, help="How much landmark residual correction is applied after keypoint-to-landmark propagation.")
    parser.add_argument("--post_refine_keypoint_only", action="store_true", help="Use only keypoint-driven propagation for 2D refine and disable landmark residual refine/direct snap.")
    parser.add_argument("--post_refine_smooth_blend", type=float, default=0.20, help="Blend between raw and neighbor-smoothed UV updates.")
    parser.add_argument("--post_refine_confidence_power", type=float, default=1.5, help="Sharpness of search-distance confidence weighting.")
    parser.add_argument("--post_refine_direct_gain", type=float, default=1.0, help="Direct matched-vertex snap gain after landmark refinement.")
    parser.add_argument("--post_refine_direct_max_step_px", type=float, default=4.0, help="Maximum direct snap step in aligned-image pixels.")
    parser.add_argument("--post_refine_diffuse_iters", type=int, default=3, help="Topology diffusion passes after direct matched-vertex snap.")
    parser.add_argument("--post_refine_direct_diffuse_blend", type=float, default=0.70, help="Blend toward neighbor-propagated displacement during direct snap diffusion.")
    parser.add_argument("--pnp_focal_scale", type=float, default=3.0, help="Focal length scale used for solvePnP camera intrinsics.")
    parser.add_argument("--pnp_optimize_focal", action="store_true", default=True, help="Search focal scale and keep the lowest reprojection-error PnP solution.")
    parser.add_argument("--no-pnp-optimize-focal", dest="pnp_optimize_focal", action="store_false", help="Disable focal-scale search and use the fixed focal scale.")
    parser.add_argument("--pnp_focal_min_scale", type=float, default=0.6, help="Minimum focal scale considered during PnP focal search.")
    parser.add_argument("--pnp_focal_max_scale", type=float, default=12.0, help="Maximum focal scale considered during PnP focal search.")
    parser.add_argument("--pnp_focal_num_scales", type=int, default=9, help="Number of coarse focal-scale samples for PnP focal search.")
    parser.add_argument("--pnp_focal_refine_rounds", type=int, default=2, help="Local focal-scale refinement rounds after the coarse PnP search.")
    parser.add_argument("--pnp_max_points", type=int, default=4000, help="Maximum spread correspondences used for solvePnP.")
    parser.add_argument("--pnp_perspective_blend", type=float, default=0.0, help="0=weak perspective from fitted pose, 1=full perspective.")
    parser.add_argument("--pnp_screen_warp_ridge", type=float, default=0.01, help="Regularization toward identity for the screen-space affine correction fitted after PnP.")
    parser.add_argument("--post_refine_3d_anchor_gain", type=float, default=1.0, help="3D anchor gain from refined 2D mesh under fitted PnP camera.")
    parser.add_argument("--post_refine_3d_diffuse_iters", type=int, default=6, help="3D mesh diffusion iterations after projection anchors.")
    parser.add_argument("--post_refine_3d_diffuse_blend", type=float, default=0.70, help="3D diffusion blend toward neighbor offsets.")
    parser.add_argument("--post_refine_3d_anchor_hold", type=float, default=0.85, help="How strongly matched vertices stay attached to their 3D anchor offsets.")
    return parser


if __name__ == "__main__":
    run_inference(create_parser().parse_args())
