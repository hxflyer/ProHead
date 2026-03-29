from __future__ import annotations

import os
import threading
import uuid
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data_utils.obj_io import load_uv_obj_file
from train_visualize_helper import load_combined_mesh_uv

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    Image = None
    _PIL_AVAILABLE = False


def is_png_intact(path: str) -> bool:
    if not path or not os.path.exists(path):
        return False
    if not str(path).lower().endswith(".png") or not _PIL_AVAILABLE:
        return os.path.getsize(path) > 0
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def _atomic_write_bytes_path(path: str) -> str:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temp_name = f".{os.path.basename(path)}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    return os.path.join(directory, temp_name)


try:
    import nvdiffrast.torch as dr

    _NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    _NVDIFFRAST_AVAILABLE = False


def load_float01_png(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    if str(path).lower().endswith(".png") and not is_png_intact(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def save_float01_png(path: str, image: np.ndarray, bit_depth: int = 8) -> None:
    arr = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        out = np.rint(arr * (65535.0 if bit_depth == 16 else 255.0)).astype(np.uint16 if bit_depth == 16 else np.uint8)
    else:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
            out = np.rint(arr * (65535.0 if bit_depth == 16 else 255.0)).astype(np.uint16 if bit_depth == 16 else np.uint8)
        else:
            out = np.rint(arr * (65535.0 if bit_depth == 16 else 255.0)).astype(np.uint16 if bit_depth == 16 else np.uint8)
            out = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2BGR)
    temp_path = _atomic_write_bytes_path(path)
    try:
        ok = cv2.imwrite(temp_path, out)
        if not ok:
            raise RuntimeError(f"Failed to write PNG: {temp_path}")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def save_mask_png(path: str, mask: np.ndarray) -> None:
    arr = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    out = np.rint(arr * 255.0).astype(np.uint8)
    temp_path = _atomic_write_bytes_path(path)
    try:
        ok = cv2.imwrite(temp_path, out)
        if not ok:
            raise RuntimeError(f"Failed to write mask PNG: {temp_path}")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def load_mask_png(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    if str(path).lower().endswith(".png") and not is_png_intact(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3:
        image = image[:, :, 0]
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32) / 255.0


def _load_combined_mesh_triangle_faces(model_dir: str = "assets/topology") -> np.ndarray:
    part_files = ["mesh_head.obj", "mesh_eye_l.obj", "mesh_eye_r.obj", "mesh_mouth.obj"]
    tris = []
    offset = 0
    for file_name in part_files:
        obj_path = os.path.join(model_dir, file_name)
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(obj_path, triangulate=True)
        if verts is None or uvs is None or v_faces is None:
            raise ValueError(f"Failed to load verts/uv/faces from {obj_path}")
        tris.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))
    return np.concatenate(tris, axis=0).astype(np.int32, copy=False)


def _remap_triangle_faces_after_vertex_filter(
    triangle_faces: np.ndarray,
    kept_vertex_indices: np.ndarray,
    original_vertex_count: int,
    vertex_positions: np.ndarray | None = None,
) -> np.ndarray:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)

    all_indices = np.arange(int(original_vertex_count), dtype=np.int64)
    filtered = all_indices[remap < 0]
    if filtered.size > 0:
        if vertex_positions is not None and vertex_positions.shape[0] == int(original_vertex_count):
            pos = np.asarray(vertex_positions, dtype=np.float32)
            kept_pos = pos[kept]
            filt_pos = pos[filtered]
            diff = filt_pos[:, None, :] - kept_pos[None, :, :]
            nn_idx = (diff * diff).sum(axis=-1).argmin(axis=-1)
        else:
            nn_idx = np.searchsorted(kept, filtered).clip(0, kept.shape[0] - 1)
        remap[filtered] = nn_idx

    return remap[tri].astype(np.int32, copy=False)


def load_filtered_mesh_uv_and_faces(model_dir: str = "assets/topology") -> tuple[np.ndarray, np.ndarray]:
    template_mesh = np.load(os.path.join(model_dir, "mesh_template.npy")).astype(np.float32)
    template_mesh_uv = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32, copy=False)
    template_mesh_faces = _load_combined_mesh_triangle_faces(model_dir=model_dir)
    template_mesh_full = template_mesh.copy()
    mesh_indices_path = os.path.join(model_dir, "mesh_indices.npy")
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path).astype(np.int64, copy=False)
        if mesh_indices.max() < template_mesh_uv.shape[0]:
            template_mesh_uv = template_mesh_uv[mesh_indices]
        template_mesh_faces = _remap_triangle_faces_after_vertex_filter(
            template_mesh_faces,
            mesh_indices,
            original_vertex_count=int(template_mesh_full.shape[0]),
            vertex_positions=template_mesh_full[:, :3],
        )
    return template_mesh_uv.astype(np.float32, copy=False), template_mesh_faces.astype(np.int32, copy=False)


def sample_feature_map_at_vertices(feature_map: torch.Tensor, vertex_uv: torch.Tensor) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}")
    if vertex_uv.ndim != 3 or vertex_uv.shape[-1] != 2:
        raise ValueError(f"vertex_uv must be [B,N,2], got {tuple(vertex_uv.shape)}")
    grid = vertex_uv.to(device=feature_map.device, dtype=feature_map.dtype).clone()
    grid = grid.mul(2.0).sub(1.0)
    grid = grid.view(feature_map.shape[0], vertex_uv.shape[1], 1, 2)
    samples = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return samples.squeeze(-1).permute(0, 2, 1).contiguous()


def rasterize_vertex_features_to_uv_atlas(
    vertex_features: torch.Tensor,
    template_uv: torch.Tensor,
    template_faces: torch.Tensor,
    atlas_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _NVDIFFRAST_AVAILABLE:
        raise RuntimeError("nvdiffrast is required for UV atlas rasterization.")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("UV atlas rasterization currently requires a CUDA device.")
    if vertex_features.ndim != 3:
        raise ValueError(f"vertex_features must be [B,N,C], got {tuple(vertex_features.shape)}")

    batch_size, num_vertices, _ = vertex_features.shape
    uv = template_uv.to(device=dev, dtype=torch.float32)
    if uv.ndim != 2 or uv.shape[0] != num_vertices or uv.shape[1] != 2:
        raise ValueError(
            f"template_uv must be [N,2] matching vertex_features, got {tuple(uv.shape)} vs N={num_vertices}"
        )
    tri = template_faces.to(device=dev, dtype=torch.int32).contiguous()

    clip_pos = torch.zeros((num_vertices, 4), device=dev, dtype=torch.float32)
    clip_pos[:, 0] = uv[:, 0] * 2.0 - 1.0
    clip_pos[:, 1] = 1.0 - (uv[:, 1] * 2.0)
    clip_pos[:, 3] = 1.0
    pos_t = clip_pos.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    feat_t = vertex_features.to(device=dev, dtype=torch.float32).contiguous()

    ctx = dr.RasterizeCudaContext(device=dev)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        rast, _ = dr.rasterize(ctx, pos_t, tri, resolution=[int(atlas_size), int(atlas_size)])
        atlas, _ = dr.interpolate(feat_t, rast, tri)
        coverage = (rast[..., 3:4] > 0).to(dtype=atlas.dtype)
        atlas = atlas * coverage

    return atlas.permute(0, 3, 1, 2).contiguous(), coverage.permute(0, 3, 1, 2).contiguous()


def build_partial_uv_atlas_inputs(
    *,
    rgb: torch.Tensor,
    pred_basecolor: torch.Tensor,
    pred_geo: torch.Tensor,
    pred_geometry_normal: torch.Tensor,
    pred_detail_normal: torch.Tensor,
    mesh_uv: torch.Tensor,
    match_mask: torch.Tensor,
    template_uv: torch.Tensor,
    template_faces: torch.Tensor,
    atlas_size: int,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    detail_normal_01 = pred_detail_normal.mul(0.5).add(0.5).clamp(0.0, 1.0)
    sampled_maps = torch.cat(
        [rgb, pred_basecolor, pred_geo, pred_geometry_normal, detail_normal_01],
        dim=1,
    )
    vertex_features = sample_feature_map_at_vertices(sampled_maps, mesh_uv)
    match_vertices = match_mask.to(device=rgb.device, dtype=rgb.dtype).unsqueeze(-1)
    packed_vertex_features = torch.cat([vertex_features, match_vertices], dim=-1)

    atlas, coverage = rasterize_vertex_features_to_uv_atlas(
        packed_vertex_features,
        template_uv,
        template_faces,
        atlas_size,
        device,
    )
    atlas = atlas * coverage

    src_color_uv = atlas[:, 0:3]
    pred_basecolor_uv = atlas[:, 3:6]
    pred_geo_uv = atlas[:, 6:9]
    pred_geometry_normal_uv = atlas[:, 9:12]
    pred_detail_normal_uv = atlas[:, 12:15]
    uv_valid_mask = atlas[:, 15:16].clamp(0.0, 1.0)

    return {
        "src_color_uv": src_color_uv,
        "pred_basecolor_uv": pred_basecolor_uv,
        "pred_geo_uv": pred_geo_uv,
        "pred_geometry_normal_uv": pred_geometry_normal_uv,
        "pred_detail_normal_uv": pred_detail_normal_uv,
        "uv_valid_mask": uv_valid_mask,
        "uv_coverage": coverage,
    }
