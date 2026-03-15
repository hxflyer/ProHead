from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SAMPLING_CACHE_VERSION = 3
SAMPLE_LAYOUT_NAME = "discrete_lloyd_relaxed"


def _import_installed_gsplat_module(module_name: str):
    hidden_paths: list[tuple[int, str]] = []
    shadow_roots = {SCRIPT_DIR.resolve(), REPO_ROOT.resolve()}

    for idx in range(len(sys.path) - 1, -1, -1):
        raw = sys.path[idx]
        resolved = Path(raw or os.getcwd()).resolve()
        if resolved in shadow_roots:
            hidden_paths.append((idx, sys.path.pop(idx)))

    try:
        return importlib.import_module(module_name)
    finally:
        for idx, value in sorted(hidden_paths, key=lambda item: item[0]):
            sys.path.insert(idx, value)


_gsplat_torch_impl_2dgs = _import_installed_gsplat_module("gsplat.cuda._torch_impl_2dgs")
_fully_fused_projection_2dgs = _gsplat_torch_impl_2dgs._fully_fused_projection_2dgs
try:
    _gsplat_wrapper = _import_installed_gsplat_module("gsplat.cuda._wrapper")
except Exception:
    _gsplat_wrapper = None
_gsplat_wrapper_runtime_available = _gsplat_wrapper is not None and (os.name != "nt" or shutil.which("cl") is not None)


def array_sha1(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha1(contiguous.view(np.uint8)).hexdigest()


def compute_face_areas_numpy(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tris = vertices[faces]
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32)


@lru_cache(maxsize=None)
def subdivision_barycentrics(level: int) -> np.ndarray:
    if level < 1:
        raise ValueError("Subdivision level must be >= 1")
    if level == 1:
        return np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)

    tris: list[np.ndarray] = []
    inv = 1.0 / float(level)

    def bary(i: int, j: int) -> np.ndarray:
        return np.array([1.0 - (i + j) * inv, i * inv, j * inv], dtype=np.float32)

    for i in range(level):
        for j in range(level - i):
            a = bary(i, j)
            b = bary(i + 1, j)
            c = bary(i, j + 1)
            tris.append(np.stack([a, b, c], axis=0))
            if i + j < level - 1:
                d = bary(i + 1, j + 1)
                tris.append(np.stack([b, d, c], axis=0))

    out = np.stack(tris, axis=0).astype(np.float32)
    expected = level * level
    if out.shape[0] != expected:
        raise AssertionError(f"Expected {expected} sub-triangles for level {level}, got {out.shape[0]}")
    return out


@lru_cache(maxsize=None)
def spread_triangle_barycentrics(count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be >= 1")
    if count == 1:
        return np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32)

    candidate_level = max(6, int(math.ceil(math.sqrt(float(count)) * 4.0)))
    candidates = subdivision_barycentrics(candidate_level).mean(axis=1).astype(np.float32)
    if count >= candidates.shape[0]:
        return candidates

    tri_height = math.sqrt(3.0) * 0.5

    def bary_to_xy(bary: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                bary[:, 1] + 0.5 * bary[:, 2],
                bary[:, 2] * tri_height,
            ],
            axis=1,
        ).astype(np.float32)

    candidate_xy = bary_to_xy(candidates)
    centroid = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    selected_indices = [int(np.argmin(np.sum((candidates - centroid[None, :]) ** 2, axis=1)))]
    min_sq_dists = np.sum((candidates - candidates[selected_indices[0]][None, :]) ** 2, axis=1)
    available = np.ones(candidates.shape[0], dtype=bool)
    available[selected_indices[0]] = False

    while len(selected_indices) < count:
        candidate_scores = np.where(available, min_sq_dists, -1.0)
        next_idx = int(np.argmax(candidate_scores))
        selected_indices.append(next_idx)
        available[next_idx] = False
        sq_dists = np.sum((candidates - candidates[next_idx][None, :]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)

    centers = candidates[np.asarray(selected_indices, dtype=np.int32)].astype(np.float32)
    centers_xy = bary_to_xy(centers)

    for _ in range(6):
        diff = candidate_xy[:, None, :] - centers_xy[None, :, :]
        assignments = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        updated = centers.copy()
        for center_idx in range(count):
            mask = assignments == center_idx
            if np.any(mask):
                updated[center_idx] = candidates[mask].mean(axis=0)
        updated = np.clip(updated, 1e-4, 1.0)
        updated /= updated.sum(axis=1, keepdims=True)
        centers = updated.astype(np.float32)
        centers_xy = bary_to_xy(centers)

    return centers.astype(np.float32)


def compute_face_sample_counts(
    face_areas: np.ndarray,
    target_area_percentile: float,
    max_subdivision_level: int,
) -> tuple[np.ndarray, float, int]:
    target_area = float(np.percentile(face_areas, target_area_percentile))
    max_samples_per_face = max(1, int(max_subdivision_level) * int(max_subdivision_level))
    sample_counts = np.ceil(np.maximum(face_areas.astype(np.float64) / max(target_area, 1e-12), 1.0)).astype(np.int32)
    sample_counts = np.clip(sample_counts, 1, max_samples_per_face).astype(np.int32)
    return sample_counts, target_area, max_samples_per_face


def build_sampling_cache_metadata(
    reference_vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    uv_faces: np.ndarray,
    target_area_percentile: float,
    max_subdivision_level: int,
) -> dict[str, str | int | float]:
    return {
        "version": int(SAMPLING_CACHE_VERSION),
        "vertex_count": int(reference_vertices.shape[0]),
        "uv_count": int(uvs.shape[0]),
        "face_count": int(faces.shape[0]),
        "uv_face_count": int(uv_faces.shape[0]),
        "reference_vertices_sha1": array_sha1(reference_vertices.astype(np.float32, copy=False)),
        "faces_sha1": array_sha1(faces.astype(np.int32, copy=False)),
        "uvs_sha1": array_sha1(uvs.astype(np.float32, copy=False)),
        "uv_faces_sha1": array_sha1(uv_faces.astype(np.int32, copy=False)),
        "target_area_percentile": float(target_area_percentile),
        "max_subdivision_level": int(max_subdivision_level),
        "sample_layout": SAMPLE_LAYOUT_NAME,
    }


def build_sampling_plan(
    reference_vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    uv_faces: np.ndarray,
    target_area_percentile: float,
    max_subdivision_level: int,
) -> dict[str, np.ndarray | float | int | dict[str, str | int | float]]:
    face_areas = compute_face_areas_numpy(reference_vertices, faces)
    face_sample_counts, target_area, max_samples_per_face = compute_face_sample_counts(
        face_areas=face_areas,
        target_area_percentile=target_area_percentile,
        max_subdivision_level=max_subdivision_level,
    )

    face_indices_parts: list[np.ndarray] = []
    bary_parts: list[np.ndarray] = []
    for face_idx, count in enumerate(face_sample_counts.tolist()):
        bary = spread_triangle_barycentrics(int(count))
        bary_parts.append(bary)
        face_indices_parts.append(np.full((bary.shape[0],), face_idx, dtype=np.int32))

    sample_face_indices = np.concatenate(face_indices_parts, axis=0).astype(np.int32, copy=False)
    sample_barys = np.concatenate(bary_parts, axis=0).astype(np.float32, copy=False)
    sample_uv_faces = uv_faces[sample_face_indices]
    sample_uvs = (
        uvs[sample_uv_faces[:, 0]] * sample_barys[:, 0:1]
        + uvs[sample_uv_faces[:, 1]] * sample_barys[:, 1:2]
        + uvs[sample_uv_faces[:, 2]] * sample_barys[:, 2:3]
    ).astype(np.float32, copy=False)

    return {
        "metadata": build_sampling_cache_metadata(
            reference_vertices=reference_vertices,
            faces=faces,
            uvs=uvs,
            uv_faces=uv_faces,
            target_area_percentile=target_area_percentile,
            max_subdivision_level=max_subdivision_level,
        ),
        "face_sample_counts": face_sample_counts.astype(np.int32, copy=False),
        "sample_face_indices": sample_face_indices,
        "sample_barys": sample_barys,
        "sample_uvs": np.clip(sample_uvs, 0.0, 1.0).astype(np.float32, copy=False),
        "target_area": float(target_area),
        "max_samples_per_face": int(max_samples_per_face),
    }


def save_sampling_plan(path: Path, plan: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(plan["metadata"], sort_keys=True)),
        face_sample_counts=np.asarray(plan["face_sample_counts"], dtype=np.int32),
        sample_face_indices=np.asarray(plan["sample_face_indices"], dtype=np.int32),
        sample_barys=np.asarray(plan["sample_barys"], dtype=np.float32),
        sample_uvs=np.asarray(plan["sample_uvs"], dtype=np.float32),
        target_area=np.asarray([float(plan["target_area"])], dtype=np.float32),
        max_samples_per_face=np.asarray([int(plan["max_samples_per_face"])], dtype=np.int32),
    )


def load_sampling_plan(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].tolist()))
        return {
            "metadata": metadata,
            "face_sample_counts": data["face_sample_counts"].astype(np.int32, copy=False),
            "sample_face_indices": data["sample_face_indices"].astype(np.int32, copy=False),
            "sample_barys": data["sample_barys"].astype(np.float32, copy=False),
            "sample_uvs": data["sample_uvs"].astype(np.float32, copy=False),
            "target_area": float(np.asarray(data["target_area"]).reshape(-1)[0]),
            "max_samples_per_face": int(np.asarray(data["max_samples_per_face"]).reshape(-1)[0]),
        }


def sampling_plan_matches_metadata(
    cached_plan: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> bool:
    cached_metadata = cached_plan.get("metadata", {})
    return dict(cached_metadata) == dict(expected_metadata)


def resolve_sampling_plan(
    reference_vertices: np.ndarray,
    uvs: np.ndarray,
    faces: np.ndarray,
    uv_faces: np.ndarray,
    target_area_percentile: float,
    max_subdivision_level: int,
    sampling_cache_path: str | Path | None = None,
    rebuild_sampling_cache: bool = False,
) -> tuple[dict[str, Any], str]:
    expected_metadata = build_sampling_cache_metadata(
        reference_vertices=reference_vertices,
        faces=faces,
        uvs=uvs,
        uv_faces=uv_faces,
        target_area_percentile=target_area_percentile,
        max_subdivision_level=max_subdivision_level,
    )

    cache_path = Path(sampling_cache_path).resolve() if sampling_cache_path else None
    if cache_path is not None and cache_path.exists() and (not rebuild_sampling_cache):
        cached_plan = load_sampling_plan(cache_path)
        if sampling_plan_matches_metadata(cached_plan, expected_metadata):
            return cached_plan, "loaded"

    plan = build_sampling_plan(
        reference_vertices=reference_vertices,
        uvs=uvs,
        faces=faces,
        uv_faces=uv_faces,
        target_area_percentile=target_area_percentile,
        max_subdivision_level=max_subdivision_level,
    )
    if cache_path is not None:
        save_sampling_plan(cache_path, plan)
    return plan, "rebuilt" if cache_path is not None else "memory_only"


def default_sampling_cache_path(
    cache_root: str | Path,
    cache_stem: str,
    vertex_count: int,
    face_count: int,
    target_area_percentile: float,
    max_subdivision_level: int,
) -> Path:
    cache_root_path = Path(cache_root)
    cache_name = (
        f"{cache_stem}_v{int(vertex_count)}_f{int(face_count)}"
        f"_p{int(round(float(target_area_percentile) * 10.0))}"
        f"_s{int(max_subdivision_level)}.npz"
    )
    return (cache_root_path / cache_name).resolve()


def _ensure_numpy_float32(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(data, dtype=np.float32)


def _ensure_numpy_int32(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(data):
        return data.detach().cpu().numpy().astype(np.int32, copy=False)
    return np.asarray(data, dtype=np.int32)


def matrix_data_to_torch(
    matrix_data: Mapping[str, Any],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    fov = matrix_data.get("fov", 50.0)
    head_matrix = matrix_data.get("head_matrix", np.eye(4, dtype=np.float32))
    camera_location = matrix_data.get("camera_location", np.zeros(3, dtype=np.float32))
    camera_rotation = matrix_data.get("camera_rotation", np.zeros(3, dtype=np.float32))
    resolution = matrix_data.get("resolution", None)

    out = {
        "head_matrix": torch.as_tensor(head_matrix, dtype=dtype, device=device),
        "camera_location": torch.as_tensor(camera_location, dtype=dtype, device=device),
        "camera_rotation": torch.as_tensor(camera_rotation, dtype=dtype, device=device),
        "fov": torch.as_tensor(float(fov), dtype=dtype, device=device),
    }
    if resolution is not None:
        out["resolution"] = torch.as_tensor(resolution, dtype=torch.int32, device=device)
    return out


def stack_mat_tensors(mat_items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not mat_items:
        raise ValueError("mat_items must not be empty")
    keys = tuple(mat_items[0].keys())
    stacked: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in mat_items]
        stacked[key] = torch.stack(values, dim=0)
    return stacked


def normalize_torch(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(vector, dim=-1, eps=eps)


def create_rotation_matrix_torch(rotation_degrees: torch.Tensor) -> torch.Tensor:
    pitch = torch.deg2rad(rotation_degrees[..., 0])
    yaw = torch.deg2rad(rotation_degrees[..., 1])
    roll = torch.deg2rad(rotation_degrees[..., 2])

    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)

    row0 = torch.stack([cos_p * cos_y, cos_p * sin_y, sin_p], dim=-1)
    row1 = torch.stack(
        [
            sin_r * sin_p * cos_y - cos_r * sin_y,
            sin_r * sin_p * sin_y + cos_r * cos_y,
            -sin_r * cos_p,
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            -cos_r * sin_p * cos_y - sin_r * sin_y,
            -cos_r * sin_p * sin_y + sin_r * cos_y,
            cos_r * cos_p,
        ],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def create_transform_matrix_torch(
    location: torch.Tensor,
    rotation_degrees: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = torch.ones_like(location)
    rotation_matrix = create_rotation_matrix_torch(rotation_degrees)
    scaled_rotation = rotation_matrix * scale.unsqueeze(-2)

    batch_shape = location.shape[:-1]
    out = torch.eye(4, dtype=location.dtype, device=location.device).expand(batch_shape + (4, 4)).clone()
    out[..., :3, :3] = scaled_rotation
    out[..., 3, :3] = location
    return out


def build_viewmats_and_intrinsics_torch(
    camera_location: torch.Tensor,
    camera_rotation: torch.Tensor,
    fov_degrees: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.ones_like(camera_location)
    camera_matrix = create_transform_matrix_torch(camera_location, camera_rotation, scale=scale)

    coord_conversion = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=camera_matrix.dtype,
        device=camera_matrix.device,
    ).expand_as(camera_matrix)

    view_row = torch.linalg.inv(camera_matrix) @ coord_conversion
    viewmat = view_row.transpose(-1, -2).contiguous()

    focal_ndc = 1.0 / torch.tan(torch.deg2rad(fov_degrees) * 0.5)
    K = torch.zeros(camera_location.shape[:-1] + (3, 3), dtype=camera_location.dtype, device=camera_location.device)
    K[..., 0, 0] = float(width) * focal_ndc * 0.5
    K[..., 1, 1] = -float(height) * focal_ndc * 0.5
    K[..., 0, 2] = float(width) * 0.5
    K[..., 1, 2] = float(height) * 0.5
    K[..., 2, 2] = 1.0
    return viewmat, K


def transform_vertices_to_world_torch(vertices_local: torch.Tensor, head_matrix_row: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(vertices_local.shape[:-1] + (1,), dtype=vertices_local.dtype, device=vertices_local.device)
    vertices_h = torch.cat([vertices_local, ones], dim=-1)
    world_h = torch.matmul(vertices_h, head_matrix_row)
    return world_h[..., :3]


def compute_mesh_face_data_torch(
    vertices_world: torch.Tensor,
    faces: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tris = vertices_world[:, faces.long()]
    edge01 = tris[:, :, 1] - tris[:, :, 0]
    edge02 = tris[:, :, 2] - tris[:, :, 0]
    cross = torch.cross(edge01, edge02, dim=-1)
    face_areas = 0.5 * torch.linalg.norm(cross, dim=-1)
    normals = normalize_torch(cross, eps=eps)

    axis_u = edge01 - (edge01 * normals).sum(dim=-1, keepdim=True) * normals
    axis_u_norm = torch.linalg.norm(axis_u, dim=-1, keepdim=True)
    axis_u_fallback = edge02 - (edge02 * normals).sum(dim=-1, keepdim=True) * normals
    axis_u = torch.where(axis_u_norm > eps, axis_u, axis_u_fallback)
    axis_u_norm = torch.linalg.norm(axis_u, dim=-1, keepdim=True)

    helper_x = torch.tensor([1.0, 0.0, 0.0], dtype=vertices_world.dtype, device=vertices_world.device)
    helper_y = torch.tensor([0.0, 1.0, 0.0], dtype=vertices_world.dtype, device=vertices_world.device)
    helper = torch.where(
        normals[..., 0:1].abs() < 0.9,
        helper_x.view(1, 1, 3),
        helper_y.view(1, 1, 3),
    )
    axis_u_fallback2 = torch.cross(helper.expand_as(normals), normals, dim=-1)
    axis_u = torch.where(axis_u_norm > eps, axis_u, axis_u_fallback2)
    axis_u = normalize_torch(axis_u, eps=eps)
    axis_v = normalize_torch(torch.cross(normals, axis_u, dim=-1), eps=eps)
    return tris, normals, face_areas, torch.stack([axis_u, axis_v, normals], dim=-2)


def rotation_matrices_to_quaternions_torch(rotations: torch.Tensor) -> torch.Tensor:
    original_shape = rotations.shape[:-2]
    r = rotations.reshape(-1, 3, 3)
    m00 = r[:, 0, 0]
    m11 = r[:, 1, 1]
    m22 = r[:, 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros((r.shape[0], 4), dtype=r.dtype, device=r.device)

    mask_trace = trace > 0.0
    if mask_trace.any():
        t = torch.sqrt(trace[mask_trace] + 1.0).clamp_min(1e-8) * 2.0
        q[mask_trace, 0] = 0.25 * t
        q[mask_trace, 1] = (r[mask_trace, 2, 1] - r[mask_trace, 1, 2]) / t
        q[mask_trace, 2] = (r[mask_trace, 0, 2] - r[mask_trace, 2, 0]) / t
        q[mask_trace, 3] = (r[mask_trace, 1, 0] - r[mask_trace, 0, 1]) / t

    mask_x = (~mask_trace) & (m00 > m11) & (m00 > m22)
    if mask_x.any():
        t = torch.sqrt(1.0 + m00[mask_x] - m11[mask_x] - m22[mask_x]).clamp_min(1e-8) * 2.0
        q[mask_x, 0] = (r[mask_x, 2, 1] - r[mask_x, 1, 2]) / t
        q[mask_x, 1] = 0.25 * t
        q[mask_x, 2] = (r[mask_x, 0, 1] + r[mask_x, 1, 0]) / t
        q[mask_x, 3] = (r[mask_x, 0, 2] + r[mask_x, 2, 0]) / t

    mask_y = (~mask_trace) & (~mask_x) & (m11 > m22)
    if mask_y.any():
        t = torch.sqrt(1.0 + m11[mask_y] - m00[mask_y] - m22[mask_y]).clamp_min(1e-8) * 2.0
        q[mask_y, 0] = (r[mask_y, 0, 2] - r[mask_y, 2, 0]) / t
        q[mask_y, 1] = (r[mask_y, 0, 1] + r[mask_y, 1, 0]) / t
        q[mask_y, 2] = 0.25 * t
        q[mask_y, 3] = (r[mask_y, 1, 2] + r[mask_y, 2, 1]) / t

    mask_z = (~mask_trace) & (~mask_x) & (~mask_y)
    if mask_z.any():
        t = torch.sqrt(1.0 + m22[mask_z] - m00[mask_z] - m11[mask_z]).clamp_min(1e-8) * 2.0
        q[mask_z, 0] = (r[mask_z, 1, 0] - r[mask_z, 0, 1]) / t
        q[mask_z, 1] = (r[mask_z, 0, 2] + r[mask_z, 2, 0]) / t
        q[mask_z, 2] = (r[mask_z, 1, 2] + r[mask_z, 2, 1]) / t
        q[mask_z, 3] = 0.25 * t

    q = normalize_torch(q, eps=1e-8)
    return q.reshape(original_shape + (4,))


def sample_texture_bilinear_torch(texture: torch.Tensor, uv_coords: torch.Tensor) -> torch.Tensor:
    if texture.ndim == 3:
        if texture.shape[0] in (1, 3, 4):
            texture = texture.unsqueeze(0)
        else:
            texture = texture.permute(2, 0, 1).unsqueeze(0)
    elif texture.ndim == 4 and texture.shape[-1] in (1, 3, 4):
        texture = texture.permute(0, 3, 1, 2)
    elif texture.ndim != 4:
        raise ValueError(f"texture must be [C,H,W], [H,W,C], [B,C,H,W], or [B,H,W,C], got {tuple(texture.shape)}")

    if texture.shape[1] < 3:
        raise ValueError(f"texture must have at least 3 channels, got {tuple(texture.shape)}")

    batch = texture.shape[0]
    if uv_coords.ndim == 2:
        uv_coords = uv_coords.unsqueeze(0).expand(batch, -1, -1)
    elif uv_coords.ndim != 3:
        raise ValueError(f"uv_coords must be [G,2] or [B,G,2], got {tuple(uv_coords.shape)}")

    if uv_coords.shape[0] != batch:
        if uv_coords.shape[0] == 1:
            uv_coords = uv_coords.expand(batch, -1, -1)
        else:
            raise ValueError(f"uv batch size {uv_coords.shape[0]} does not match texture batch size {batch}")

    grid = torch.empty((batch, uv_coords.shape[1], 1, 2), dtype=texture.dtype, device=texture.device)
    grid[..., 0] = uv_coords[..., 0].clamp(0.0, 1.0).mul(2.0).sub(1.0).unsqueeze(-1)
    grid[..., 1] = uv_coords[..., 1].clamp(0.0, 1.0).mul(-2.0).add(1.0).unsqueeze(-1)
    sampled = F.grid_sample(texture, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[..., 0].permute(0, 2, 1).contiguous()


def _normalize_vertices_batch(mesh_positions: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if mesh_positions.ndim == 2:
        return mesh_positions.unsqueeze(0), True
    if mesh_positions.ndim != 3:
        raise ValueError(f"mesh_positions must be [N,3] or [B,N,3], got {tuple(mesh_positions.shape)}")
    return mesh_positions, False


def _normalize_texture_batch(texture: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    texture = texture.to(device=device, dtype=dtype)
    if texture.ndim == 3:
        if texture.shape[0] in (1, 3, 4):
            texture = texture.unsqueeze(0)
        else:
            texture = texture.permute(2, 0, 1).unsqueeze(0)
    elif texture.ndim == 4 and texture.shape[-1] in (1, 3, 4):
        texture = texture.permute(0, 3, 1, 2)
    elif texture.ndim != 4:
        raise ValueError(f"texture must be [C,H,W], [H,W,C], [B,C,H,W], or [B,H,W,C], got {tuple(texture.shape)}")

    if texture.shape[0] != batch_size:
        if texture.shape[0] == 1:
            texture = texture.expand(batch_size, -1, -1, -1)
        else:
            raise ValueError(f"texture batch size {texture.shape[0]} does not match mesh batch size {batch_size}")
    return texture.contiguous()


def _normalize_mat_batch(
    mat: Mapping[str, torch.Tensor] | torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    if torch.is_tensor(mat):
        if mat.ndim == 1:
            mat = mat.unsqueeze(0)
        if mat.ndim != 2 or mat.shape[-1] != 7:
            raise ValueError("Tensor mat input must have shape [7] or [B,7] as [cam_xyz(3), cam_rot(3), fov]")
        if mat.shape[0] != batch_size:
            if mat.shape[0] == 1:
                mat = mat.expand(batch_size, -1)
            else:
                raise ValueError(f"mat batch size {mat.shape[0]} does not match mesh batch size {batch_size}")
        identity = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        return {
            "head_matrix": identity.clone(),
            "camera_location": mat[:, 0:3].to(device=device, dtype=dtype),
            "camera_rotation": mat[:, 3:6].to(device=device, dtype=dtype),
            "fov": mat[:, 6].to(device=device, dtype=dtype),
        }

    out: dict[str, torch.Tensor] = {}
    for key in ("head_matrix", "camera_location", "camera_rotation", "fov"):
        if key not in mat:
            raise KeyError(f"mat is missing required key '{key}'")
        value = mat[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=dtype, device=device)
        else:
            value = value.to(device=device, dtype=dtype)
        if key == "head_matrix":
            if value.ndim == 2:
                value = value.unsqueeze(0)
        elif key == "fov":
            if value.ndim == 0:
                value = value.unsqueeze(0)
        else:
            if value.ndim == 1:
                value = value.unsqueeze(0)

        if value.shape[0] != batch_size:
            if value.shape[0] == 1:
                value = value.expand((batch_size,) + value.shape[1:])
            else:
                raise ValueError(f"mat['{key}'] batch size {value.shape[0]} does not match mesh batch size {batch_size}")
        out[key] = value.contiguous()
    return out


def _try_wrapper_rasterize(
    *,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    ray_transforms: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    normals: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    background: torch.Tensor,
) -> torch.Tensor | None:
    global _gsplat_wrapper_runtime_available
    if (not _gsplat_wrapper_runtime_available) or _gsplat_wrapper is None:
        return None

    tile_width = int(math.ceil(float(width) / float(tile_size)))
    tile_height = int(math.ceil(float(height) / float(tile_size)))

    try:
        isect_ids, flatten_ids, _ = _gsplat_wrapper.isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
        )
        isect_offsets = _gsplat_wrapper.isect_offset_encode(
            isect_ids,
            int(means2d.shape[0]),
            tile_width,
            tile_height,
        )
        densify = torch.zeros_like(means2d)
        render_colors, _, _, _, _ = _gsplat_wrapper.rasterize_to_pixels_2dgs(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=background,
        )
        return render_colors.permute(0, 3, 1, 2).contiguous()
    except Exception:
        _gsplat_wrapper_runtime_available = False
        return None


def _exclusive_prefix_product_by_segment(
    values: torch.Tensor,
    segment_start: torch.Tensor,
) -> torch.Tensor:
    log_values = torch.log(values.clamp_min(1e-8))
    inclusive = torch.cumsum(log_values, dim=0)
    start_indices = torch.nonzero(segment_start, as_tuple=False).flatten()
    segment_ids = torch.cumsum(segment_start.to(torch.int64), dim=0) - 1
    segment_bases = torch.zeros(start_indices.shape[0], dtype=log_values.dtype, device=log_values.device)
    if start_indices.numel() > 1:
        segment_bases[1:] = inclusive[start_indices[1:] - 1]
    exclusive_log = inclusive - log_values - segment_bases[segment_ids]
    return torch.exp(exclusive_log)


def rasterize_color_fallback(
    *,
    means2d: torch.Tensor,
    radii: torch.Tensor,
    depths: torch.Tensor,
    ray_transforms: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    width: int,
    height: int,
    background: torch.Tensor,
    chunk_size: int,
    alpha_threshold: float,
) -> torch.Tensor:
    batch_size = means2d.shape[0]
    device = means2d.device
    dtype = means2d.dtype
    total_pixels = int(width) * int(height)

    renders: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        depth_order = torch.argsort(depths[batch_idx], dim=0, descending=False)
        render_flat = torch.zeros((total_pixels, colors.shape[-1]), dtype=dtype, device=device)
        trans_flat = torch.ones((total_pixels,), dtype=dtype, device=device)

        for start in range(0, depth_order.shape[0], int(chunk_size)):
            order = depth_order[start : start + int(chunk_size)]
            means_chunk = means2d[batch_idx, order]
            radii_chunk = radii[batch_idx, order].to(torch.int64)
            ray_chunk = ray_transforms[batch_idx, order]
            colors_chunk = colors[batch_idx, order]
            opacity_chunk = opacities[batch_idx, order]

            xmin = (torch.floor(means_chunk[:, 0]) - radii_chunk[:, 0]).clamp(0, width - 1)
            xmax = (torch.ceil(means_chunk[:, 0]) + radii_chunk[:, 0]).clamp(0, width - 1)
            ymin = (torch.floor(means_chunk[:, 1]) - radii_chunk[:, 1]).clamp(0, height - 1)
            ymax = (torch.ceil(means_chunk[:, 1]) + radii_chunk[:, 1]).clamp(0, height - 1)
            box_w = (xmax - xmin + 1).clamp_min(0)
            box_h = (ymax - ymin + 1).clamp_min(0)
            valid = (box_w > 0) & (box_h > 0)
            if not bool(valid.any()):
                continue

            means_chunk = means_chunk[valid]
            ray_chunk = ray_chunk[valid]
            colors_chunk = colors_chunk[valid]
            opacity_chunk = opacity_chunk[valid]
            xmin = xmin[valid]
            ymin = ymin[valid]
            box_w = box_w[valid]
            box_h = box_h[valid]

            max_w = int(box_w.max().item())
            max_h = int(box_h.max().item())
            grid_x = torch.arange(max_w, device=device).view(1, 1, max_w)
            grid_y = torch.arange(max_h, device=device).view(1, max_h, 1)

            valid_mask = (grid_x < box_w.view(-1, 1, 1)) & (grid_y < box_h.view(-1, 1, 1))
            if not bool(valid_mask.any()):
                continue

            pixel_x = xmin.view(-1, 1, 1) + grid_x
            pixel_y = ymin.view(-1, 1, 1) + grid_y
            gaussian_local_ids = torch.arange(means_chunk.shape[0], device=device, dtype=torch.int64).view(-1, 1, 1)
            gaussian_local_ids = gaussian_local_ids.expand(-1, max_h, max_w)

            pixel_x = pixel_x.expand_as(valid_mask)[valid_mask].to(dtype)
            pixel_y = pixel_y.expand_as(valid_mask)[valid_mask].to(dtype)
            gaussian_local_ids = gaussian_local_ids[valid_mask]

            means_active = means_chunk[gaussian_local_ids]
            rays_active = ray_chunk[gaussian_local_ids]
            opacity_active = opacity_chunk[gaussian_local_ids]
            colors_active = colors_chunk[gaussian_local_ids]

            px_center = pixel_x + 0.5
            py_center = pixel_y + 0.5
            h_u = -rays_active[:, 0, :] + rays_active[:, 2, :] * px_center[:, None]
            h_v = -rays_active[:, 1, :] + rays_active[:, 2, :] * py_center[:, None]
            tmp = torch.cross(h_u, h_v, dim=-1)
            denom = torch.where(tmp[:, 2].abs() > 1e-8, tmp[:, 2], tmp[:, 2].new_full(tmp[:, 2].shape, 1e-8))
            us = tmp[:, 0] / denom
            vs = tmp[:, 1] / denom
            sigma_3d = us.square() + vs.square()
            delta_x = px_center - means_active[:, 0]
            delta_y = py_center - means_active[:, 1]
            sigma_2d = 2.0 * (delta_x.square() + delta_y.square())
            sigma = 0.5 * torch.minimum(sigma_3d, sigma_2d)
            alpha = torch.clamp(opacity_active * torch.exp(-sigma), max=0.999)

            active_mask = alpha > float(alpha_threshold)
            if not bool(active_mask.any()):
                continue

            alpha = alpha[active_mask]
            colors_active = colors_active[active_mask]
            pixel_x = pixel_x[active_mask].to(torch.int64)
            pixel_y = pixel_y[active_mask].to(torch.int64)
            pixel_ids = pixel_y * int(width) + pixel_x

            sort_idx = torch.argsort(pixel_ids, stable=True)
            pixel_ids = pixel_ids[sort_idx]
            alpha = alpha[sort_idx]
            colors_active = colors_active[sort_idx]

            segment_start = torch.ones(pixel_ids.shape[0], dtype=torch.bool, device=device)
            if pixel_ids.shape[0] > 1:
                segment_start[1:] = pixel_ids[1:] != pixel_ids[:-1]
            unique_pixels = pixel_ids[segment_start]
            segment_ids = torch.cumsum(segment_start.to(torch.int64), dim=0) - 1

            one_minus_alpha = 1.0 - alpha
            prefix = _exclusive_prefix_product_by_segment(one_minus_alpha, segment_start)
            weights = prefix * alpha

            chunk_colors = torch.zeros((unique_pixels.shape[0], colors.shape[-1]), dtype=dtype, device=device)
            chunk_colors = chunk_colors.index_add(0, segment_ids, weights[:, None] * colors_active)

            chunk_log_trans = torch.zeros((unique_pixels.shape[0],), dtype=dtype, device=device)
            chunk_log_trans = chunk_log_trans.index_add(0, segment_ids, torch.log(one_minus_alpha.clamp_min(1e-8)))
            chunk_trans = torch.exp(chunk_log_trans)

            incoming = trans_flat[unique_pixels]
            render_flat = render_flat.index_add(0, unique_pixels, incoming[:, None] * chunk_colors)
            trans_flat = trans_flat.scatter(0, unique_pixels, incoming * chunk_trans)

        render_flat = render_flat + trans_flat[:, None] * background[batch_idx].view(1, -1)
        renders.append(render_flat.view(height, width, colors.shape[-1]))

    return torch.stack(renders, dim=0).permute(0, 3, 1, 2).contiguous()


@dataclass(frozen=True)
class DifferentiableRendererConfig:
    image_width: int = 512
    image_height: int = 512
    target_area_percentile: float = 65.0
    max_subdivision_level: int = 5
    min_planar_scale: float = 0.03
    planar_extent_divisor: float = 4.0
    planar_scale_multiplier: float = 3.2
    thickness_ratio: float = 0.06
    opacity: float = 0.99
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_plane: float = 0.01
    far_plane: float = 1.0e10
    projection_eps: float = 1.0e-6
    tile_size: int = 16
    rasterize_chunk_size: int = 2048
    alpha_threshold: float = 1.0e-4
    prefer_gsplat_wrapper: bool = True


class DifferentiableHeadRenderer(nn.Module):
    def __init__(
        self,
        reference_vertices: np.ndarray | torch.Tensor,
        uvs: np.ndarray | torch.Tensor,
        faces: np.ndarray | torch.Tensor,
        uv_faces: np.ndarray | torch.Tensor | None = None,
        config: DifferentiableRendererConfig | None = None,
        sampling_cache_path: str | Path | None = None,
        rebuild_sampling_cache: bool = False,
    ) -> None:
        super().__init__()

        self.config = config or DifferentiableRendererConfig()
        reference_vertices_np = _ensure_numpy_float32(reference_vertices)
        uvs_np = _ensure_numpy_float32(uvs)
        faces_np = _ensure_numpy_int32(faces)
        uv_faces_np = _ensure_numpy_int32(uv_faces if uv_faces is not None else faces_np)

        sampling_plan, sampling_cache_status = resolve_sampling_plan(
            reference_vertices=reference_vertices_np,
            uvs=uvs_np,
            faces=faces_np,
            uv_faces=uv_faces_np,
            target_area_percentile=self.config.target_area_percentile,
            max_subdivision_level=self.config.max_subdivision_level,
            sampling_cache_path=sampling_cache_path,
            rebuild_sampling_cache=rebuild_sampling_cache,
        )

        self.sampling_cache_status = sampling_cache_status
        self.sampling_cache_path = str(Path(sampling_cache_path).resolve()) if sampling_cache_path else None

        self.register_buffer("faces", torch.as_tensor(faces_np, dtype=torch.int64))
        self.register_buffer("uv_faces", torch.as_tensor(uv_faces_np, dtype=torch.int64))
        self.register_buffer("sample_face_indices", torch.as_tensor(sampling_plan["sample_face_indices"], dtype=torch.int64))
        self.register_buffer("sample_barys", torch.as_tensor(sampling_plan["sample_barys"], dtype=torch.float32))
        self.register_buffer("sample_uvs", torch.as_tensor(sampling_plan["sample_uvs"], dtype=torch.float32))
        self.register_buffer("face_sample_counts", torch.as_tensor(sampling_plan["face_sample_counts"], dtype=torch.float32))
        self.register_buffer("background", torch.tensor(self.config.background, dtype=torch.float32))
        self.register_buffer("sample_face_counts", self.face_sample_counts[self.sample_face_indices].clone())

    def forward(
        self,
        mat: Mapping[str, torch.Tensor] | torch.Tensor,
        mesh_positions: torch.Tensor,
        texture: torch.Tensor,
    ) -> torch.Tensor:
        mesh_batch, squeezed = _normalize_vertices_batch(mesh_positions)
        device = mesh_batch.device
        dtype = mesh_batch.dtype
        batch_size = mesh_batch.shape[0]

        mat_batch = _normalize_mat_batch(mat, batch_size=batch_size, device=device, dtype=dtype)
        texture_batch = _normalize_texture_batch(texture, batch_size=batch_size, device=device, dtype=dtype)
        render_batch = self.render_batch(mat_batch=mat_batch, mesh_batch=mesh_batch, texture_batch=texture_batch)
        return render_batch[0] if squeezed else render_batch

    def render_batch(
        self,
        *,
        mat_batch: Mapping[str, torch.Tensor],
        mesh_batch: torch.Tensor,
        texture_batch: torch.Tensor,
    ) -> torch.Tensor:
        world_vertices = transform_vertices_to_world_torch(mesh_batch, mat_batch["head_matrix"])
        tris, face_normals, face_areas, face_frames = compute_mesh_face_data_torch(world_vertices, self.faces)

        sample_faces = self.sample_face_indices.long()
        sample_barys = self.sample_barys.to(device=world_vertices.device, dtype=world_vertices.dtype)
        sample_face_counts = self.sample_face_counts.to(device=world_vertices.device, dtype=world_vertices.dtype)

        sample_tris = tris[:, sample_faces]
        means = (
            sample_tris[:, :, 0] * sample_barys[:, 0].view(1, -1, 1)
            + sample_tris[:, :, 1] * sample_barys[:, 1].view(1, -1, 1)
            + sample_tris[:, :, 2] * sample_barys[:, 2].view(1, -1, 1)
        )

        sample_face_areas = face_areas[:, sample_faces]
        sample_area = (sample_face_areas / sample_face_counts.view(1, -1)).clamp_min(1e-12)
        planar_scale = (
            sample_area.sqrt() / float(self.config.planar_extent_divisor) * float(self.config.planar_scale_multiplier)
        ).clamp_min(float(self.config.min_planar_scale))
        thickness = (planar_scale * float(self.config.thickness_ratio)).clamp_min(float(self.config.min_planar_scale) * 0.25)
        scales = torch.stack([planar_scale, planar_scale, thickness], dim=-1)

        sample_frames = face_frames[:, sample_faces]
        quats = rotation_matrices_to_quaternions_torch(sample_frames)
        colors = sample_texture_bilinear_torch(
            texture_batch,
            self.sample_uvs.to(device=texture_batch.device, dtype=texture_batch.dtype),
        )[..., :3].contiguous()
        opacities = torch.full(
            (mesh_batch.shape[0], self.sample_face_indices.shape[0]),
            float(self.config.opacity),
            dtype=mesh_batch.dtype,
            device=mesh_batch.device,
        )

        viewmats, Ks = build_viewmats_and_intrinsics_torch(
            camera_location=mat_batch["camera_location"],
            camera_rotation=mat_batch["camera_rotation"],
            fov_degrees=mat_batch["fov"],
            width=int(self.config.image_width),
            height=int(self.config.image_height),
        )
        viewmats = viewmats.unsqueeze(1)
        Ks = Ks.unsqueeze(1)

        radii, means2d, depths, ray_transforms, normals = _fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            int(self.config.image_width),
            int(self.config.image_height),
            near_plane=float(self.config.near_plane),
            far_plane=float(self.config.far_plane),
            eps=float(self.config.projection_eps),
        )
        radii = radii[:, 0]
        means2d = means2d[:, 0]
        depths = depths[:, 0]
        ray_transforms = ray_transforms[:, 0]
        normals = normals[:, 0]

        background = self.background.to(device=mesh_batch.device, dtype=mesh_batch.dtype).view(1, 3).expand(mesh_batch.shape[0], -1)
        render = None
        if bool(self.config.prefer_gsplat_wrapper):
            render = _try_wrapper_rasterize(
                means2d=means2d,
                radii=radii,
                depths=depths,
                ray_transforms=ray_transforms,
                colors=colors,
                opacities=opacities,
                normals=normals,
                width=int(self.config.image_width),
                height=int(self.config.image_height),
                tile_size=int(self.config.tile_size),
                background=background,
            )
        if render is None:
            render = rasterize_color_fallback(
                means2d=means2d,
                radii=radii,
                depths=depths,
                ray_transforms=ray_transforms,
                colors=colors,
                opacities=opacities,
                width=int(self.config.image_width),
                height=int(self.config.image_height),
                background=background,
                chunk_size=int(self.config.rasterize_chunk_size),
                alpha_threshold=float(self.config.alpha_threshold),
            )
        return render.clamp(0.0, 1.0)
