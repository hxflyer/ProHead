from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .camera_io import compute_vertex_depth, load_matrix_data


@dataclass
class GeometryTemplateSet:
    landmark_indices: np.ndarray | None
    mesh_indices: np.ndarray | None
    default_landmarks: np.ndarray
    default_mesh: np.ndarray
    template_landmark_depth: np.ndarray
    template_mesh_depth: np.ndarray


def load_landmark_pixels(filepath: Optional[str]) -> Optional[np.ndarray]:
    if filepath is None or not os.path.exists(filepath):
        return None
    pts = []
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 5:
                pts.append([float(parts[3]), float(parts[4])])
    return np.array(pts, dtype=np.float32) if pts else None


def load_geometry_txt(filepath: str, return_raw_xyz: bool = False):
    try:
        geom = []
        with open(filepath, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) >= 5:
                    geom.append([float(value) for value in parts[:5]])
                elif len(parts) >= 3:
                    geom.append([float(parts[0]), float(parts[1]), float(parts[2]), 0.0, 0.0])

        geom = np.array(geom, dtype=np.float32)
        if geom.shape[0] == 0:
            return (None, None) if return_raw_xyz else None

        xyz = geom[:, 0:3]
        raw_xyz = xyz.copy()
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        center = (min_xyz + max_xyz) / 2.0
        scale = np.max(max_xyz - min_xyz) / 2.0
        if scale > 0:
            geom[:, 0:3] = (xyz - center) / scale

        geom[:, 3:5] = geom[:, 3:5] / 1024.0
        if return_raw_xyz:
            return geom, raw_xyz
        return geom
    except Exception:
        return (None, None) if return_raw_xyz else None


def compute_geometry_found_mask(geom: Optional[np.ndarray]) -> np.ndarray:
    if geom is None or geom.ndim != 2 or geom.shape[1] < 5:
        return np.zeros((0,), dtype=bool)
    uv = geom[:, 3:5].astype(np.float32, copy=False)
    return np.isfinite(uv).all(axis=1) & np.all(uv >= 0.0, axis=1)


def load_geometry_template_set(model_dir: str = "assets/topology") -> GeometryTemplateSet:
    landmark_indices = None
    mesh_indices = None

    landmark_indices_path = os.path.join(model_dir, "landmark_indices.npy")
    mesh_indices_path = os.path.join(model_dir, "mesh_indices.npy")
    landmark_template_path = os.path.join(model_dir, "landmark_template.npy")
    mesh_template_path = os.path.join(model_dir, "mesh_template.npy")

    if os.path.exists(landmark_indices_path):
        landmark_indices = np.load(landmark_indices_path)
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path)

    landmark_template = np.load(landmark_template_path).astype(np.float32) if os.path.exists(landmark_template_path) else None
    mesh_template = np.load(mesh_template_path).astype(np.float32) if os.path.exists(mesh_template_path) else None
    if landmark_template is None or mesh_template is None:
        raise FileNotFoundError("Missing model templates for default geometry fallback.")

    if landmark_indices is not None and landmark_indices.max() < landmark_template.shape[0]:
        landmark_template = landmark_template[landmark_indices]
    if mesh_indices is not None and mesh_indices.max() < mesh_template.shape[0]:
        mesh_template = mesh_template[mesh_indices]

    if landmark_template.shape[1] < 5 or mesh_template.shape[1] < 5:
        raise ValueError("Template geometry must contain at least 5 dims (x,y,z,u,v).")

    if landmark_template.shape[1] >= 6:
        template_landmark_depth = landmark_template[:, 5].astype(np.float32, copy=True)
    else:
        template_landmark_depth = np.zeros(landmark_template.shape[0], dtype=np.float32)

    if mesh_template.shape[1] >= 6:
        template_mesh_depth = mesh_template[:, 5].astype(np.float32, copy=True)
    else:
        template_mesh_depth = np.zeros(mesh_template.shape[0], dtype=np.float32)

    return GeometryTemplateSet(
        landmark_indices=landmark_indices,
        mesh_indices=mesh_indices,
        default_landmarks=landmark_template[:, :5].astype(np.float32, copy=True),
        default_mesh=mesh_template[:, :5].astype(np.float32, copy=True),
        template_landmark_depth=template_landmark_depth,
        template_mesh_depth=template_mesh_depth,
    )


def apply_geometry_indices(
    geometry: np.ndarray | None,
    raw_xyz: np.ndarray | None,
    found_mask: np.ndarray | None,
    weights: np.ndarray | None,
    indices: np.ndarray | None,
):
    if geometry is None or indices is None or len(indices) == 0:
        return geometry, raw_xyz, found_mask, weights
    if indices.max() >= len(geometry):
        return geometry, raw_xyz, found_mask, weights

    geometry = geometry[indices]
    if raw_xyz is not None and indices.max() < len(raw_xyz):
        raw_xyz = raw_xyz[indices]
    if found_mask is not None and indices.max() < len(found_mask):
        found_mask = found_mask[indices]
    if weights is not None and indices.max() < len(weights):
        weights = weights[indices]
    return geometry, raw_xyz, found_mask, weights


def ensure_geometry_with_fallback(
    geometry: np.ndarray | None,
    raw_xyz: np.ndarray | None,
    found_mask: np.ndarray | None,
    weights: np.ndarray | None,
    fallback_geometry: np.ndarray,
):
    if geometry is not None and geometry.shape[0] == fallback_geometry.shape[0]:
        return geometry, raw_xyz, found_mask, weights

    geometry = fallback_geometry.copy()
    raw_xyz = None
    found_mask = np.zeros((geometry.shape[0],), dtype=bool)
    weights = np.zeros((geometry.shape[0],), dtype=np.float32)
    return geometry, raw_xyz, found_mask, weights


def _compute_depth_column(
    geometry: np.ndarray,
    raw_xyz: np.ndarray | None,
    matrix_data,
    template_depth: np.ndarray,
    sample_id: str,
    color_path: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if raw_xyz is not None and raw_xyz.shape[0] == geometry.shape[0]:
        xyz_original = raw_xyz.copy()
    else:
        xyz_original = geometry[:, 0:3].copy()

    depth_raw = compute_vertex_depth(xyz_original, matrix_data)
    if np.any(~np.isfinite(depth_raw)):
        print(f"[DEPTH WARNING] {sample_id}: {label}_depth_raw contains NaN/inf")
        print(f"  Path: {color_path}")
        print(f"  min={np.nanmin(depth_raw):.4f}, max={np.nanmax(depth_raw):.4f}")
        return np.zeros_like(depth_raw, dtype=np.float32), None

    depth_min = float(depth_raw.min())
    depth_max = float(depth_raw.max())
    depth_range = depth_max - depth_min
    if depth_min < 0:
        print(f"[DEPTH WARNING] {sample_id}: {label} depth_min={depth_min:.4f} < 0 (behind camera)")
        print(f"  Path: {color_path}")
    if depth_range < 1e-6:
        print(f"[DEPTH WARNING] {sample_id}: {label} depth_range={depth_range:.8f} (too small, using zeros)")
        print(f"  Path: {color_path}")
        return np.zeros_like(depth_raw, dtype=np.float32), None

    depth_norm = (depth_raw - depth_min) / (depth_range + 1e-8)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_norm = depth_norm - template_depth
    if np.any(~np.isfinite(depth_norm)):
        print(f"[DEPTH ERROR] {sample_id}: {label}_depth contains NaN/inf after normalization!")
        return np.zeros_like(depth_raw, dtype=np.float32), None

    rendered_depth = depth_raw.astype(np.float32, copy=False) if label == "mesh" else None
    return depth_norm.astype(np.float32, copy=False), rendered_depth


def _append_depth_column(geometry: np.ndarray, depth_column: np.ndarray) -> np.ndarray:
    if geometry.shape[1] >= 6:
        out = geometry.copy()
        out[:, 5] = depth_column.astype(np.float32, copy=False)
        return out
    return np.concatenate([geometry, depth_column[:, None].astype(np.float32, copy=False)], axis=1)


def attach_depth_channels(
    landmarks: np.ndarray | None,
    landmarks_raw_xyz: np.ndarray | None,
    mesh: np.ndarray | None,
    mesh_raw_xyz: np.ndarray | None,
    mat_path: str | None,
    template_landmark_depth: np.ndarray,
    template_mesh_depth: np.ndarray,
    sample_id: str,
    color_path: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    mesh_depth = None
    if mat_path and os.path.exists(mat_path):
        try:
            matrix_data = load_matrix_data(mat_path)
            if landmarks is not None:
                landmark_depth, _ = _compute_depth_column(
                    geometry=landmarks,
                    raw_xyz=landmarks_raw_xyz,
                    matrix_data=matrix_data,
                    template_depth=template_landmark_depth,
                    sample_id=sample_id,
                    color_path=color_path,
                    label="landmark",
                )
                landmarks = _append_depth_column(landmarks, landmark_depth)

            if mesh is not None:
                mesh_depth_norm, mesh_depth = _compute_depth_column(
                    geometry=mesh,
                    raw_xyz=mesh_raw_xyz,
                    matrix_data=matrix_data,
                    template_depth=template_mesh_depth,
                    sample_id=sample_id,
                    color_path=color_path,
                    label="mesh",
                )
                mesh = _append_depth_column(mesh, mesh_depth_norm)
        except Exception:
            if landmarks is not None:
                landmarks = _append_depth_column(
                    landmarks,
                    np.zeros((landmarks.shape[0],), dtype=np.float32),
                )
            if mesh is not None:
                mesh = _append_depth_column(
                    mesh,
                    np.zeros((mesh.shape[0],), dtype=np.float32),
                )
            mesh_depth = None
    else:
        if landmarks is not None and landmarks.shape[1] < 6:
            landmarks = _append_depth_column(
                landmarks,
                np.zeros((landmarks.shape[0],), dtype=np.float32),
            )
        if mesh is not None and mesh.shape[1] < 6:
            mesh = _append_depth_column(
                mesh,
                np.zeros((mesh.shape[0],), dtype=np.float32),
            )

    return landmarks, mesh, mesh_depth
