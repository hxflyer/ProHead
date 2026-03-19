"""
Precompute geometry normal maps for all dataset samples and save as PNG.

Usage:
    python precompute_geo_normal.py --data_roots G:/CapturedFrames_final8_processed G:/CapturedFrames_final9_processed
    python precompute_geo_normal.py --data_roots G:/CapturedFrames_final*_processed --workers 8
"""
import argparse
import glob
import multiprocessing
import os
import sys

import cv2
import numpy as np

from mat_load_helper import compute_vertex_depth, get_world_to_view_rotation, load_matrix_data, project_3d_to_2d_cpp_exact
from obj_load_helper import load_uv_obj_file


# ---------------------------------------------------------------------------
# Mesh face loading (from model/ OBJ files)
# ---------------------------------------------------------------------------

def _load_combined_mesh_triangle_faces(model_dir: str = "model") -> np.ndarray:
    part_files = ["mesh_head.obj", "mesh_eye_l.obj", "mesh_eye_r.obj", "mesh_mouth.obj"]
    triangles = []
    offset = 0
    for file_name in part_files:
        obj_path = os.path.join(model_dir, file_name)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing mesh OBJ: {obj_path}")
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(obj_path, triangulate=True)
        if verts is None or uvs is None or v_faces is None:
            raise ValueError(f"Failed to load from {obj_path}")
        triangles.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))
    if not triangles:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(triangles, axis=0).astype(np.int32, copy=False)


def _remap_triangle_faces_after_vertex_filter(
    triangle_faces: np.ndarray,
    kept_vertex_indices: np.ndarray,
    original_vertex_count: int,
) -> np.ndarray:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)
    filtered = np.where(remap < 0)[0]
    if filtered.size > 0:
        right = np.searchsorted(kept, filtered, side="left")
        right = np.clip(right, 0, kept.shape[0] - 1)
        left = np.clip(right - 1, 0, kept.shape[0] - 1)
        right_dist = np.abs(kept[right] - filtered)
        left_dist = np.abs(kept[left] - filtered)
        nearest = np.where(left_dist <= right_dist, left, right)
        remap[filtered] = nearest
    return remap[tri].astype(np.int32, copy=False)


def _load_mesh_faces(model_dir: str = "model") -> dict[int, np.ndarray]:
    full_faces = _load_combined_mesh_triangle_faces(model_dir)
    supported: dict[int, np.ndarray] = {}
    if full_faces.size > 0:
        full_vertex_count = int(full_faces.max()) + 1
        supported[full_vertex_count] = full_faces
        mesh_indices_path = os.path.join(model_dir, "mesh_indices.npy")
        if os.path.exists(mesh_indices_path):
            mesh_indices = np.load(mesh_indices_path).astype(np.int64, copy=False)
            filtered_faces = _remap_triangle_faces_after_vertex_filter(
                full_faces, mesh_indices, full_vertex_count
            )
            supported[int(mesh_indices.shape[0])] = filtered_faces
    return supported


# ---------------------------------------------------------------------------
# Geometry loading and rendering
# ---------------------------------------------------------------------------

def _load_geometry(filepath: str):
    """Load raw XYZ from mesh txt file. Returns None on failure."""
    if not os.path.exists(filepath):
        return None
    try:
        geom = []
        with open(filepath, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) >= 3:
                    geom.append([float(parts[0]), float(parts[1]), float(parts[2])])
        arr = np.asarray(geom, dtype=np.float32)
        return arr if arr.shape[0] > 0 else None
    except Exception:
        return None


def _compute_vertex_normals_encoded(mesh_xyz: np.ndarray, faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    xyz = mesh_xyz[:, :3].astype(np.float32, copy=False)
    v0, v1, v2 = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32, copy=False)
    vertex_normals = np.zeros_like(xyz, dtype=np.float32)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)
    norm = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.clip(norm, 1e-6, None)
    n01 = vertex_normals * 0.5 + 0.5
    return np.stack([1 - n01[:, 0], 1 - n01[:, 1], n01[:, 2]], axis=1).astype(np.float32, copy=False)


def _render_projected_vertex_attrs_to_image(
    projected_xy: np.ndarray,
    vertex_depth: np.ndarray,
    vertex_attrs: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if _worker_glctx is not None:
        return _render_nvdiffrast(projected_xy, vertex_depth, vertex_attrs, faces, width, height)
    return _render_scipy(projected_xy, vertex_depth, vertex_attrs, faces, width, height)


def _render_nvdiffrast(
    projected_xy: np.ndarray,
    vertex_depth: np.ndarray,
    vertex_attrs: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    import nvdiffrast.torch as dr
    import torch

    pts = projected_xy.astype(np.float32)
    depth = vertex_depth.astype(np.float32)
    attrs = vertex_attrs.astype(np.float32)
    V = pts.shape[0]

    # Pixel coords → NDC. nvdiffrast: x=-1 left, +1 right; y=-1 bottom, +1 top.
    x_ndc = (pts[:, 0] / width) * 2.0 - 1.0
    y_ndc = 1.0 - (pts[:, 1] / height) * 2.0

    # compute_vertex_depth returns near/vz: LARGER = closer to camera.
    # nvdiffrast keeps the fragment with the SMALLEST z_ndc (OpenGL near=-1 wins).
    # So map: larger depth (closer) → more negative z_ndc.
    d_min, d_max = float(depth.min()), float(depth.max())
    z_ndc = 1.0 - (depth - d_min) / max(d_max - d_min, 1e-6) * 2.0

    verts = np.stack([x_ndc, y_ndc, z_ndc, np.ones(V, dtype=np.float32)], axis=1)
    verts_t = torch.from_numpy(verts).cuda().unsqueeze(0)       # [1, V, 4]
    faces_t = torch.from_numpy(faces.astype(np.int32)).cuda()   # [F, 3]
    attrs_t = torch.from_numpy(attrs).cuda().unsqueeze(0)       # [1, V, C]

    rast, _ = dr.rasterize(_worker_glctx, verts_t, faces_t, resolution=[height, width])
    interp, _ = dr.interpolate(attrs_t, rast, faces_t)          # [1, H, W, C]

    # nvdiffrast row 0 = y_ndc=-1 (bottom). Flip to image convention (row 0 = top).
    result = interp[0].cpu().numpy()[::-1].copy()
    mask   = rast[0, :, :, 3].cpu().numpy()[::-1]
    result[mask == 0] = 0.0
    return np.clip(result, 0.0, 1.0)


def _render_scipy(
    projected_xy: np.ndarray,
    vertex_depth: np.ndarray,
    vertex_attrs: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    from scipy.interpolate import LinearNDInterpolator

    render = np.zeros((height, width, vertex_attrs.shape[1]), dtype=np.float32)
    pts = projected_xy.astype(np.float32)
    depth = vertex_depth.astype(np.float32)
    attrs = vertex_attrs.astype(np.float32)

    visible = (
        np.isfinite(pts).all(axis=1)
        & (depth > 0)
        & (pts[:, 0] >= 0) & (pts[:, 0] < width)
        & (pts[:, 1] >= 0) & (pts[:, 1] < height)
    )
    if visible.sum() < 3:
        return render

    face_pts = pts[faces].astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, face_pts, 255)                            # type: ignore[call-overload]

    interp = LinearNDInterpolator(pts[visible], attrs[visible], fill_value=0.0)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    render = interp(grid_x, grid_y).astype(np.float32)
    render[mask == 0] = 0.0
    return np.clip(render, 0.0, 1.0)


def render_geometry_normal(mesh_raw_xyz: np.ndarray, mat_path: str, src_w: int, src_h: int,
                           mesh_faces: dict[int, np.ndarray]) -> np.ndarray:
    """Render geometry normal map in source image space. Returns float32 [H,W,3]."""
    geometry_normal = np.zeros((src_h, src_w, 3), dtype=np.float32)
    faces = mesh_faces.get(mesh_raw_xyz.shape[0])
    if faces is None or faces.size == 0:
        return geometry_normal
    try:
        matrix_data = load_matrix_data(mat_path)
        mesh_f32 = mesh_raw_xyz.astype(np.float32)
        projected_xy = project_3d_to_2d_cpp_exact(mesh_f32, matrix_data).astype(np.float32, copy=False)
        depth = compute_vertex_depth(mesh_f32, matrix_data).astype(np.float32, copy=False)
    except Exception:
        return geometry_normal
    resolution = matrix_data.get("resolution", (src_w, src_h))
    proj_w = float(max(1, int(resolution[0])))
    proj_h = float(max(1, int(resolution[1])))
    if abs(proj_w - float(src_w)) > 1e-3:
        projected_xy[:, 0] *= float(src_w) / proj_w
    if abs(proj_h - float(src_h)) > 1e-3:
        projected_xy[:, 1] *= float(src_h) / proj_h
    head_mat = matrix_data.get("head_matrix")
    if head_mat is not None:
        ones = np.ones((mesh_f32.shape[0], 1), dtype=np.float32)
        v_world = (np.concatenate([mesh_f32, ones], axis=1) @ head_mat.astype(np.float32))[:, :3]
    else:
        v_world = mesh_f32
    v_view = v_world @ get_world_to_view_rotation(matrix_data)
    vertex_normals = _compute_vertex_normals_encoded(v_view, faces)
    return _render_projected_vertex_attrs_to_image(projected_xy, depth, vertex_normals, faces, src_w, src_h)


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------

def _parse_sample_id(color_file: str) -> str:
    base = os.path.basename(color_file)
    name = os.path.splitext(base)[0]
    name = name[len("Color_"):]
    for suffix in ["_gemini", "_flux", "_seedream"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _collect_samples(data_root: str) -> list[tuple]:
    color_files = glob.glob(os.path.join(data_root, "Color_*"))
    color_files = [
        f for f in color_files
        if os.path.basename(f).startswith("Color_")
        and not os.path.basename(f).endswith("_mask.png")
    ]
    samples = []
    for color_file in color_files:
        sample_id = _parse_sample_id(color_file)
        mesh_path = _find_first_existing([
            os.path.join(data_root, "mesh", f"mesh_{sample_id}.txt"),
            os.path.join(data_root, "mesh", f"{sample_id}.txt"),
        ])
        mat_path = _find_first_existing([
            os.path.join(data_root, "mat", f"Mats_{sample_id}.txt"),
            os.path.join(data_root, f"Mats_{sample_id}.txt"),
        ])
        if mesh_path is not None and mat_path is not None:
            samples.append((data_root, sample_id, color_file, mesh_path, mat_path))
    return samples


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_worker_mesh_faces: dict[int, np.ndarray] | None = None
_worker_glctx = None


def _worker_init(model_dir: str):
    global _worker_mesh_faces, _worker_glctx
    _worker_mesh_faces = _load_mesh_faces(model_dir=model_dir)
    try:
        import nvdiffrast.torch as dr
        _worker_glctx = dr.RasterizeCudaContext()
    except Exception:
        _worker_glctx = None


def _render_one(args: tuple) -> str:
    data_root, sample_id, color_file, mesh_path, mat_path = args

    out_dir = os.path.join(data_root, "geo_normal")
    out_path = os.path.join(out_dir, f"GeoNormal_{sample_id}.png")

    img = cv2.imread(color_file, cv2.IMREAD_COLOR)
    if img is None:
        return f"no_img {sample_id}"
    src_h, src_w = img.shape[:2]

    mesh_raw_xyz = _load_geometry(mesh_path)
    if mesh_raw_xyz is None:
        return f"no_mesh {sample_id}"

    geometry_normal = render_geometry_normal(mesh_raw_xyz, mat_path, src_w, src_h, _worker_mesh_faces)  # type: ignore[arg-type]

    if geometry_normal.shape[:2] != (src_h, src_w):
        geometry_normal = cv2.resize(geometry_normal, (src_w, src_h), interpolation=cv2.INTER_LINEAR)

    os.makedirs(out_dir, exist_ok=True)
    out_u8 = np.clip(geometry_normal * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))
    return f"ok {sample_id}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_roots", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--model_dir", default="model")
    args = parser.parse_args()

    data_roots = []
    for pattern in args.data_roots:
        expanded = glob.glob(pattern)
        data_roots.extend(expanded if expanded else [pattern])
    data_roots = [r for r in data_roots if os.path.isdir(r)]

    if not data_roots:
        print("No valid data roots found.", file=sys.stderr)
        sys.exit(1)

    all_samples = []
    for root in data_roots:
        samples = _collect_samples(root)
        print(f"{root}: {len(samples)} samples with mesh+mat")
        all_samples.extend(samples)

    print(f"Total: {len(all_samples)} samples, workers={args.workers}")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with multiprocessing.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.model_dir,),
    ) as pool:
        counts = {"ok": 0, "skip": 0, "fail": 0}
        results = pool.imap_unordered(_render_one, all_samples, chunksize=4)
        if tqdm is not None:
            results = tqdm(results, total=len(all_samples), unit="img")
        for i, result in enumerate(results):
            tag = result.split()[0]
            counts["ok" if tag == "ok" else "skip" if tag == "skip" else "fail"] += 1
            if tqdm is not None:
                results.set_postfix(ok=counts["ok"], skip=counts["skip"], fail=counts["fail"])  # type: ignore[union-attr]
            elif (i + 1) % 100 == 0 or (i + 1) == len(all_samples):
                print(f"  {i+1}/{len(all_samples)}  ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}")

    print("Done.")


if __name__ == "__main__":
    main()
