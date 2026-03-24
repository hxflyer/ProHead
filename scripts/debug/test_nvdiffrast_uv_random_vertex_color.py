import argparse
import os
import math

import cv2
import numpy as np
import torch
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
import nvdiffrast.torch as dr
from data_utils.obj_io import load_uv_obj_file


# Keep these in sync with training UV packing.
EYE_BOX_SIZE = 0.20
EYE_L_U_START = 0.005
EYE_R_U_START = 0.795
BOTTOM_MARGIN = 0.005

MOUTH_SPLIT_V = 0.61
MOUTH_BOX_SIZE = 0.25
MOUTH_V_GT_U_START = 0.50
MOUTH_V_LE_U_START = 0.22

HEAD_SCALE = 1.0
HEAD_TX = 0.0
HEAD_TY = 0.0


PART_FILES = {
    "head": "mesh_head.obj",
    "eye_l": "mesh_eye_l.obj",
    "eye_r": "mesh_eye_r.obj",
    "mouth": "mesh_mouth.obj",
}


def _transform_uv_center(uv: np.ndarray, scale: float, tx: float, ty: float) -> np.ndarray:
    center = uv.mean(axis=0, keepdims=True)
    out = (uv - center) * float(scale) + center
    out[:, 0] += float(tx)
    out[:, 1] += float(ty)
    return out.astype(np.float32, copy=False)


def _place_uv_in_box(
    uv: np.ndarray,
    u_start: float,
    v_start: float,
    box_size: float,
    align_bottom: bool,
) -> np.ndarray:
    if uv.shape[0] == 0:
        return uv.copy().astype(np.float32, copy=False)

    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    span = np.maximum(uv_max - uv_min, 1e-8)
    scale = float(box_size) / float(max(span[0], span[1]))

    uv_local = (uv - uv_min) * scale
    local_min = uv_local.min(axis=0)
    local_max = uv_local.max(axis=0)
    local_size = local_max - local_min

    tx = float(u_start) - float(local_min[0]) + 0.5 * (float(box_size) - float(local_size[0]))
    if align_bottom:
        ty = float(v_start) - float(local_min[1])
    else:
        ty = float(v_start) - float(local_min[1]) + 0.5 * (float(box_size) - float(local_size[1]))

    out = uv_local.copy()
    out[:, 0] += tx
    out[:, 1] += ty
    return out.astype(np.float32, copy=False)


def _load_part_mesh(model_dir: str, part_name: str):
    obj_path = os.path.join(model_dir, PART_FILES[part_name])
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Missing OBJ: {obj_path}")

    verts, uvs, _, v_faces, uv_faces, _ = load_uv_obj_file(obj_path, triangulate=True)
    if uvs is None:
        raise ValueError(f"OBJ has no UVs: {obj_path}")
    if verts is None or len(verts) == 0:
        raise ValueError(f"OBJ has no vertices: {obj_path}")

    # This pipeline expects 1:1 vertex/uv indexing for mesh parts.
    if len(verts) != len(uvs):
        raise ValueError(
            f"Vertex/UV count mismatch in {obj_path}: verts={len(verts)}, uvs={len(uvs)}"
        )

    if uv_faces is not None and uv_faces.shape == v_faces.shape:
        if not np.array_equal(v_faces, uv_faces):
            print(
                f"[Warn] {part_name}: vertex-face and uv-face indices differ; using vertex indices."
            )

    return np.asarray(uvs[:, :2], dtype=np.float32), np.asarray(v_faces, dtype=np.int32)


def load_combined_uv_and_triangles(model_dir: str):
    order = ["head", "eye_l", "eye_r", "mouth"]

    uv_parts = {}
    face_parts = {}
    for part in order:
        uv_parts[part], face_parts[part] = _load_part_mesh(model_dir, part)

    # Apply same combined-layout transforms as dataset/training.
    uv_parts["head"] = _transform_uv_center(uv_parts["head"], HEAD_SCALE, HEAD_TX, HEAD_TY)
    uv_parts["eye_l"] = _place_uv_in_box(
        uv_parts["eye_l"],
        u_start=EYE_L_U_START,
        v_start=BOTTOM_MARGIN,
        box_size=EYE_BOX_SIZE,
        align_bottom=False,
    )
    uv_parts["eye_r"] = _place_uv_in_box(
        uv_parts["eye_r"],
        u_start=EYE_R_U_START,
        v_start=BOTTOM_MARGIN,
        box_size=EYE_BOX_SIZE,
        align_bottom=False,
    )

    mouth_src = uv_parts["mouth"].copy()
    mouth_out = mouth_src.copy()
    mouth_high = mouth_src[:, 1] > float(MOUTH_SPLIT_V)
    mouth_low = ~mouth_high
    if np.any(mouth_high):
        mouth_out[mouth_high] = _place_uv_in_box(
            mouth_src[mouth_high],
            u_start=MOUTH_V_GT_U_START,
            v_start=BOTTOM_MARGIN,
            box_size=MOUTH_BOX_SIZE,
            align_bottom=True,
        )
    if np.any(mouth_low):
        mouth_out[mouth_low] = _place_uv_in_box(
            mouth_src[mouth_low],
            u_start=MOUTH_V_LE_U_START,
            v_start=BOTTOM_MARGIN,
            box_size=MOUTH_BOX_SIZE,
            align_bottom=True,
        )
    uv_parts["mouth"] = mouth_out

    uv_all = []
    tri_all = []
    offset = 0
    part_ranges = {}

    for part in order:
        uv = np.clip(uv_parts[part], 0.0, 1.0)
        tri = face_parts[part] + int(offset)

        part_ranges[part] = (int(offset), int(offset + len(uv)))
        uv_all.append(uv)
        tri_all.append(tri)
        offset += len(uv)

    uv_all = np.concatenate(uv_all, axis=0).astype(np.float32, copy=False)
    tri_all = np.concatenate(tri_all, axis=0).astype(np.int32, copy=False)
    return uv_all, tri_all, part_ranges


def render_random_vertex_colors(uv: np.ndarray, tri: np.ndarray, tex_size: int, seed: int, device: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvdiffrast RasterizeCudaContext.")

    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"Use a CUDA device, got: {device}")

    rng = np.random.default_rng(seed)
    vtx_color = rng.random((uv.shape[0], 3), dtype=np.float32)

    # UV -> clip space. Keep v=1 at top to match your texture convention.
    clip_xy = np.empty_like(uv, dtype=np.float32)
    clip_xy[:, 0] = uv[:, 0] * 2.0 - 1.0
    # Match dataset texture convention: image y = (1 - v)
    clip_xy[:, 1] = 1.0 - (uv[:, 1] * 2.0)

    clip_pos = np.concatenate(
        [
            clip_xy,
            np.zeros((uv.shape[0], 1), dtype=np.float32),
            np.ones((uv.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )

    pos_t = torch.from_numpy(clip_pos).to(dev)[None, ...]  # [1, V, 4]
    tri_t = torch.from_numpy(tri).to(device=dev, dtype=torch.int32)
    col_t = torch.from_numpy(vtx_color).to(dev)[None, ...]  # [1, V, 3]

    ctx = dr.RasterizeCudaContext(device=dev)
    rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[tex_size, tex_size])

    tex, _ = dr.interpolate(col_t, rast, tri_t)  # [1, H, W, 3]
    tex_aa = dr.antialias(tex, rast, pos_t, tri_t)

    cov = (rast[..., 3:] > 0).float()  # [1, H, W, 1]
    tex = tex_aa * cov

    tex_np = tex[0].detach().cpu().numpy()
    cov_np = cov[0, ..., 0].detach().cpu().numpy()
    return tex_np, cov_np, vtx_color


def _save_images(out_dir: str, tex: np.ndarray, cov: np.ndarray, uv: np.ndarray, vtx_color: np.ndarray, tex_size: int):
    os.makedirs(out_dir, exist_ok=True)

    tex_u8 = np.clip(tex * 255.0, 0.0, 255.0).astype(np.uint8)
    cov_u8 = np.clip(cov * 255.0, 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(
        os.path.join(out_dir, "nvdiffrast_random_vertex_color_texture.png"),
        cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(out_dir, "nvdiffrast_random_vertex_color_coverage.png"),
        cov_u8,
    )

    # Quick UV point debug with the same random colors as vertices.
    uv_map = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    x = np.clip(np.rint(uv[:, 0] * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    y = np.clip(np.rint((1.0 - uv[:, 1]) * (tex_size - 1)).astype(np.int32), 0, tex_size - 1)
    color_u8 = np.clip(vtx_color * 255.0, 0.0, 255.0).astype(np.uint8)
    for px, py, c in zip(x, y, color_u8):
        uv_map[py, px] = c

    cv2.imwrite(
        os.path.join(out_dir, "nvdiffrast_random_vertex_color_uv_points.png"),
        cv2.cvtColor(uv_map, cv2.COLOR_RGB2BGR),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test nvdiffrast UV rasterization by rendering random per-vertex colors."
    )
    parser.add_argument("--model_dir", type=str, default="assets/topology", help="Directory containing mesh_* OBJ files.")
    parser.add_argument("--out_dir", type=str, default="training_samples", help="Output directory.")
    parser.add_argument("--tex_size", type=int, default=1024, help="Output texture resolution.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for vertex colors.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g. cuda:0")
    args = parser.parse_args()

    uv, tri, part_ranges = load_combined_uv_and_triangles(args.model_dir)
    print(f"Loaded combined UV mesh: vertices={uv.shape[0]}, triangles={tri.shape[0]}")
    for k, (s, e) in part_ranges.items():
        print(f"  {k}: [{s}, {e})  count={e - s}")

    tex, cov, vtx_color = render_random_vertex_colors(
        uv=uv,
        tri=tri,
        tex_size=int(args.tex_size),
        seed=int(args.seed),
        device=args.device,
    )

    _save_images(args.out_dir, tex, cov, uv, vtx_color, int(args.tex_size))
    print(f"Saved outputs to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
