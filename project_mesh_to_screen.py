from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from mat_load_helper import compute_vertex_depth, load_matrix_data, project_3d_to_2d_cpp_exact
from tex_pack_helper import TexturePackHelper
from train_visualize_helper import load_combined_mesh_uv


def load_geometry_like_dataset(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror FastGeometryDataset._load_geometry while preserving raw values for projection."""
    geom_rows: list[list[float]] = []
    valid_screen_rows: list[bool] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()

            if len(parts) >= 5:
                geom_rows.append([float(x) for x in parts[:5]])
                valid_screen_rows.append(True)
            elif len(parts) >= 3:
                geom_rows.append([float(parts[0]), float(parts[1]), float(parts[2]), 0.0, 0.0])
                valid_screen_rows.append(False)

    if not geom_rows:
        raise ValueError(f"No geometry rows found in {filepath}")

    geom_raw = np.asarray(geom_rows, dtype=np.float32)
    raw_xyz = geom_raw[:, 0:3].copy()
    raw_screen = geom_raw[:, 3:5].copy()
    raw_screen_valid = np.asarray(valid_screen_rows, dtype=bool)
    return geom_raw, raw_xyz, raw_screen, raw_screen_valid


def _find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_mesh_path(data_root: Path, sample_id: str) -> Path:
    path = _find_first_existing(
        [
            data_root / "mesh" / f"mesh_{sample_id}.txt",
            data_root / "mesh" / f"{sample_id}.txt",
        ]
    )
    if path is None:
        raise FileNotFoundError(f"Could not find mesh txt for sample '{sample_id}' under {data_root}")
    return path


def resolve_mat_path(data_root: Path, sample_id: str) -> Path:
    path = _find_first_existing(
        [
            data_root / "mat" / f"Mats_{sample_id}.txt",
            data_root / f"Mats_{sample_id}.txt",
        ]
    )
    if path is None:
        raise FileNotFoundError(f"Could not find Mats txt for sample '{sample_id}' under {data_root}")
    return path


def resolve_color_path(data_root: Path, sample_id: str) -> Optional[Path]:
    candidates = []
    for ext in [".png", ".jpg", ".jpeg"]:
        candidates.append(data_root / f"Color_{sample_id}{ext}")
        for suffix in ["_gemini", "_flux", "_seedream"]:
            candidates.append(data_root / f"Color_{sample_id}{suffix}{ext}")
    return _find_first_existing(candidates)


def default_texture_root() -> str:
    return "G:/textures" if os.name == "nt" else "/hy-tmp/textures"


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    if depth.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max - depth_min < 1e-8:
        depth_norm = np.zeros_like(depth, dtype=np.float32)
    else:
        depth_norm = ((depth - depth_min) / (depth_max - depth_min)).astype(np.float32)

    colors = np.zeros((depth.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip(255.0 * depth_norm, 0.0, 255.0).astype(np.uint8)
    colors[:, 1] = np.clip(255.0 * (1.0 - np.abs(depth_norm - 0.5) * 2.0), 0.0, 255.0).astype(np.uint8)
    colors[:, 2] = np.clip(255.0 * (1.0 - depth_norm), 0.0, 255.0).astype(np.uint8)
    return colors


def sample_texture_bilinear(texture_rgb: np.ndarray, uv_coords: np.ndarray) -> np.ndarray:
    h, w = texture_rgb.shape[:2]
    uv = np.clip(np.asarray(uv_coords, dtype=np.float32), 0.0, 1.0)
    x = uv[:, 0] * float(max(w - 1, 1))
    y = (1.0 - uv[:, 1]) * float(max(h - 1, 1))

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    tx = (x - x0).astype(np.float32)[:, None]
    ty = (y - y0).astype(np.float32)[:, None]

    c00 = texture_rgb[y0, x0]
    c10 = texture_rgb[y0, x1]
    c01 = texture_rgb[y1, x0]
    c11 = texture_rgb[y1, x1]

    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return (c0 * (1.0 - ty) + c1 * ty).astype(np.float32)


def load_vertex_texture_colors(
    vertex_count: int,
    data_root: Optional[Path],
    sample_id: str,
    texture_root: Optional[str],
    texture_image: Optional[Path],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    packed_texture = None
    texture_source = None

    if texture_image is not None and texture_image.exists():
        tex_bgr = cv2.imread(str(texture_image), cv2.IMREAD_COLOR)
        if tex_bgr is not None:
            packed_texture = cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            texture_source = str(texture_image)
    elif data_root is not None:
        helper = TexturePackHelper(texture_root=texture_root or default_texture_root())
        packed_texture = helper.load_mesh_texture_map(str(data_root), sample_id)
        if packed_texture is not None:
            texture_source = f"packed:{helper.get_texture_root(str(data_root))}"

    if packed_texture is None:
        return None, None, None

    uv = load_combined_mesh_uv(model_dir="model", copy=True).astype(np.float32, copy=False)
    if uv.shape[0] != vertex_count:
        mesh_indices_path = Path("model/mesh_indices.npy")
        if mesh_indices_path.exists():
            mesh_indices = np.load(str(mesh_indices_path))
            if mesh_indices.shape[0] == vertex_count and int(mesh_indices.max()) < int(uv.shape[0]):
                uv = uv[mesh_indices]

    if uv.shape[0] != vertex_count:
        return packed_texture, None, texture_source

    sampled = sample_texture_bilinear(packed_texture, uv)
    sampled_uint8 = np.clip(sampled * 255.0, 0.0, 255.0).astype(np.uint8)
    return packed_texture, sampled_uint8, texture_source


def draw_projected_points(
    image_rgb: Optional[np.ndarray],
    projected_xy: np.ndarray,
    width: int,
    height: int,
    depth: np.ndarray,
    point_colors: Optional[np.ndarray] = None,
    source_xy: Optional[np.ndarray] = None,
    source_mask: Optional[np.ndarray] = None,
    radius: int = 1,
    show_source_points: bool = False,
) -> np.ndarray:
    if image_rgb is None:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    else:
        if image_rgb.shape[0] != height or image_rgb.shape[1] != width:
            canvas = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            canvas = image_rgb.copy()

    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    point_colors_uint8 = point_colors if point_colors is not None else colorize_depth(depth)
    inside = (
        (projected_xy[:, 0] >= 0.0)
        & (projected_xy[:, 0] < float(width))
        & (projected_xy[:, 1] >= 0.0)
        & (projected_xy[:, 1] < float(height))
    )

    for idx in np.where(inside)[0]:
        px = int(round(float(projected_xy[idx, 0])))
        py = int(round(float(projected_xy[idx, 1])))
        bgr = tuple(int(x) for x in point_colors_uint8[idx][::-1])
        cv2.circle(canvas_bgr, (px, py), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)

    if show_source_points and source_xy is not None and source_mask is not None:
        for idx in np.where(source_mask)[0]:
            px = int(round(float(source_xy[idx, 0])))
            py = int(round(float(source_xy[idx, 1])))
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(canvas_bgr, (px, py), max(radius + 1, 2), (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def save_projected_txt(
    output_path: Path,
    xyz: np.ndarray,
    projected_xy: np.ndarray,
    depth: np.ndarray,
    source_xy: np.ndarray,
    source_mask: np.ndarray,
) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# x y z screen_x screen_y depth src_x src_y src_valid\n")
        for i in range(xyz.shape[0]):
            src_x = float(source_xy[i, 0]) if source_mask[i] else 0.0
            src_y = float(source_xy[i, 1]) if source_mask[i] else 0.0
            src_valid = 1 if source_mask[i] else 0
            f.write(
                f"{float(xyz[i, 0]):.6f} {float(xyz[i, 1]):.6f} {float(xyz[i, 2]):.6f} "
                f"{float(projected_xy[i, 0]):.6f} {float(projected_xy[i, 1]):.6f} {float(depth[i]):.6f} "
                f"{src_x:.6f} {src_y:.6f} {src_valid}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a dataset mesh txt and project its raw XYZ into screen space using Mats_*.txt."
    )
    parser.add_argument("--data-root", type=str, help="Dataset sample root containing mesh/ and mat/")
    parser.add_argument("--sample-id", type=str, help="Sample id, e.g. 0001_0002")
    parser.add_argument("--mesh-path", type=str, help="Direct path to a mesh txt file")
    parser.add_argument("--mat-path", type=str, help="Direct path to a Mats txt file")
    parser.add_argument("--image-path", type=str, help="Optional image path for overlay preview")
    parser.add_argument("--texture-root", type=str, default="", help="Texture root used by TexturePackHelper, default matches training")
    parser.add_argument("--texture-image", type=str, help="Optional direct packed texture image to sample instead of dataset lookup")
    parser.add_argument("--output-dir", type=str, default="projection_output", help="Directory for saved outputs")
    parser.add_argument("--point-radius", type=int, default=1, help="Radius for projected points in preview PNG")
    parser.add_argument("--show-source-points", action="store_true", help="Draw the original stored 2D mesh points as blue outlines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mesh_path and args.mat_path:
        mesh_path = Path(args.mesh_path).resolve()
        mat_path = Path(args.mat_path).resolve()
        sample_id = mesh_path.stem.replace("mesh_", "")
        color_path = Path(args.image_path).resolve() if args.image_path else None
        data_root = None
    elif args.data_root and args.sample_id:
        data_root = Path(args.data_root).resolve()
        sample_id = args.sample_id
        mesh_path = resolve_mesh_path(data_root, sample_id)
        mat_path = resolve_mat_path(data_root, sample_id)
        color_path = Path(args.image_path).resolve() if args.image_path else resolve_color_path(data_root, sample_id)
    else:
        raise ValueError("Provide either (--mesh-path and --mat-path) or (--data-root and --sample-id).")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geom_raw, raw_xyz, raw_screen, raw_screen_valid = load_geometry_like_dataset(str(mesh_path))
    matrix_data = load_matrix_data(str(mat_path))
    projected_xy = project_3d_to_2d_cpp_exact(raw_xyz, matrix_data).astype(np.float32)
    raw_depth = compute_vertex_depth(raw_xyz, matrix_data).astype(np.float32)

    resolution = matrix_data.get("resolution", (1024, 1024))
    width = int(resolution[0])
    height = int(resolution[1])

    image_rgb = None
    if color_path is not None and color_path.exists():
        image_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    texture_image_path = Path(args.texture_image).resolve() if args.texture_image else None
    packed_texture, vertex_texture_colors, texture_source = load_vertex_texture_colors(
        vertex_count=raw_xyz.shape[0],
        data_root=data_root if "data_root" in locals() else None,
        sample_id=sample_id,
        texture_root=args.texture_root.strip() if args.texture_root else default_texture_root(),
        texture_image=texture_image_path,
    )

    preview_rgb = draw_projected_points(
        image_rgb=image_rgb,
        projected_xy=projected_xy,
        width=width,
        height=height,
        depth=raw_depth,
        point_colors=vertex_texture_colors,
        source_xy=raw_screen if np.any(raw_screen_valid) else None,
        source_mask=raw_screen_valid if np.any(raw_screen_valid) else None,
        radius=max(1, int(args.point_radius)),
        show_source_points=bool(args.show_source_points),
    )

    inside_mask = (
        (projected_xy[:, 0] >= 0.0)
        & (projected_xy[:, 0] < float(width))
        & (projected_xy[:, 1] >= 0.0)
        & (projected_xy[:, 1] < float(height))
    )

    source_error_stats = None
    if np.any(raw_screen_valid):
        reproj_error = np.linalg.norm(projected_xy[raw_screen_valid] - raw_screen[raw_screen_valid], axis=1)
        source_error_stats = {
            "count": int(reproj_error.shape[0]),
            "mean_px": float(reproj_error.mean()),
            "median_px": float(np.median(reproj_error)),
            "max_px": float(reproj_error.max()),
        }

    out_txt = output_dir / f"{sample_id}_projected_mesh.txt"
    out_png = output_dir / f"{sample_id}_projected_mesh.png"
    out_json = output_dir / f"{sample_id}_projected_mesh_stats.json"
    out_texture_png = output_dir / f"{sample_id}_sampled_texture.png"

    save_projected_txt(out_txt, raw_xyz, projected_xy, raw_depth, raw_screen, raw_screen_valid)
    cv2.imwrite(str(out_png), cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    if packed_texture is not None:
        cv2.imwrite(str(out_texture_png), cv2.cvtColor(np.clip(packed_texture * 255.0, 0.0, 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))

    stats = {
        "mesh_path": str(mesh_path),
        "mat_path": str(mat_path),
        "image_path": str(color_path) if color_path is not None else None,
        "texture_source": texture_source,
        "texture_preview": str(out_texture_png) if packed_texture is not None else None,
        "point_color_mode": "sampled_texture" if vertex_texture_colors is not None else "depth_fallback",
        "show_source_points": bool(args.show_source_points),
        "vertex_count": int(raw_xyz.shape[0]),
        "resolution": [width, height],
        "inside_screen_count": int(inside_mask.sum()),
        "inside_screen_ratio": float(inside_mask.mean()),
        "screen_x_range": [float(projected_xy[:, 0].min()), float(projected_xy[:, 0].max())],
        "screen_y_range": [float(projected_xy[:, 1].min()), float(projected_xy[:, 1].max())],
        "depth_range": [float(raw_depth.min()), float(raw_depth.max())],
        "source_screen_error_px": source_error_stats,
        "output_txt": str(out_txt),
        "output_png": str(out_png),
    }
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Mesh: {mesh_path}")
    print(f"Mats: {mat_path}")
    if color_path is not None:
        print(f"Image: {color_path}")
    if texture_source is not None:
        print(f"Texture: {texture_source}")
    print(f"Vertices: {raw_xyz.shape[0]}")
    print(f"Inside screen: {int(inside_mask.sum())} / {raw_xyz.shape[0]} ({inside_mask.mean():.4f})")
    print(f"Saved text: {out_txt}")
    print(f"Saved preview: {out_png}")
    if source_error_stats is not None:
        print(
            "Source 2D error:"
            f" mean={source_error_stats['mean_px']:.3f}px"
            f" median={source_error_stats['median_px']:.3f}px"
            f" max={source_error_stats['max_px']:.3f}px"
        )


if __name__ == "__main__":
    main()
