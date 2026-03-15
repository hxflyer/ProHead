from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from render_head import (  # noqa: E402
    DifferentiableHeadRenderer,
    DifferentiableRendererConfig,
    default_sampling_cache_path,
    matrix_data_to_torch,
    stack_mat_tensors,
)
from mat_load_helper import load_matrix_data  # noqa: E402
from obj_load_helper import load_uv_obj_file  # noqa: E402
from tex_pack_helper import TexturePackHelper  # noqa: E402
from train_visualize_helper import load_combined_mesh_uv  # noqa: E402


def default_texture_root() -> str:
    return "G:/textures" if os.name == "nt" else "/hy-tmp/textures"


def save_rgb(path: Path, image_rgb: np.ndarray | torch.Tensor) -> None:
    if torch.is_tensor(image_rgb):
        array = image_rgb.detach().float().cpu()
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = array.permute(1, 2, 0)
        image_np = array.numpy()
    else:
        image_np = np.asarray(image_rgb, dtype=np.float32)

    image_np = np.clip(image_np, 0.0, 1.0)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("image_rgb must have shape [H, W, 3] or [3, H, W]")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((image_np * 255.0).astype(np.uint8), mode="RGB").save(path)


def load_geometry_txt(filepath: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"No mesh rows found in {filepath}")
    return np.asarray(rows, dtype=np.float32)


def discover_sample_ids(data_root: Path, limit: int) -> list[str]:
    mesh_dir = data_root / "mesh"
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Missing mesh directory: {mesh_dir}")
    sample_ids = []
    for mesh_path in sorted(mesh_dir.glob("mesh_*.txt")):
        sample_ids.append(mesh_path.stem[len("mesh_") :])
    if not sample_ids:
        raise FileNotFoundError(f"No mesh_*.txt files found under {mesh_dir}")
    return sample_ids[: max(1, int(limit))]


def resolve_sample_paths(data_root: Path, sample_id: str) -> tuple[Path, Path]:
    mesh_path = data_root / "mesh" / f"mesh_{sample_id}.txt"
    mats_path = data_root / "mat" / f"Mats_{sample_id}.txt"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh txt not found for sample '{sample_id}' under {data_root}")
    if not mats_path.exists():
        raise FileNotFoundError(f"Mats txt not found for sample '{sample_id}' under {data_root}")
    return mesh_path, mats_path


def load_combined_mesh_triangle_faces(model_dir: Path) -> np.ndarray:
    part_files = ["mesh_head.obj", "mesh_eye_l.obj", "mesh_eye_r.obj", "mesh_mouth.obj"]
    tris = []
    offset = 0
    for file_name in part_files:
        obj_path = model_dir / file_name
        if not obj_path.exists():
            raise FileNotFoundError(f"Missing mesh OBJ for face loading: {obj_path}")
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(str(obj_path), triangulate=True)
        if verts is None or uvs is None or v_faces is None:
            raise ValueError(f"Failed to load verts/uv/faces from {obj_path}")
        if len(verts) != len(uvs):
            raise ValueError(f"Vertex/UV count mismatch in {obj_path}: verts={len(verts)}, uvs={len(uvs)}")
        tris.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))
    return np.concatenate(tris, axis=0).astype(np.int32, copy=False)


def remap_triangle_faces_after_vertex_filter(
    triangle_faces: np.ndarray,
    kept_vertex_indices: np.ndarray,
    original_vertex_count: int,
) -> np.ndarray:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)
    tri_new = remap[tri]
    tri_new = tri_new[np.all(tri_new >= 0, axis=1)]
    return tri_new.astype(np.int32, copy=False)


def load_mesh_uv_and_faces(model_dir: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    uv_full = load_combined_mesh_uv(model_dir=str(model_dir), copy=True).astype(np.float32, copy=False)
    faces_full = load_combined_mesh_triangle_faces(model_dir=model_dir)

    if uv_full.shape[0] == vertex_count:
        return uv_full, faces_full

    mesh_indices_path = model_dir / "mesh_indices.npy"
    if mesh_indices_path.exists():
        mesh_indices = np.load(str(mesh_indices_path))
        if mesh_indices.shape[0] == vertex_count and int(mesh_indices.max()) < int(uv_full.shape[0]):
            uv = uv_full[mesh_indices]
            faces = remap_triangle_faces_after_vertex_filter(
                faces_full,
                mesh_indices,
                original_vertex_count=int(uv_full.shape[0]),
            )
            return uv.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)

    raise ValueError(f"Could not match static UV layout to mesh vertex count {vertex_count}")


def load_packed_texture(data_root: Path, sample_id: str, texture_root: str) -> np.ndarray:
    helper = TexturePackHelper(texture_root=texture_root)
    texture = helper.load_mesh_texture_map(str(data_root), sample_id)
    if texture is None:
        raise FileNotFoundError(
            f"Could not build packed texture for sample '{sample_id}'. Checked texture root: {texture_root}"
        )
    return texture.astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a few dataset samples, convert them to torch, and render them.")
    parser.add_argument("--data-root", default=r"G:\CapturedFrames_final8_processed", help="Dataset root containing mesh/ and mat/")
    parser.add_argument("--sample-ids", default="", help="Comma-separated sample ids. Empty means auto-discover.")
    parser.add_argument("--num-samples", type=int, default=3, help="How many samples to auto-discover when --sample-ids is empty")
    parser.add_argument("--batch-size", type=int, default=2, help="How many samples to render per forward pass")
    parser.add_argument("--texture-root", default=default_texture_root(), help="Texture root used by TexturePackHelper")
    parser.add_argument("--model-dir", default="model", help="Model directory with static UV topology assets")
    parser.add_argument("--output-dir", default="gsplat/dataset_output_training_demo", help="Directory for saved renders")
    parser.add_argument("--width", type=int, default=512, help="Output render width")
    parser.add_argument("--height", type=int, default=512, help="Output render height")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument("--sampling-cache-path", default="", help="Optional NPZ path for cached face sampling plan")
    parser.add_argument("--rebuild-sampling-cache", action="store_true", help="Force rebuilding the saved face sampling cache")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    model_dir = (REPO_ROOT / args.model_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir / data_root.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.sample_ids.strip():
        sample_ids = [sample_id.strip() for sample_id in args.sample_ids.split(",") if sample_id.strip()]
    else:
        sample_ids = discover_sample_ids(data_root, limit=int(args.num_samples))
    if not sample_ids:
        raise ValueError("No sample ids selected for rendering")

    reference_mesh_path, _ = resolve_sample_paths(data_root, sample_ids[0])
    reference_vertices = load_geometry_txt(reference_mesh_path)
    uvs, faces = load_mesh_uv_and_faces(model_dir=model_dir, vertex_count=reference_vertices.shape[0])

    cache_path = (
        Path(args.sampling_cache_path).resolve()
        if args.sampling_cache_path
        else default_sampling_cache_path(
            cache_root=(REPO_ROOT / "gsplat" / "cache").resolve(),
            cache_stem=f"{data_root.name}_combined_mesh",
            vertex_count=int(reference_vertices.shape[0]),
            face_count=int(faces.shape[0]),
            target_area_percentile=65.0,
            max_subdivision_level=5,
        )
    )

    config = DifferentiableRendererConfig(
        image_width=int(args.width),
        image_height=int(args.height),
    )
    renderer = DifferentiableHeadRenderer(
        reference_vertices=reference_vertices,
        uvs=uvs,
        faces=faces,
        uv_faces=faces,
        config=config,
        sampling_cache_path=cache_path,
        rebuild_sampling_cache=bool(args.rebuild_sampling_cache),
    ).to(device)
    renderer.eval()

    records: list[dict[str, object]] = []
    for start in range(0, len(sample_ids), max(1, int(args.batch_size))):
        batch_sample_ids = sample_ids[start : start + max(1, int(args.batch_size))]
        batch_mats = []
        batch_meshes = []
        batch_textures = []
        batch_meta = []

        for sample_id in batch_sample_ids:
            mesh_path, mats_path = resolve_sample_paths(data_root, sample_id)
            vertices_local = load_geometry_txt(mesh_path)
            if vertices_local.shape[0] != reference_vertices.shape[0]:
                raise ValueError(
                    f"Sample '{sample_id}' vertex count {vertices_local.shape[0]} does not match reference {reference_vertices.shape[0]}"
                )

            matrix_data = load_matrix_data(str(mats_path))
            texture = load_packed_texture(data_root=data_root, sample_id=sample_id, texture_root=args.texture_root)

            batch_mats.append(matrix_data_to_torch(matrix_data, device=device))
            batch_meshes.append(torch.from_numpy(vertices_local).to(device=device, dtype=torch.float32))
            batch_textures.append(torch.from_numpy(texture).permute(2, 0, 1).to(device=device, dtype=torch.float32))
            batch_meta.append({"sample_id": sample_id, "mesh_path": str(mesh_path), "mats_path": str(mats_path)})

        mat_batch = stack_mat_tensors(batch_mats)
        mesh_batch = torch.stack(batch_meshes, dim=0)
        texture_batch = torch.stack(batch_textures, dim=0)

        with torch.no_grad():
            render_batch = renderer(mat_batch, mesh_batch, texture_batch)

        for idx, meta in enumerate(batch_meta):
            sample_id = str(meta["sample_id"])
            image = render_batch[idx]
            size_tag = f"{config.image_width}x{config.image_height}"
            render_path = output_dir / f"{sample_id}_gaussian_render_{size_tag}.png"
            stats_path = output_dir / f"{sample_id}_gaussian_render_{size_tag}.json"
            save_rgb(render_path, image)

            image_cpu = image.detach().float().cpu()
            stats = {
                "sample_id": sample_id,
                "mesh_path": meta["mesh_path"],
                "mats_path": meta["mats_path"],
                "output_path": str(render_path),
                "image_width": int(config.image_width),
                "image_height": int(config.image_height),
                "render_mean": float(image_cpu.mean().item()),
                "render_std": float(image_cpu.std().item()),
                "render_min": float(image_cpu.min().item()),
                "render_max": float(image_cpu.max().item()),
                "non_empty": bool(float(image_cpu.std().item()) > 1.0e-5),
                "sampling_cache_status": renderer.sampling_cache_status,
                "sampling_cache_path": renderer.sampling_cache_path,
            }
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            records.append(stats)
            print(f"{sample_id}: saved {render_path}")

    summary_path = output_dir / "render_summary.json"
    summary_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Rendered {len(records)} sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
