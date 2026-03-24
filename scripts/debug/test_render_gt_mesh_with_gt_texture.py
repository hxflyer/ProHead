

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

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

from geometry_train_core import load_template_mesh_uv, load_combined_mesh_triangle_faces, remap_triangle_faces_after_vertex_filter
from geometry_dataset import GeometryDataset
from data_utils.camera_io import load_matrix_data, compute_vertex_depth
import platform as platform_module


def load_uv_faces_for_dataset(model_dir: str = "assets/topology"):
    mesh_template = np.load(os.path.join(model_dir, "mesh_template.npy"))
    original_vertex_count = int(mesh_template.shape[0])

    uv = load_template_mesh_uv(model_dir=model_dir)
    tri = load_combined_mesh_triangle_faces(model_dir=model_dir)

    mesh_indices_path = os.path.join(model_dir, "mesh_indices.npy")
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path)
        if mesh_indices.max() < uv.shape[0]:
            uv = uv[mesh_indices]
        tri = remap_triangle_faces_after_vertex_filter(
            tri,
            mesh_indices,
            original_vertex_count=original_vertex_count,
        )

    if tri.shape[0] == 0:
        raise ValueError("No triangles after filtering mesh indices.")
    if tri.min() < 0 or tri.max() >= uv.shape[0]:
        raise ValueError(f"Triangle index out of range: tri max={tri.max()}, uv count={uv.shape[0]}")

    return uv.astype(np.float32, copy=False), tri.astype(np.int32, copy=False)


def render_image_from_gt(
    mesh_xy_norm: np.ndarray,
    texture_rgb: np.ndarray,
    uv_coords: np.ndarray,
    tri: np.ndarray,
    out_h: int,
    out_w: int,
    mesh_depth: np.ndarray = None,
    device: str = "cuda:0",
    flip_uv_v: bool = True,
):
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"nvdiffrast render requires CUDA device, got: {device}")

    if mesh_xy_norm.ndim != 2 or mesh_xy_norm.shape[1] != 2:
        raise ValueError(f"mesh_xy_norm must be [V,2], got {mesh_xy_norm.shape}")
    if texture_rgb.ndim != 3 or texture_rgb.shape[2] != 3:
        raise ValueError(f"texture_rgb must be [H,W,3], got {texture_rgb.shape}")
    if uv_coords.shape[0] != mesh_xy_norm.shape[0]:
        raise ValueError(f"UV vertex count mismatch: uv={uv_coords.shape[0]} mesh={mesh_xy_norm.shape[0]}")

    xy = np.clip(mesh_xy_norm.astype(np.float32), 0.0, 1.0)
    clip_pos = np.zeros((xy.shape[0], 4), dtype=np.float32)
    clip_pos[:, 0] = xy[:, 0] * 2.0 - 1.0
    clip_pos[:, 1] = 1.0 - (xy[:, 1] * 2.0)
    # Use mesh_depth if available for proper z-ordering (negate for correct order)
    if mesh_depth is not None and mesh_depth.shape[0] == xy.shape[0]:
        clip_pos[:, 2] = -mesh_depth.astype(np.float32)
    else:
        clip_pos[:, 2] = 0.0
    clip_pos[:, 3] = 1.0

    pos_t = torch.from_numpy(clip_pos).to(dev)[None, ...].contiguous()  # [1, V, 4]
    tri_t = torch.from_numpy(tri).to(device=dev, dtype=torch.int32).contiguous()
    uv_t = torch.from_numpy(np.clip(uv_coords, 0.0, 1.0)).to(dev)[None, ...].contiguous()  # [1, V, 2]
    tex_t = torch.from_numpy(np.clip(texture_rgb, 0.0, 1.0)).to(dev)[None, ...].contiguous()  # [1, Ht, Wt, 3]

    ctx = dr.RasterizeCudaContext(device=dev)

    # Render at 2x resolution for anti-aliasing, then downscale
    render_h = int(out_h) * 2
    render_w = int(out_w) * 2

    with torch.amp.autocast(device_type="cuda", enabled=False):
        rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[render_h, render_w])
        uv_pix, _ = dr.interpolate(uv_t.float(), rast, tri_t)
        if flip_uv_v:
            uv_pix = torch.stack([uv_pix[..., 0], 1.0 - uv_pix[..., 1]], dim=-1)
        uv_pix = uv_pix.clamp(0.0, 1.0)

        color = dr.texture(tex_t.float(), uv_pix, filter_mode="linear", boundary_mode="clamp")
        color = dr.antialias(color, rast, pos_t, tri_t)
        cov = (rast[..., 3:4] > 0).float()
        color = color * cov

    # Convert to NCHW format for downsampling
    color = color.permute(0, 3, 1, 2)  # [1, 3, H*2, W*2]
    cov = cov.permute(0, 3, 1, 2)  # [1, 1, H*2, W*2]
    
    # Downsample from 2x to target resolution for anti-aliasing
    import torch.nn.functional as F
    color = F.interpolate(color, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
    cov = F.interpolate(cov, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
    
    render = color[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
    coverage = cov[0, 0].detach().cpu().numpy()  # [H,W]
    
    # Flip vertically to correct coordinate system
    render = np.flip(render, axis=0).copy()
    coverage = np.flip(coverage, axis=0).copy()
    
    return render.astype(np.float32), coverage.astype(np.float32)


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8)


def _default_texture_root() -> str:
    """Default texture root based on platform."""
    system = platform_module.system().lower()
    if system == "linux":
        return "/hy-tmp/textures"
    else:
        return "G:/textures"


def main():
    parser = argparse.ArgumentParser(description="Render GT 2D mesh with GT texture using nvdiffrast.")
    parser.add_argument("--data_roots", nargs="+", required=True, help="Dataset root(s).")
    parser.add_argument("--texture_root", type=str, default="", help="Texture root override (default: G:/textures on Windows, /hy-tmp/textures on Linux).")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split.")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Train/val split ratio.")
    parser.add_argument("--image_size", type=int, default=512, help="Dataset image size.")
    parser.add_argument("--max_samples", type=int, default=8, help="Number of rendered samples to save.")
    parser.add_argument("--start_idx", type=int, default=0, help="Start scanning dataset index.")
    parser.add_argument("--model_dir", type=str, default="assets/topology", help="Model dir with template/obj/index files.")
    parser.add_argument("--out_dir", type=str, default="artifacts/debug/gt_render_debug", help="Output directory.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this nvdiffrast test.")

    os.makedirs(args.out_dir, exist_ok=True)

    texture_root = args.texture_root.strip() if args.texture_root else _default_texture_root()
    print(f"Using texture_root: {texture_root}")
    print(f"Creating dataset...")

    dataset = GeometryDataset(
        data_roots=args.data_roots,
        split=args.split,
        image_size=int(args.image_size),
        train_ratio=float(args.train_ratio),
        augment=False,
        texture_root=texture_root,
        texture_png_cache_max_items=0,
        combined_texture_cache_max_items=0,
    )

    print(f"Dataset created successfully")
    print(f"Loading UV/triangle faces from {args.model_dir}...")
    uv, tri = load_uv_faces_for_dataset(model_dir=args.model_dir)
    print(f"Loaded template UV/faces: verts={uv.shape[0]}, tris={tri.shape[0]}")
    print(f"Dataset size ({args.split}): {len(dataset)}")
    print(f"Starting to iterate through dataset...")

    saved = 0
    idx = max(0, int(args.start_idx))
    while idx < len(dataset) and saved < int(args.max_samples):
        print(f"\n[{idx}] Loading sample from dataset...")
        sample = dataset[idx]
        print(f"[{idx}] Sample loaded, checking validity...")
        idx += 1

        has_geo = float(sample.get("has_geometry_gt", torch.tensor(0.0)).item()) > 0.5
        tex_valid = float(sample.get("mesh_texture_valid", torch.tensor(0.0)).item()) > 0.5
        print(f"[{idx-1}] has_geometry={has_geo}, texture_valid={tex_valid}")
        if (not has_geo) or (not tex_valid):
            print(f"[{idx-1}] Skipping (missing geometry or texture)")
            continue

        mesh = sample["mesh"].cpu().numpy()  # [V,5]
        print(f"[{idx-1}] Mesh shape: {mesh.shape}, UV shape: {uv.shape}")
        if mesh.shape[0] != uv.shape[0]:
            print(f"[Skip] vertex mismatch at idx={idx-1}: mesh={mesh.shape[0]} uv={uv.shape[0]}")
            continue

        print(f"[{idx-1}] Preparing data for rendering...")
        mesh_xy = np.clip(mesh[:, 3:5], 0.0, 1.0)
        tex = sample["mesh_texture"].permute(1, 2, 0).cpu().numpy().astype(np.float32)

        rgb = sample["rgb"].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        basecolor = sample["basecolor"].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        basecolor_valid = float(sample.get("basecolor_valid", torch.tensor(0.0)).item()) > 0.5

        # Get mesh_depth from sample if available
        mesh_depth_np = None
        if "mesh_depth" in sample:
            mesh_depth_np = sample["mesh_depth"].cpu().numpy()
            print(f"[{idx-1}] Using mesh depth for proper z-ordering")

        print(f"[{idx-1}] Rendering with nvdiffrast...")
        render, cov = render_image_from_gt(
            mesh_xy_norm=mesh_xy,
            texture_rgb=tex,
            uv_coords=uv,
            tri=tri,
            out_h=rgb.shape[0],
            out_w=rgb.shape[1],
            mesh_depth=mesh_depth_np,
            device=args.device,
            flip_uv_v=True,
        )

        if basecolor_valid:
            diff = np.abs(render - basecolor)
            panel = np.concatenate([rgb, basecolor, render, diff], axis=1)
            panel_name = f"sample_{saved:03d}_rgb_basecolor_render_diff.png"
        else:
            diff = np.abs(render - rgb)
            panel = np.concatenate([rgb, render, diff], axis=1)
            panel_name = f"sample_{saved:03d}_rgb_render_diff.png"

        cov_u8 = to_u8(cov[..., None].repeat(3, axis=2))
        render_u8 = to_u8(render)
        panel_u8 = to_u8(panel)

        print(f"[{idx-1}] Saving images to {args.out_dir}...")
        cv2.imwrite(os.path.join(args.out_dir, panel_name), cv2.cvtColor(panel_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out_dir, f"sample_{saved:03d}_render.png"), cv2.cvtColor(render_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out_dir, f"sample_{saved:03d}_coverage.png"), cv2.cvtColor(cov_u8, cv2.COLOR_RGB2BGR))

        print(f"✓ Saved sample {saved}: dataset_idx={idx-1}, has_basecolor={basecolor_valid}")
        saved += 1

    print(f"Done. Saved {saved} samples to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
