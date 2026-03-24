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

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import glob

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
from data_utils.texture_pack import TexturePackHelper
from typing import Optional


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
        tri = remap_triangle_faces_after_vertex_filter(tri, mesh_indices, original_vertex_count=original_vertex_count)
    return uv.astype(np.float32, copy=False), tri.astype(np.int32, copy=False)


def render_image_from_gt(mesh_xy_norm, texture_rgb, uv_coords, tri, out_h, out_w, device="cuda:0", flip_uv_v=True):
    dev = torch.device(device)
    xy = np.clip(mesh_xy_norm.astype(np.float32), 0.0, 1.0)
    clip_pos = np.zeros((xy.shape[0], 4), dtype=np.float32)
    clip_pos[:, 0] = xy[:, 0] * 2.0 - 1.0
    clip_pos[:, 1] = 1.0 - (xy[:, 1] * 2.0)
    clip_pos[:, 3] = 1.0
    pos_t = torch.from_numpy(clip_pos).to(dev)[None, ...].contiguous()
    tri_t = torch.from_numpy(tri).to(device=dev, dtype=torch.int32).contiguous()
    uv_t = torch.from_numpy(np.clip(uv_coords, 0.0, 1.0)).to(dev)[None, ...].contiguous()
    tex_t = torch.from_numpy(np.clip(texture_rgb, 0.0, 1.0)).to(dev)[None, ...].contiguous()
    ctx = dr.RasterizeCudaContext(device=dev)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[int(out_h), int(out_w)])
        uv_pix, _ = dr.interpolate(uv_t.float(), rast, tri_t)
        if flip_uv_v:
            uv_pix = torch.stack([uv_pix[..., 0], 1.0 - uv_pix[..., 1]], dim=-1)
        uv_pix = uv_pix.clamp(0.0, 1.0)
        color = dr.texture(tex_t.float(), uv_pix, filter_mode="linear", boundary_mode="clamp")
        color = dr.antialias(color, rast, pos_t, tri_t)
        cov = (rast[..., 3:4] > 0).float()
        color = color * cov
    return color[0].detach().cpu().numpy(), cov[0, ..., 0].detach().cpu().numpy()


def load_mesh(filepath: str, mesh_indices) -> Optional[np.ndarray]:
    try:
        geom = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 5:
                    geom.append([float(x) for x in parts[:5]])
        geom = np.array(geom, dtype=np.float32)
        if geom.shape[0] == 0:
            return None
        xyz = geom[:, 0:3]
        center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
        scale = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2
        if scale > 0:
            geom[:, 0:3] = (xyz - center) / scale
        geom[:, 3:5] = geom[:, 3:5] / 1024.0
        if mesh_indices is not None and mesh_indices.max() < len(geom):
            geom = geom[mesh_indices]
        return geom
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Quick test: render GT mesh with GT texture (no full dataset scan)")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--texture_root", type=str, default="G:/textures", help="Texture root")
    parser.add_argument("--max_samples", type=int, default=8, help="Max samples to render")
    parser.add_argument("--start_idx", type=int, default=0, help="Start from which Color file")
    parser.add_argument("--model_dir", type=str, default="assets/topology", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="artifacts/debug/quick_test", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using texture_root: {args.texture_root}")
    print(f"Loading UV/triangle faces...")
    
    mesh_indices = np.load(os.path.join(args.model_dir, "mesh_indices.npy")) if os.path.exists(os.path.join(args.model_dir, "mesh_indices.npy")) else None
    uv, tri = load_uv_faces_for_dataset(model_dir=args.model_dir)
    print(f"UV verts={uv.shape[0]}, tris={tri.shape[0]}")
    
    texture_packer = TexturePackHelper(texture_root=args.texture_root, texture_png_cache_max_items=0, combined_texture_cache_max_items=0)
    
    print(f"Finding Color files in {args.data_root}...")
    color_files = sorted(glob.glob(os.path.join(args.data_root, "Color_*")))
    print(f"Found {len(color_files)} Color files")
    
    saved = 0
    idx = args.start_idx
    while idx < len(color_files) and saved < args.max_samples:
        color_file = color_files[idx]
        print(f"\n[{idx}] Checking {os.path.basename(color_file)}...")
        idx += 1
        
        filename = os.path.basename(color_file)
        if not filename.startswith("Color_"):
            continue
        sample_id = os.path.splitext(filename[6:])[0]
        for s in ['_gemini', '_flux', '_seedream']:
            if sample_id.endswith(s):
                sample_id = sample_id[:-len(s)]
                break
        
        mesh_file = os.path.join(args.data_root, "mesh", f"mesh_{sample_id}.txt")
        if not os.path.exists(mesh_file):
            mesh_file = os.path.join(args.data_root, f"mesh_{sample_id}.txt")
        if not os.path.exists(mesh_file):
            print(f"  No mesh file, skipping")
            continue
        
        print(f"  Loading mesh...")
        mesh = load_mesh(mesh_file, mesh_indices)
        if mesh is None or mesh.shape[0] != uv.shape[0]:
            print(f"  Mesh invalid, skipping")
            continue
        
        print(f"  Loading texture...")
        tex = texture_packer.load_mesh_texture_map(args.data_root, sample_id)
        if tex is None:
            print(f"  Texture unavailable, skipping")
            continue
        
        print(f"  Loading RGB...")
        rgb = cv2.imread(color_file)
        if rgb is None:
            print(f"  RGB load failed, skipping")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        print(f"  Rendering...")
        mesh_xy = np.clip(mesh[:, 3:5], 0.0, 1.0)
        render, cov = render_image_from_gt(mesh_xy, tex, uv, tri, rgb.shape[0], rgb.shape[1], device="cuda:0")
        
        diff = np.abs(render - rgb)
        panel = np.concatenate([rgb, render, diff], axis=1)
        panel_u8 = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        
        out_path = os.path.join(args.out_dir, f"quick_{saved:03d}_rgb_render_diff.png")
        cv2.imwrite(out_path, cv2.cvtColor(panel_u8, cv2.COLOR_RGB2BGR))
        print(f"  ✓ Saved sample {saved}")
        saved += 1
    
    print(f"\nDone. Saved {saved} samples to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
