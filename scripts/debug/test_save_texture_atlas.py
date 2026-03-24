"""
Save a few samples of the built texture atlas and export the combined template mesh
as OBJ + PNG texture (both albedo and normal).

Atlas layout (matching UV remapping):
  - Head:  original UVs (u=[0,1], v~[0.15,1])
  - Eye_L: u=[0, 0.20],    v=[0, 0.15]
  - Eye_R: u=[0.80, 1.0],  v=[0, 0.15]
  - Mouth: u=[0.22, 0.78], v=[0, 0.15]
"""

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


import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np

from data_utils.obj_io import load_uv_obj_file
from data_utils.texture_pack import TexturePackHelper


MODEL_DIR = "model"
OUTPUT_DIR = "test_output_atlas"
DATA_ROOTS = ["G:/CapturedFrames_final1_processed"]
TEXTURE_ROOT = "G:/textures"
NUM_SAMPLES = 5
ATLAS_SIZE = 2048

# Atlas UV regions for each part: (u_min, v_min, u_size, v_size)
# Head keeps original UVs; eye/mouth are remapped to the bottom strip
PART_REGIONS = {
    "mesh_eye_l":  (0.00, 0.00, 0.20, 0.15),
    "mesh_eye_r":  (0.80, 0.00, 0.20, 0.15),
    "mesh_mouth":  (0.22, 0.00, 0.56, 0.15),
}

# Texture file names per part
ALBEDO_FILES = {
    "head":  "T_Head_BC.png",
    "eye_l": "T_EyeL_Composite_BC.png",
    "eye_r": "T_EyeR_Composite_BC.png",
    "mouth": "T_Teeth_BC.png",
}
NORMAL_FILES = {
    "head": "T_Head_N.png",
}
TEXTURE_KEYS = {
    "head": "face",
    "eye_l": "eye_l",
    "eye_r": "eye_r",
    "mouth": "mouth_default",
}


def load_and_merge_mesh_objs(model_dir: str) -> tuple:
    """Load 4 mesh OBJ parts, remap eye/mouth UVs to atlas regions, merge into one."""
    all_verts = []
    all_uvs = []
    all_faces_v = []
    all_faces_uv = []
    v_offset = 0
    uv_offset = 0

    for part in ["mesh_head", "mesh_eye_l", "mesh_eye_r", "mesh_mouth"]:
        obj_path = os.path.join(model_dir, f"{part}.obj")
        vertices, uvs, _, vertex_faces, uv_faces, _ = load_uv_obj_file(obj_path)

        # Remap UVs for non-head parts
        if part in PART_REGIONS and uvs is not None:
            u_min, v_min, u_size, v_size = PART_REGIONS[part]
            uvs = uvs.copy()
            uvs[:, 0] = uvs[:, 0] * u_size + u_min
            uvs[:, 1] = uvs[:, 1] * v_size + v_min

        all_verts.append(vertices)
        if uvs is not None:
            all_uvs.append(uvs)
        all_faces_v.append(vertex_faces + v_offset)
        if uv_faces is not None:
            all_faces_uv.append(uv_faces + uv_offset)

        v_offset += len(vertices)
        uv_offset += len(uvs) if uvs is not None else 0
        n_uvs = len(uvs) if uvs is not None else 0
        print(f"  {part}: {len(vertices)} verts, {n_uvs} uvs, {len(vertex_faces)} faces")

    verts = np.concatenate(all_verts, axis=0)
    uvs = np.concatenate(all_uvs, axis=0) if all_uvs else None
    faces_v = np.concatenate(all_faces_v, axis=0)
    faces_uv = np.concatenate(all_faces_uv, axis=0) if all_faces_uv else None

    n_uvs = len(uvs) if uvs is not None else 0
    print(f"  Combined: {len(verts)} verts, {n_uvs} uvs, {len(faces_v)} faces")
    return verts, uvs, faces_v, faces_uv


def write_obj(path: str, verts: np.ndarray, uvs: np.ndarray, faces_v: np.ndarray, faces_uv: np.ndarray, mtl_name: str = None):
    with open(path, "w") as f:
        if mtl_name:
            f.write(f"mtllib {mtl_name}\nusemtl material0\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        for i in range(len(faces_v)):
            fv = faces_v[i]
            if faces_uv is not None:
                fuv = faces_uv[i]
                f.write(f"f {fv[0]+1}/{fuv[0]+1} {fv[1]+1}/{fuv[1]+1} {fv[2]+1}/{fuv[2]+1}\n")
            else:
                f.write(f"f {fv[0]+1} {fv[1]+1} {fv[2]+1}\n")


def write_mtl(path: str, albedo_png: str, normal_png: str = None):
    with open(path, "w") as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\n")
        f.write(f"map_Kd {albedo_png}\n")
        if normal_png:
            f.write(f"bump {normal_png}\n")


def _place_texture_in_region(canvas: np.ndarray, tex_rgb: np.ndarray, u_min: float, v_min: float, u_size: float, v_size: float):
    """Place a texture into the atlas region. UV v=0 is bottom of image, v=1 is top."""
    h = canvas.shape[0]
    # UV to pixel: x = u * (h-1), y = (1-v) * (h-1)
    x0 = int(round(u_min * (h - 1)))
    x1 = int(round((u_min + u_size) * (h - 1)))
    y1 = int(round((1.0 - v_min) * (h - 1)))  # v_min -> bottom -> large y
    y0 = int(round((1.0 - v_min - v_size) * (h - 1)))  # v_min+v_size -> top -> small y
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    resized = cv2.resize(tex_rgb, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
    canvas[y0:y0 + box_h, x0:x0 + box_w] = resized


def build_atlas(packer: TexturePackHelper, data_root: str, sample_id: str, file_names: dict, default_color=None) -> np.ndarray | None:
    """Build a texture atlas matching the UV layout.

    file_names: dict mapping part key -> texture filename (e.g. {"head": "T_Head_BC.png"})
    """
    texture_root = packer.get_texture_root(data_root)
    if texture_root is None:
        return None

    mats_path = packer.find_mats_file(data_root, sample_id)
    texture_info = packer._parse_mats_texture_info(mats_path)

    # Load head texture (required)
    head_path = packer._resolve_part_texture_path(texture_root, texture_info, TEXTURE_KEYS["head"], file_names.get("head", ""))
    if head_path is None:
        return None
    head_rgb, head_alpha = packer._load_texture_png(head_path)
    if head_rgb is None:
        return None

    sz = ATLAS_SIZE
    if default_color is not None:
        canvas = np.full((sz, sz, 3), default_color, dtype=np.float32)
    else:
        canvas = np.zeros((sz, sz, 3), dtype=np.float32)

    # Head: resize to full atlas, composite with alpha
    head_r = cv2.resize(head_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
    if head_alpha is not None:
        alpha_r = cv2.resize(head_alpha, (sz, sz), interpolation=cv2.INTER_LINEAR)
        if alpha_r.ndim == 2:
            alpha_r = alpha_r[:, :, None]
        canvas = canvas * (1.0 - alpha_r) + head_r * alpha_r
    else:
        canvas = head_r

    # Eye/mouth parts: load and place into their atlas regions
    part_to_mesh = {"eye_l": "mesh_eye_l", "eye_r": "mesh_eye_r", "mouth": "mesh_mouth"}
    for part_key, mesh_key in part_to_mesh.items():
        if part_key not in file_names:
            continue
        tex_path = packer._resolve_part_texture_path(texture_root, texture_info, TEXTURE_KEYS[part_key], file_names[part_key])
        if tex_path is None:
            continue
        part_rgb, _ = packer._load_texture_png(tex_path)
        if part_rgb is None:
            continue
        u_min, v_min, u_size, v_size = PART_REGIONS[mesh_key]
        _place_texture_in_region(canvas, part_rgb, u_min, v_min, u_size, v_size)

    return np.clip(canvas, 0.0, 1.0)


def find_sample_ids(data_roots: list, max_samples: int) -> list:
    results = []
    for root in data_roots:
        mat_dir = os.path.join(root, "mat")
        if not os.path.isdir(mat_dir):
            continue
        for name in sorted(os.listdir(mat_dir)):
            if name.startswith("Mats_") and name.endswith(".txt"):
                sample_id = name[5:-4]
                results.append((root, sample_id))
                if len(results) >= max_samples:
                    return results
    return results


def save_image(path: str, rgb_float: np.ndarray):
    u8 = (rgb_float * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Merge template mesh OBJs with remapped UVs
    print("Merging template mesh OBJs (remapping eye/mouth UVs)...")
    verts, uvs, faces_v, faces_uv = load_and_merge_mesh_objs(MODEL_DIR)

    obj_path = os.path.join(OUTPUT_DIR, "combined_mesh.obj")
    mtl_path = os.path.join(OUTPUT_DIR, "combined_mesh.mtl")
    write_obj(obj_path, verts, uvs, faces_v, faces_uv, mtl_name="combined_mesh.mtl")
    write_mtl(mtl_path, "albedo.png", "normal.png")
    print(f"Saved: {obj_path}")

    # 2. Find samples and build atlases
    packer = TexturePackHelper(texture_root=TEXTURE_ROOT)
    samples = find_sample_ids(DATA_ROOTS, NUM_SAMPLES)
    print(f"\nFound {len(samples)} samples to export")

    first_albedo_saved = False
    first_normal_saved = False

    for i, (data_root, sample_id) in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] {sample_id}")

        # Albedo atlas
        albedo = build_atlas(packer, data_root, sample_id, ALBEDO_FILES)
        if albedo is not None:
            out = os.path.join(OUTPUT_DIR, f"albedo_{sample_id}.png")
            save_image(out, albedo)
            print(f"  Albedo: {out}")
            if not first_albedo_saved:
                save_image(os.path.join(OUTPUT_DIR, "albedo.png"), albedo)
                first_albedo_saved = True
        else:
            print("  Albedo: FAILED")

        # Normal atlas (head only; flat normal [0.5, 0.5, 1.0] as default)
        normal = build_atlas(packer, data_root, sample_id, NORMAL_FILES, default_color=[0.5, 0.5, 1.0])
        if normal is not None:
            out = os.path.join(OUTPUT_DIR, f"normal_{sample_id}.png")
            save_image(out, normal)
            print(f"  Normal: {out}")
            if not first_normal_saved:
                save_image(os.path.join(OUTPUT_DIR, "normal.png"), normal)
                first_normal_saved = True
        else:
            print("  Normal: FAILED")

    print(f"\nDone! Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
