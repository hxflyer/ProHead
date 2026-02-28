import glob
import os
import re
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np


class TexturePackHelper:
    """Resolve source textures and compose the combined UV texture map."""

    def __init__(
        self,
        texture_root: Optional[str] = None,
        texture_png_cache_max_items: int = 64,
        combined_texture_cache_max_items: int = 0,
        mesh_texture_size: int = 1024,
        eye_box_size: float = 0.20,
        eye_l_u_start: float = 0.01,
        eye_r_u_start: float = 0.79,
        bottom_margin: float = 0.01,
        mouth_split_v: float = 0.61,
        mouth_box_size: float = 0.25,
        mouth_v_gt_u_start: float = 0.50,
        mouth_v_le_u_start: float = 0.22,
    ):
        self.texture_root = texture_root

        self.mesh_texture_size = int(mesh_texture_size)
        self.eye_box_size = float(eye_box_size)
        self.eye_l_u_start = float(eye_l_u_start)
        self.eye_r_u_start = float(eye_r_u_start)
        self.bottom_margin = float(bottom_margin)
        self.mouth_split_v = float(mouth_split_v)
        self.mouth_box_size = float(mouth_box_size)
        self.mouth_v_gt_u_start = float(mouth_v_gt_u_start)
        self.mouth_v_le_u_start = float(mouth_v_le_u_start)

        self._mats_cache = {}
        self._texture_path_cache = {}
        self._texture_image_cache = OrderedDict()
        self._combined_texture_cache = OrderedDict()
        self._texture_png_cache_max_items = int(max(0, texture_png_cache_max_items))
        self._combined_texture_cache_max_items = int(max(0, combined_texture_cache_max_items))
        self._texture_root_cache = None

    def get_texture_root(self, data_root: str) -> Optional[str]:
        if self._texture_root_cache is not None:
            return self._texture_root_cache

        if self.texture_root:
            if os.path.isdir(self.texture_root):
                self._texture_root_cache = self.texture_root
                return self._texture_root_cache
            return None
        return None

    def find_mats_file(self, data_root: str, sample_id: str) -> Optional[str]:
        search_dirs = [os.path.join(data_root, "mat"), data_root]
        pattern = re.compile(rf"^Mats_{re.escape(sample_id)}\.txt$")
        for base_dir in search_dirs:
            try:
                for name in os.listdir(base_dir):
                    if pattern.match(name):
                        return os.path.join(base_dir, name)
            except Exception:
                pass
        return None

    @staticmethod
    def _extract_game_folder_from_asset_path(asset_path: str) -> Optional[str]:
        if not asset_path:
            return None
        p = asset_path.strip().replace("\\", "/")
        if p.startswith("/Game/"):
            p = p[len("/Game/"):]
        if "/Face/Materials/" in p:
            p = p.split("/Face/Materials/", 1)[0]
        elif "/Face/" in p:
            p = p.split("/Face/", 1)[0]
        p = p.strip("/")
        return p or None

    def _parse_mats_texture_info(self, mats_path: Optional[str]) -> dict:
        if not mats_path or not os.path.exists(mats_path):
            return {}
        if mats_path in self._mats_cache:
            return self._mats_cache[mats_path]

        sections = {
            "Face Texture:": "face",
            "Left Eye Texture:": "eye_l",
            "Right Eye Texture:": "eye_r",
        }
        current_key = None
        info = {}
        try:
            with open(mats_path, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        current_key = None
                        continue
                    if line in sections:
                        current_key = sections[line]
                        info.setdefault(current_key, {})
                        continue
                    if current_key is None:
                        continue
                    if line.startswith("Path:"):
                        info[current_key]["path"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Name:"):
                        info[current_key]["name"] = line.split(":", 1)[1].strip()
        except Exception:
            info = {}

        for key in ("face", "eye_l", "eye_r"):
            d = info.get(key, {})
            asset_path = d.get("path", "")
            folder = self._extract_game_folder_from_asset_path(asset_path)
            d["folder"] = folder
            if "name" not in d or not d["name"]:
                name_from_path = asset_path.split("/")[-1] if asset_path else ""
                if "." in name_from_path:
                    name_from_path = name_from_path.split(".")[-1] or name_from_path.split(".")[0]
                d["name"] = name_from_path
            if d:
                info[key] = d

        self._mats_cache[mats_path] = info
        return info

    def _resolve_part_texture_path(self, texture_root: str, texture_info: dict, texture_key: str, file_name: str) -> Optional[str]:
        if texture_key == "mouth_default":
            path = os.path.join(texture_root, "3DScanStore", "F_15", "MI_Teeth_Baked", file_name)
            return path if os.path.exists(path) else None

        entry = texture_info.get(texture_key, {})
        material_name = entry.get("name")
        if not material_name:
            return None

        folder_key = entry.get("folder")
        cache_key = (texture_root, texture_key, folder_key, material_name, file_name)
        if cache_key in self._texture_path_cache:
            return self._texture_path_cache[cache_key]

        if folder_key:
            path = os.path.join(texture_root, *folder_key.split("/"), material_name, file_name)
            if os.path.exists(path):
                self._texture_path_cache[cache_key] = path
                return path
            return None

        recursive_pat = os.path.join(texture_root, "**", material_name, file_name)
        matches = glob.glob(recursive_pat, recursive=True)
        if matches:
            path = sorted(set(matches))[0]
            self._texture_path_cache[cache_key] = path
            return path
        return None

    @staticmethod
    def _derive_alpha_from_rgb(rgb: np.ndarray) -> np.ndarray:
        m = (rgb.max(axis=2, keepdims=True) > (1.0 / 255.0)).astype(np.float32)
        return m

    def _load_texture_png(self, filepath: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not filepath:
            return None, None
        if filepath in self._texture_image_cache:
            cached = self._texture_image_cache.pop(filepath)
            self._texture_image_cache[filepath] = cached
            return cached
        if not os.path.exists(filepath):
            return None, None

        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None

        if img.ndim == 2:
            bgr = np.stack([img, img, img], axis=2)
            alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=np.float32)
        elif img.shape[2] >= 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        else:
            bgr = img[:, :, :3]
            alpha = None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if alpha is None:
            alpha = self._derive_alpha_from_rgb(rgb)
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)

        if self._texture_png_cache_max_items > 0:
            self._texture_image_cache[filepath] = (rgb, alpha)
            while len(self._texture_image_cache) > self._texture_png_cache_max_items:
                self._texture_image_cache.popitem(last=False)
        return rgb, alpha

    @staticmethod
    def _uv_box_to_pixel_rect(u0: float, v0: float, box_size: float, tex_size: int) -> tuple[int, int, int, int]:
        x0 = int(round(float(u0) * float(tex_size - 1)))
        x1 = int(round(float(u0 + box_size) * float(tex_size - 1)))
        y_bottom = int(round((1.0 - float(v0)) * float(tex_size - 1)))
        y_top = int(round((1.0 - float(v0 + box_size)) * float(tex_size - 1)))

        x0 = max(0, min(tex_size - 1, x0))
        x1 = max(0, min(tex_size - 1, x1))
        y_top = max(0, min(tex_size - 1, y_top))
        y_bottom = max(0, min(tex_size - 1, y_bottom))

        if x1 < x0:
            x0, x1 = x1, x0
        if y_bottom < y_top:
            y_top, y_bottom = y_bottom, y_top
        return x0, y_top, x1, y_bottom

    @staticmethod
    def _crop_by_alpha(rgb: np.ndarray, alpha: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if rgb is None or alpha is None or rgb.size == 0 or alpha.size == 0:
            return None, None
        mask = alpha[..., 0] > 1e-6
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            return None, None
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        return rgb[y0:y1, x0:x1].copy(), alpha[y0:y1, x0:x1].copy()

    @staticmethod
    def _resize_rgba(rgb: np.ndarray, alpha: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
        w = max(1, int(w))
        h = max(1, int(h))
        rgb_r = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha_r = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        if alpha_r.ndim == 2:
            alpha_r = alpha_r[:, :, None]
        alpha_r = np.clip(alpha_r.astype(np.float32), 0.0, 1.0)
        return rgb_r.astype(np.float32), alpha_r

    def _paste_texture_into_uv_box(
        self,
        canvas_rgb: np.ndarray,
        src_rgb: Optional[np.ndarray],
        src_alpha: Optional[np.ndarray],
        u0: float,
        v0: float,
        box_size: float,
        align_bottom: bool = False,
    ) -> None:
        if src_rgb is None or src_alpha is None:
            return
        if src_rgb.size == 0 or src_alpha.size == 0:
            return

        src_rgb, src_alpha = self._crop_by_alpha(src_rgb, src_alpha)
        if src_rgb is None or src_alpha is None:
            return

        tex_size = int(canvas_rgb.shape[0])
        x0, y_top, x1, y_bottom = self._uv_box_to_pixel_rect(u0, v0, box_size, tex_size)
        box_w = max(1, x1 - x0 + 1)
        box_h = max(1, y_bottom - y_top + 1)

        h_s, w_s = src_rgb.shape[:2]
        scale = min(float(box_w) / float(max(1, w_s)), float(box_h) / float(max(1, h_s)))
        out_w = max(1, int(round(w_s * scale)))
        out_h = max(1, int(round(h_s * scale)))
        rgb_r, alpha_r = self._resize_rgba(src_rgb, src_alpha, out_w, out_h)

        dst_x = x0 + max(0, (box_w - out_w) // 2)
        if align_bottom:
            dst_y = y_top + max(0, box_h - out_h)
        else:
            dst_y = y_top + max(0, (box_h - out_h) // 2)

        x_end = min(tex_size, dst_x + out_w)
        y_end = min(tex_size, dst_y + out_h)
        if x_end <= dst_x or y_end <= dst_y:
            return

        cut_w = x_end - dst_x
        cut_h = y_end - dst_y
        rgb_cut = rgb_r[:cut_h, :cut_w]
        alpha_cut = alpha_r[:cut_h, :cut_w]

        roi = canvas_rgb[dst_y:y_end, dst_x:x_end]
        canvas_rgb[dst_y:y_end, dst_x:x_end] = roi * (1.0 - alpha_cut) + rgb_cut * alpha_cut

    @staticmethod
    def _split_texture_by_v_threshold(
        rgb: np.ndarray,
        alpha: np.ndarray,
        v_threshold: float,
    ) -> tuple[tuple[Optional[np.ndarray], Optional[np.ndarray]], tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        if rgb is None or alpha is None or rgb.size == 0 or alpha.size == 0:
            return (None, None), (None, None)
        h = int(rgb.shape[0])
        if h <= 1:
            return (None, None), (rgb.copy(), alpha.copy())

        y_bound = int(np.floor((1.0 - float(v_threshold)) * float(h - 1)))
        y_bound = max(0, min(h, y_bound))

        alpha_high = np.zeros_like(alpha, dtype=np.float32)
        alpha_low = np.zeros_like(alpha, dtype=np.float32)
        if y_bound > 0:
            alpha_high[:y_bound, :, :] = alpha[:y_bound, :, :]
        alpha_low[y_bound:, :, :] = alpha[y_bound:, :, :]

        rgb_high, alpha_high = TexturePackHelper._crop_by_alpha(rgb, alpha_high)
        rgb_low, alpha_low = TexturePackHelper._crop_by_alpha(rgb, alpha_low)
        return (rgb_high, alpha_high), (rgb_low, alpha_low)

    def load_mesh_texture_map(self, data_root: str, sample_id: str) -> Optional[np.ndarray]:
        texture_root = self.get_texture_root(data_root)
        if texture_root is None:
            return None

        mats_path = self.find_mats_file(data_root, sample_id)
        texture_info = self._parse_mats_texture_info(mats_path)
        required_parts = {
            "head": {"texture_key": "face", "texture_file_name": "T_Head_BC.png"},
            "eye_l": {"texture_key": "eye_l", "texture_file_name": "T_EyeL_Composite_BC.png"},
            "eye_r": {"texture_key": "eye_r", "texture_file_name": "T_EyeR_Composite_BC.png"},
            "mouth": {"texture_key": "mouth_default", "texture_file_name": "T_Teeth_BC.png"},
        }
        resolved_paths = {}
        for part, spec in required_parts.items():
            tex_name = spec.get("texture_file_name")
            if not tex_name:
                return None
            path = self._resolve_part_texture_path(
                texture_root=texture_root,
                texture_info=texture_info,
                texture_key=spec["texture_key"],
                file_name=tex_name,
            )
            if path is None or not os.path.exists(path):
                return None
            resolved_paths[part] = path

        cache_key = (
            self.mesh_texture_size,
            self.eye_box_size,
            self.eye_l_u_start,
            self.eye_r_u_start,
            self.bottom_margin,
            self.mouth_split_v,
            self.mouth_box_size,
            self.mouth_v_gt_u_start,
            self.mouth_v_le_u_start,
            resolved_paths["head"],
            resolved_paths["eye_l"],
            resolved_paths["eye_r"],
            resolved_paths["mouth"],
        )
        if cache_key in self._combined_texture_cache:
            cached = self._combined_texture_cache.pop(cache_key)
            self._combined_texture_cache[cache_key] = cached
            return cached.copy()

        head_rgb, head_alpha = self._load_texture_png(resolved_paths["head"])
        eye_l_rgb, eye_l_alpha = self._load_texture_png(resolved_paths["eye_l"])
        eye_r_rgb, eye_r_alpha = self._load_texture_png(resolved_paths["eye_r"])
        mouth_rgb, mouth_alpha = self._load_texture_png(resolved_paths["mouth"])
        if head_rgb is None or eye_l_rgb is None or eye_r_rgb is None or mouth_rgb is None:
            return None

        eye_l_alpha = np.ones((eye_l_rgb.shape[0], eye_l_rgb.shape[1], 1), dtype=np.float32)
        eye_r_alpha = np.ones((eye_r_rgb.shape[0], eye_r_rgb.shape[1], 1), dtype=np.float32)

        tex_size = int(self.mesh_texture_size)
        canvas = np.zeros((tex_size, tex_size, 3), dtype=np.float32)

        head_rgb_r, head_alpha_r = self._resize_rgba(head_rgb, head_alpha, tex_size, tex_size)
        canvas = canvas * (1.0 - head_alpha_r) + head_rgb_r * head_alpha_r

        self._paste_texture_into_uv_box(
            canvas_rgb=canvas,
            src_rgb=eye_l_rgb,
            src_alpha=eye_l_alpha,
            u0=self.eye_l_u_start,
            v0=self.bottom_margin,
            box_size=self.eye_box_size,
            align_bottom=False,
        )
        self._paste_texture_into_uv_box(
            canvas_rgb=canvas,
            src_rgb=eye_r_rgb,
            src_alpha=eye_r_alpha,
            u0=self.eye_r_u_start,
            v0=self.bottom_margin,
            box_size=self.eye_box_size,
            align_bottom=False,
        )

        (mouth_high_rgb, mouth_high_alpha), (mouth_low_rgb, mouth_low_alpha) = self._split_texture_by_v_threshold(
            mouth_rgb,
            mouth_alpha,
            v_threshold=self.mouth_split_v,
        )
        self._paste_texture_into_uv_box(
            canvas_rgb=canvas,
            src_rgb=mouth_high_rgb,
            src_alpha=mouth_high_alpha,
            u0=self.mouth_v_gt_u_start,
            v0=self.bottom_margin,
            box_size=self.mouth_box_size,
            align_bottom=True,
        )
        self._paste_texture_into_uv_box(
            canvas_rgb=canvas,
            src_rgb=mouth_low_rgb,
            src_alpha=mouth_low_alpha,
            u0=self.mouth_v_le_u_start,
            v0=self.bottom_margin,
            box_size=self.mouth_box_size,
            align_bottom=True,
        )

        canvas = np.clip(canvas, 0.0, 1.0).astype(np.float32)
        if self._combined_texture_cache_max_items > 0:
            self._combined_texture_cache[cache_key] = canvas
            while len(self._combined_texture_cache) > self._combined_texture_cache_max_items:
                self._combined_texture_cache.popitem(last=False)
        return canvas.copy()

