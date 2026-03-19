"""
Fast RGB+Geometry Dataset for Geometry Training (Landmark + Mesh)
Loads RGB images, landmarks, and mesh vertices with aligned normalization
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, Optional
import cv2

from align_5pt_helper import Align5PtHelper
from tex_pack_helper import TexturePackHelper
from mat_load_helper import load_matrix_data, compute_vertex_depth

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not installed. Using basic augmentation. Install with: pip install albumentations")


def _should_log_dataset_init() -> bool:
    """Log dataset construction once in DDP (rank 0 only)."""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        pass
    return True


def _load_landmark_pixels(filepath: Optional[str]) -> Optional[np.ndarray]:
    """Read [N, 2] pixel coords (cols 3, 4) from a landmark txt file."""
    if filepath is None or not os.path.exists(filepath):
        return None
    pts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 5:
                pts.append([float(parts[3]), float(parts[4])])
    return np.array(pts, dtype=np.float32) if pts else None


def _load_exr_as_float32(path: str) -> Optional[np.ndarray]:
    """Load an EXR file and return RGB float32 [H,W,3] clipped to [0,1], or None on failure."""
    geo_float = None
    try:
        import Imath, OpenEXR as _OE
        _ef = _OE.InputFile(path)
        _hdr = _ef.header()
        _dw = _hdr['dataWindow']
        _ew, _eh = _dw.max.x - _dw.min.x + 1, _dw.max.y - _dw.min.y + 1
        _chs = {}
        for _ch in ['R', 'G', 'B']:
            if _ch in _hdr['channels']:
                _ct = _hdr['channels'][_ch].type
                if _ct == Imath.PixelType(Imath.PixelType.FLOAT):
                    _raw = _ef.channel(_ch, Imath.PixelType(Imath.PixelType.FLOAT))
                    _chs[_ch] = np.frombuffer(_raw, dtype=np.float32).reshape(_eh, _ew)
                else:
                    _raw = _ef.channel(_ch, Imath.PixelType(Imath.PixelType.HALF))
                    _chs[_ch] = np.frombuffer(_raw, dtype=np.float16).reshape(_eh, _ew).astype(np.float32)
        if _chs:
            geo_float = np.stack([_chs.get(c, np.zeros((_eh, _ew), np.float32)) for c in ['R', 'G', 'B']], axis=-1)
    except ImportError:
        pass
    except Exception:
        pass
    if geo_float is None:
        try:
            import imageio.v3 as _iio
            _arr = _iio.imread(path).astype(np.float32)
            if _arr.ndim == 3 and _arr.shape[2] >= 3:
                geo_float = _arr[..., :3]
        except Exception:
            pass
    if geo_float is None:
        try:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            _arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if _arr is not None:
                geo_float = cv2.cvtColor(_arr.astype(np.float32), cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    if geo_float is None:
        return None
    if geo_float.ndim == 2:
        geo_float = np.repeat(geo_float[..., None], 3, axis=2)
    elif geo_float.ndim == 3 and geo_float.shape[2] >= 3:
        geo_float = geo_float[..., :3]
    elif geo_float.ndim == 3 and geo_float.shape[2] == 1:
        geo_float = np.repeat(geo_float, 3, axis=2)
    else:
        return None
    geo_float = np.nan_to_num(geo_float.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(geo_float, 0.0, 1.0)


class FastGeometryDataset(Dataset):
    """
    Fast dataset for RGB geometry training (landmark + mesh).
    Loads RGB images, landmarks, and mesh vertices.
    """

    ALIGN_SCALE_JITTER = 0.08
    ALIGN_TRANSLATE_JITTER = 0.03
    
    def __init__(
        self,
        data_roots,
        split: str = 'train',
        image_size: int = 512,
        train_ratio: float = 0.8,
        augment: bool = True,
        texture_root: Optional[str] = None,
        texture_png_cache_max_items: int = 64,
        combined_texture_cache_max_items: int = 0,
    ):
        """
        Args:
            data_roots: Path or list of paths to MetaHuman dataset directories
            split: 'train' or 'val'
            image_size: Target image size
            train_ratio: Ratio for train/val split
            augment: Apply data augmentation
            texture_root: Optional override for mesh textures root directory
            texture_png_cache_max_items: Max cached source texture PNGs per dataset instance (0 disables cache)
            combined_texture_cache_max_items: Max cached composed 1024 texture maps per dataset instance (0 disables cache)

        Note: Samples without both landmark and mesh txt files are skipped entirely.
        """
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = data_roots
            
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.texture_root = texture_root

        self.align_helper = Align5PtHelper(
            image_size=self.image_size,
            scale_jitter=float(max(0.0, self.ALIGN_SCALE_JITTER)),
            translate_jitter=float(max(0.0, self.ALIGN_TRANSLATE_JITTER)),
        )
        self.texture_packer = TexturePackHelper(
            texture_root=self.texture_root,
            texture_png_cache_max_items=texture_png_cache_max_items,
            combined_texture_cache_max_items=combined_texture_cache_max_items,
        )
        self._mesh_texture_size = int(self.texture_packer.mesh_texture_size)
        self.landmark_indices = None
        self.mesh_indices = None
        self.default_landmarks = None
        self.default_mesh = None
        self._load_default_templates()

        # Find all samples
        self.samples = []  # List of (data_root, color_path, landmark_path|None, mesh_path|None, basecolor_path|None)
        
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                continue
            
            # Find Color files (exclude mask files)
            color_files = glob.glob(os.path.join(data_root, "Color_*"))
            color_files = [f for f in color_files if os.path.basename(f).startswith("Color_") and not f.endswith("_mask.png")]
            
            for color_file in color_files:
                filename = os.path.basename(color_file)
                if filename.startswith("Color_"):
                    temp_name = filename[6:]
                else:
                    continue
                
                full_id = os.path.splitext(temp_name)[0]
                
                # Parse base_id (remove variant suffixes)
                base_id = full_id
                for s in ['_gemini', '_flux', '_seedream']:
                    if full_id.endswith(s):
                        base_id = full_id[:-len(s)]
                        break
                
                lm_p   = os.path.join(data_root, "landmark", f"landmark_{base_id}.txt")
                mesh_p = os.path.join(data_root, "mesh",     f"mesh_{base_id}.txt")
                basecolor_path = os.path.join(data_root, "basecolor", f"BaseColor_{base_id}.png")

                if not os.path.exists(lm_p) or not os.path.exists(mesh_p):
                    continue

                self.samples.append((data_root, color_file, lm_p, mesh_p, basecolor_path))
        
        # Remove duplicates and sort deterministically.
        self.samples = list(set(self.samples))
        self.samples.sort()

        # Deterministic per-folder split:
        # - validation = first (1 - train_ratio) fraction from each folder
        # - training   = remaining samples from each folder
        # This keeps val stable and comparable across runs/dataset mixes.
        samples_by_root = {}
        for sample in self.samples:
            root = sample[0]
            samples_by_root.setdefault(root, []).append(sample)

        split_samples = []
        val_ratio = float(np.clip(1.0 - float(train_ratio), 0.0, 1.0))
        for root in sorted(samples_by_root.keys()):
            root_samples = sorted(samples_by_root[root])
            val_count = int(len(root_samples) * val_ratio)
            if split == 'train':
                split_samples.extend(root_samples[val_count:])
            else:
                split_samples.extend(root_samples[:val_count])

        self.samples = split_samples

        if _should_log_dataset_init():
            print(f"[{self.__class__.__name__}] split={split}  total={len(self.samples)}")

        # Augmentation transforms (image-only; geometric augment is disabled).
        self.image_only_augment = None
        self.image_only_augment_tv = None

        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.image_only_augment = A.Compose([
                A.ISONoise(
                    color_shift=(0.0, 0.02),
                    intensity=(0.1, 0.5),
                    p=0.3
                ),
                A.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                    p=0.8
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                ], p=0.3),
                A.UnsharpMask(
                    blur_limit=(3, 7),
                    sigma_limit=0.0,
                    alpha=(0.2, 0.5),
                    threshold=10,
                    p=0.3
                ),
            ])
        elif self.augment:
            self.image_only_augment_tv = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            ])

    def _get_texture_root(self, data_root: str) -> Optional[str]:
        return self.texture_packer.get_texture_root(data_root)

    def _find_mats_file(self, data_root: str, sample_id: str) -> Optional[str]:
        return self.texture_packer.find_mats_file(data_root, sample_id)

    def _load_mesh_texture_map(self, data_root: str, sample_id: str) -> Optional[np.ndarray]:
        return self.texture_packer.load_mesh_texture_map(data_root, sample_id)


    def _load_default_templates(self) -> None:
        landmark_path = "model/landmark_template.npy"
        mesh_path = "model/mesh_template.npy"
        if os.path.exists("model/landmark_indices.npy"):
            self.landmark_indices = np.load("model/landmark_indices.npy")
        if os.path.exists("model/mesh_indices.npy"):
            self.mesh_indices = np.load("model/mesh_indices.npy")

        landmark_template = np.load(landmark_path).astype(np.float32) if os.path.exists(landmark_path) else None
        mesh_template = np.load(mesh_path).astype(np.float32) if os.path.exists(mesh_path) else None

        if landmark_template is None or mesh_template is None:
            raise FileNotFoundError("Missing model templates for default geometry fallback.")

        if self.landmark_indices is not None and self.landmark_indices.max() < landmark_template.shape[0]:
            landmark_template = landmark_template[self.landmark_indices]
        if self.mesh_indices is not None and self.mesh_indices.max() < mesh_template.shape[0]:
            mesh_template = mesh_template[self.mesh_indices]

        if landmark_template.shape[1] < 5 or mesh_template.shape[1] < 5:
            raise ValueError("Template geometry must contain at least 5 dims (x,y,z,u,v).")

        self.default_landmarks = landmark_template[:, :5].astype(np.float32, copy=True)
        self.default_mesh = mesh_template[:, :5].astype(np.float32, copy=True)

        # Template depth (column 5) for computing depth offsets.
        # If templates have a 6th column with precomputed average depth, use it;
        # otherwise fall back to zeros.
        if landmark_template.shape[1] >= 6:
            self.template_landmark_depth = landmark_template[:, 5].astype(np.float32, copy=True)
        else:
            self.template_landmark_depth = np.zeros(landmark_template.shape[0], dtype=np.float32)
        if mesh_template.shape[1] >= 6:
            self.template_mesh_depth = mesh_template[:, 5].astype(np.float32, copy=True)
        else:
            self.template_mesh_depth = np.zeros(mesh_template.shape[0], dtype=np.float32)

    def _find_basecolor_file(self, data_root: str, color_path: str, sample_id: str) -> Optional[str]:
        # Color: Color_id1_id2_suffix.png -> BaseColor: basecolor/BaseColor_id1_id2.png
        # sample_id is id1_id2 (suffix already removed)
        basecolor_path = os.path.join(data_root, "basecolor", f"BaseColor_{sample_id}.png")
        if os.path.exists(basecolor_path):
            return basecolor_path
        return None

    def _find_file(self, data_root: str, pattern: str, sample_id: str) -> Optional[str]:
        """Find file matching pattern."""
        for ext in ['.txt', '.png', '.jpg', '.jpeg']:
            filepath = os.path.join(data_root, f"{pattern}_{sample_id}{ext}")
            if os.path.exists(filepath):
                return filepath
            
            if pattern == 'landmark':
                filepath = os.path.join(data_root, "landmark", f"{pattern}_{sample_id}{ext}")
                if os.path.exists(filepath):
                    return filepath
                filepath = os.path.join(data_root, "landmark", f"{sample_id}{ext}")
                if os.path.exists(filepath):
                    return filepath
            elif pattern == 'mesh':
                filepath = os.path.join(data_root, "mesh", f"{pattern}_{sample_id}{ext}")
                if os.path.exists(filepath):
                    return filepath
                filepath = os.path.join(data_root, "mesh", f"{sample_id}{ext}")
                if os.path.exists(filepath):
                    return filepath
        return None
    
    def _load_geometry(self, filepath: str, return_raw_xyz: bool = False):
        """Load geometry from txt file (5 cols: x3d y3d z3d x2d y2d)."""
        try:
            geom = []
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.replace(',', ' ').split()
                    
                    if len(parts) >= 5:
                        vals = [float(x) for x in parts[:5]]
                        geom.append(vals)
                    elif len(parts) >= 3:
                        # XYZ only, pad with 0,0 for UV
                        geom.append([float(parts[0]), float(parts[1]), float(parts[2]), 0.0, 0.0])
            
            geom = np.array(geom, dtype=np.float32)
            
            if geom.shape[0] == 0:
                return (None, None) if return_raw_xyz else None

            # Instance Normalization (aligned via build_template)
            xyz = geom[:, 0:3]
            raw_xyz = xyz.copy()
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            center = (min_xyz + max_xyz) / 2
            scale = np.max(max_xyz - min_xyz) / 2
            
            if scale > 0:
                geom[:, 0:3] = (xyz - center) / scale
            
            # Normalize 2D coords
            geom[:, 3:5] = geom[:, 3:5] / 1024.0
            
            if return_raw_xyz:
                return geom, raw_xyz
            return geom
        except Exception:
            return (None, None) if return_raw_xyz else None

    @staticmethod
    def _compute_geometry_found_mask(geom: Optional[np.ndarray]) -> np.ndarray:
        if geom is None or geom.ndim != 2 or geom.shape[1] < 5:
            return np.zeros((0,), dtype=bool)
        uv = geom[:, 3:5].astype(np.float32, copy=False)
        return np.isfinite(uv).all(axis=1) & np.all(uv >= 0.0, axis=1)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        data_root, color_path, landmark_path, mesh_path, basecolor_path = self.samples[idx]

        # Load RGB
        try:
            image_np = cv2.imread(color_path)
            if image_np is None:
                raise ValueError(f"Could not read {color_path}")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except Exception:
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        h, w = image_np.shape[:2]

        basecolor_np = None
        basecolor_valid = 0.0
        if basecolor_path is not None and os.path.exists(basecolor_path):
            try:
                bc = cv2.imread(basecolor_path)
                if bc is not None:
                    basecolor_np = cv2.cvtColor(bc, cv2.COLOR_BGR2RGB)
                    basecolor_valid = 1.0
            except Exception:
                basecolor_np = None
                basecolor_valid = 0.0
        if basecolor_np is None:
            basecolor_np = np.zeros_like(image_np, dtype=np.uint8)

        # Load geometry
        landmarks, landmarks_raw_xyz = self._load_geometry(landmark_path, return_raw_xyz=True)
        mesh, mesh_raw_xyz = self._load_geometry(mesh_path, return_raw_xyz=True)

        mesh_texture = None
        mesh_texture_valid = None

        geometry_fallback = landmarks is None or mesh is None
        if geometry_fallback:
            # Should not happen since we skip samples without geometry files,
            # but guard against corrupted files.
            landmarks = self.default_landmarks.copy()
            mesh = self.default_mesh.copy()
            landmarks_raw_xyz = None
            mesh_raw_xyz = None

        landmark_found_mask = self._compute_geometry_found_mask(landmarks)
        mesh_found_mask = self._compute_geometry_found_mask(mesh)
        if geometry_fallback:
            landmark_found_mask = np.zeros((landmarks.shape[0],), dtype=bool)
            mesh_found_mask = np.zeros((mesh.shape[0],), dtype=bool)

        landmark_weights = np.ones((landmarks.shape[0],), dtype=np.float32)
        mesh_weights = np.ones((mesh.shape[0],), dtype=np.float32)
        landmark_weights[~landmark_found_mask] = 0.0
        mesh_weights[~mesh_found_mask] = 0.0

        # Load mask (if present) to down-weight occluded landmarks/mesh vertices.
        dir_name = os.path.dirname(color_path)
        filename = os.path.basename(color_path)
        name_no_ext, ext = os.path.splitext(filename)
        if name_no_ext.startswith("Color_"):
            name_no_ext = name_no_ext[len("Color_"):]

        for suffix in ['_gemini', '_flux', '_seedream']:
            if name_no_ext.endswith(suffix):
                name_no_ext = name_no_ext[:-len(suffix)]
                break
        sample_id = name_no_ext

        mesh_texture_valid = 1.0
        try:
            mesh_texture = self._load_mesh_texture_map(
                data_root=data_root,
                sample_id=sample_id,
            )
            if mesh_texture is None:
                mesh_texture_valid = 0.0
        except Exception:
            mesh_texture = None
            mesh_texture_valid = 0.0

        mask_path = os.path.join(dir_name, f"{name_no_ext}_mask{ext}")
        if not os.path.exists(mask_path) and ext.lower() != '.png':
            mask_path = os.path.join(dir_name, f"{name_no_ext}_mask.png")

        if os.path.exists(mask_path):
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_img is not None:
                    if mask_img.ndim == 3 and mask_img.shape[2] == 4:
                        mask_alpha = mask_img[:, :, 3]
                    elif mask_img.ndim == 2:
                        mask_alpha = mask_img
                    else:
                        mask_alpha = mask_img[:, :, 0]

                    mh, mw = mask_alpha.shape[:2]
                    if mh != h or mw != w:
                        mask_alpha = cv2.resize(mask_alpha, (w, h), interpolation=cv2.INTER_NEAREST)

                    mask_bin = (mask_alpha > 0).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)

                    if landmarks is not None and landmark_weights is not None:
                        pts_norm = landmarks[:, 3:5]
                        pts_x = (pts_norm[:, 0] * w).astype(np.int32)
                        pts_y = (pts_norm[:, 1] * h).astype(np.int32)
                        for i in range(len(landmarks)):
                            px, py = pts_x[i], pts_y[i]
                            if 0 <= px < w and 0 <= py < h and mask_dilated[py, px] > 0:
                                landmark_weights[i] = 0.0

                    if mesh is not None and mesh_weights is not None:
                        pts_norm = mesh[:, 3:5]
                        pts_x = (pts_norm[:, 0] * w).astype(np.int32)
                        pts_y = (pts_norm[:, 1] * h).astype(np.int32)
                        for i in range(len(mesh)):
                            px, py = pts_x[i], pts_y[i]
                            if 0 <= px < w and 0 <= py < h and mask_dilated[py, px] > 0:
                                mesh_weights[i] = 0.0
            except Exception:
                pass

        # Load face mask before alignment so it can be warped alongside the GT images.
        facemask_np = None
        facemask_path = os.path.join(data_root, "facemask", f"Face_Mask_{sample_id}.png")
        if os.path.exists(facemask_path):
            try:
                fm = cv2.imread(facemask_path, cv2.IMREAD_UNCHANGED)
                if fm is not None:
                    if fm.ndim == 3 and fm.shape[2] == 4:
                        facemask_np = fm[:, :, 3]          # alpha channel
                    elif fm.ndim == 3:
                        facemask_np = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY)
                    else:
                        facemask_np = fm
            except Exception:
                facemask_np = None

        # Load geo GT before alignment so warpAffine can be applied uniformly.
        geo_np = None
        geo_valid = 0.0
        geo_path = os.path.join(data_root, "geo", f"Geo_{sample_id}.exr")
        if os.path.exists(geo_path):
            geo_np = _load_exr_as_float32(geo_path)
            if geo_np is not None:
                geo_valid = 1.0

        # MediaPipe face alignment with GT landmark fallback (same as dense_image_dataset).
        lm_px_gt = _load_landmark_pixels(landmark_path)
        lm6, detection_source = self.align_helper.detect_landmarks(image_np, fallback_lm_px=lm_px_gt)
        M = self.align_helper.estimate_alignment_matrix(lm6, src_w=w, src_h=h, split=self.split)
        image_np = cv2.warpAffine(
            image_np, M, (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        if basecolor_np is not None:
            basecolor_np = cv2.warpAffine(
                basecolor_np, M, (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
        if geo_np is not None:
            geo_np = cv2.warpAffine(
                geo_np, M, (self.image_size, self.image_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
        if facemask_np is not None:
            facemask_np = cv2.warpAffine(
                facemask_np, M, (self.image_size, self.image_size),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        landmarks = self.align_helper.apply_alignment_to_geometry(landmarks, m=M, src_w=w, src_h=h)
        mesh = self.align_helper.apply_alignment_to_geometry(mesh, m=M, src_w=w, src_h=h)
        h = self.image_size
        w = self.image_size
        # Filter using indices
        if landmarks is not None and self.landmark_indices is not None:
            if self.landmark_indices.max() < len(landmarks):
                landmarks = landmarks[self.landmark_indices]
                if landmarks_raw_xyz is not None and self.landmark_indices.max() < len(landmarks_raw_xyz):
                    landmarks_raw_xyz = landmarks_raw_xyz[self.landmark_indices]
                if landmark_found_mask is not None and self.landmark_indices.max() < len(landmark_found_mask):
                    landmark_found_mask = landmark_found_mask[self.landmark_indices]
                if landmark_weights is not None:
                    landmark_weights = landmark_weights[self.landmark_indices]

        if mesh is not None and self.mesh_indices is not None:
            if self.mesh_indices.max() < len(mesh):
                mesh = mesh[self.mesh_indices]
                if mesh_raw_xyz is not None and self.mesh_indices.max() < len(mesh_raw_xyz):
                    mesh_raw_xyz = mesh_raw_xyz[self.mesh_indices]
                if mesh_found_mask is not None and self.mesh_indices.max() < len(mesh_found_mask):
                    mesh_found_mask = mesh_found_mask[self.mesh_indices]
                if mesh_weights is not None:
                    mesh_weights = mesh_weights[self.mesh_indices]

        # Guard against malformed per-sample geometry lengths.
        if landmarks is None or landmarks.shape[0] != self.default_landmarks.shape[0]:
            landmarks = self.default_landmarks.copy()
            landmarks_raw_xyz = None
            landmark_found_mask = np.zeros((landmarks.shape[0],), dtype=bool)
            landmark_weights = np.zeros((landmarks.shape[0],), dtype=np.float32)
        if mesh is None or mesh.shape[0] != self.default_mesh.shape[0]:
            mesh = self.default_mesh.copy()
            mesh_raw_xyz = None
            mesh_found_mask = np.zeros((mesh.shape[0],), dtype=bool)
            mesh_weights = np.zeros((mesh.shape[0],), dtype=np.float32)

        # Image-only augmentation path (used with fixed 5-point alignment).
        if self.image_only_augment is not None and ALBUMENTATIONS_AVAILABLE:
            image_np = self.image_only_augment(image=image_np)['image']
        elif self.image_only_augment_tv is not None:
            img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            img_tensor = self.image_only_augment_tv(img_tensor)
            image_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)

        # Resize image if no augmentation was applied (val split)
        if image_np.shape[0] != self.image_size or image_np.shape[1] != self.image_size:
            image_np = cv2.resize(image_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if basecolor_np is None:
            basecolor_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        elif basecolor_np.shape[0] != self.image_size or basecolor_np.shape[1] != self.image_size:
            basecolor_np = cv2.resize(basecolor_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if geo_np is None:
            geo_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        elif geo_np.shape[0] != self.image_size or geo_np.shape[1] != self.image_size:
            geo_np = cv2.resize(geo_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        if facemask_np is None:
            facemask_np = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 255
        elif facemask_np.shape[0] != self.image_size or facemask_np.shape[1] != self.image_size:
            facemask_np = cv2.resize(facemask_np, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Apply face mask to GT images (zero out pixels outside the face region)
        mask_bin = (facemask_np > 127).astype(np.uint8)   # [H, W] binary
        mask3 = mask_bin[:, :, np.newaxis]                 # [H, W, 1] for broadcasting
        basecolor_np = basecolor_np * mask3
        geo_np       = geo_np       * mask3.astype(np.float32, copy=False)

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        basecolor_tensor = torch.from_numpy(basecolor_np).permute(2, 0, 1).float() / 255.0

        # Load matrix data and compute z-depth for mesh
        mesh_depth = None
        landmark_depth = None
        mat_path = os.path.join(data_root, "mat", f"Mats_{sample_id}.txt")
        if os.path.exists(mat_path):
            try:
                matrix_data = load_matrix_data(mat_path)

                # Compute depth for landmarks
                if landmarks is not None:
                    if landmarks_raw_xyz is not None and landmarks_raw_xyz.shape[0] == landmarks.shape[0]:
                        landmark_xyz_original = landmarks_raw_xyz.copy()
                    else:
                        landmark_xyz_original = landmarks[:, 0:3].copy()
                    landmark_depth_raw = compute_vertex_depth(landmark_xyz_original, matrix_data)

                    # Validate raw depth values
                    if np.any(~np.isfinite(landmark_depth_raw)):
                        print(f"[DEPTH WARNING] {sample_id}: landmark_depth_raw contains NaN/inf")
                        print(f"  Path: {color_path}")
                        print(f"  min={np.nanmin(landmark_depth_raw):.4f}, max={np.nanmax(landmark_depth_raw):.4f}")
                        landmark_depth = np.zeros_like(landmark_depth_raw)
                    else:
                        depth_min = landmark_depth_raw.min()
                        depth_max = landmark_depth_raw.max()
                        depth_range = depth_max - depth_min

                        # Log if depth values are behind camera or range is too small
                        if depth_min < 0:
                            print(f"[DEPTH WARNING] {sample_id}: landmark depth_min={depth_min:.4f} < 0 (behind camera)")
                            print(f"  Path: {color_path}")
                        if depth_range < 1e-6:
                            print(f"[DEPTH WARNING] {sample_id}: landmark depth_range={depth_range:.8f} (too small, using zeros)")
                            print(f"  Path: {color_path}")
                            landmark_depth = np.zeros_like(landmark_depth_raw)
                        else:
                            # Normalize to [0, 1] per sample with epsilon for safety
                            landmark_depth = (landmark_depth_raw - depth_min) / (depth_range + 1e-8)
                            landmark_depth = np.clip(landmark_depth, 0.0, 1.0)
                            # Convert to offset from template depth
                            landmark_depth = landmark_depth - self.template_landmark_depth

                            # Final validation
                            if np.any(~np.isfinite(landmark_depth)):
                                print(f"[DEPTH ERROR] {sample_id}: landmark_depth contains NaN/inf after normalization!")
                                landmark_depth = np.zeros_like(landmark_depth_raw)

                    # Add depth as 6th column to landmarks
                    landmarks = np.concatenate([landmarks, landmark_depth[:, None]], axis=1)

                # Compute depth for mesh
                if mesh is not None:
                    if mesh_raw_xyz is not None and mesh_raw_xyz.shape[0] == mesh.shape[0]:
                        mesh_xyz_original = mesh_raw_xyz.copy()
                    else:
                        mesh_xyz_original = mesh[:, 0:3].copy()
                    mesh_depth_raw = compute_vertex_depth(mesh_xyz_original, matrix_data)

                    # Validate raw depth values
                    if np.any(~np.isfinite(mesh_depth_raw)):
                        print(f"[DEPTH WARNING] {sample_id}: mesh_depth_raw contains NaN/inf")
                        print(f"  Path: {color_path}")
                        print(f"  min={np.nanmin(mesh_depth_raw):.4f}, max={np.nanmax(mesh_depth_raw):.4f}")
                        mesh_depth_normalized = np.zeros_like(mesh_depth_raw)
                        mesh_depth = None
                    else:
                        depth_min = mesh_depth_raw.min()
                        depth_max = mesh_depth_raw.max()
                        depth_range = depth_max - depth_min

                        # Log if depth values are behind camera or range is too small
                        if depth_min < 0:
                            print(f"[DEPTH WARNING] {sample_id}: mesh depth_min={depth_min:.4f} < 0 (behind camera)")
                            print(f"  Path: {color_path}")
                        if depth_range < 1e-6:
                            print(f"[DEPTH WARNING] {sample_id}: mesh depth_range={depth_range:.8f} (too small, using zeros)")
                            print(f"  Path: {color_path}")
                            mesh_depth_normalized = np.zeros_like(mesh_depth_raw)
                            mesh_depth = None
                        else:
                            # Normalize to [0, 1] per sample with epsilon for safety
                            mesh_depth_normalized = (mesh_depth_raw - depth_min) / (depth_range + 1e-8)
                            mesh_depth_normalized = np.clip(mesh_depth_normalized, 0.0, 1.0)
                            # Convert to offset from template depth
                            mesh_depth_normalized = mesh_depth_normalized - self.template_mesh_depth

                            # Final validation
                            if np.any(~np.isfinite(mesh_depth_normalized)):
                                print(f"[DEPTH ERROR] {sample_id}: mesh_depth contains NaN/inf after normalization!")
                                mesh_depth_normalized = np.zeros_like(mesh_depth_raw)
                                mesh_depth = None
                            else:
                                # Keep raw depth for rendering
                                mesh_depth = mesh_depth_raw

                    # Add depth as 6th column to mesh
                    mesh = np.concatenate([mesh, mesh_depth_normalized[:, None]], axis=1)
            except Exception:
                # If depth computation fails, pad with zeros
                if landmarks is not None:
                    landmarks = np.concatenate([landmarks, np.zeros((landmarks.shape[0], 1), dtype=np.float32)], axis=1)
                if mesh is not None:
                    mesh = np.concatenate([mesh, np.zeros((mesh.shape[0], 1), dtype=np.float32)], axis=1)
                mesh_depth = None
        else:
            # No matrix file, pad with zeros
            if landmarks is not None and landmarks.shape[1] < 6:
                landmarks = np.concatenate([landmarks, np.zeros((landmarks.shape[0], 1), dtype=np.float32)], axis=1)
            if mesh is not None and mesh.shape[1] < 6:
                mesh = np.concatenate([mesh, np.zeros((mesh.shape[0], 1), dtype=np.float32)], axis=1)

        result = {
            'rgb': rgb_tensor,
            'basecolor': basecolor_tensor,
            'basecolor_valid': torch.tensor(1.0 if float(basecolor_valid) > 0.5 else 0.0, dtype=torch.float32),
            'image_path': color_path,
            'landmarks': torch.from_numpy(landmarks).float(),
            'landmark_found_mask': torch.from_numpy(landmark_found_mask.astype(np.float32, copy=False)),
            'landmark_weights': torch.from_numpy(landmark_weights).float(),
            'mesh': torch.from_numpy(mesh).float(),
            'mesh_found_mask': torch.from_numpy(mesh_found_mask.astype(np.float32, copy=False)),
            'mesh_weights': torch.from_numpy(mesh_weights).float(),
        }

        if mesh_depth is not None:
            result['mesh_depth'] = torch.from_numpy(mesh_depth.astype(np.float32))

        if mesh_texture is None:
            mesh_texture = np.zeros((int(self._mesh_texture_size), int(self._mesh_texture_size), 3), dtype=np.float32)
            if mesh_texture_valid is None:
                mesh_texture_valid = 0.0
        result['mesh_texture'] = torch.from_numpy(mesh_texture).permute(2, 0, 1).float()
        result['mesh_texture_valid'] = torch.tensor(
            1.0 if float(mesh_texture_valid or 0.0) > 0.5 else 0.0,
            dtype=torch.float32,
        )

        result['geo_gt'] = torch.from_numpy(geo_np.astype(np.float32, copy=False)).permute(2, 0, 1)
        result['geo_valid'] = torch.tensor(1.0 if geo_valid > 0.5 else 0.0, dtype=torch.float32)

        key5_px, key5_valid = self.align_helper.extract_key5_from_lm68(lm6)
        key5_out = self.align_helper.transform_points_px(key5_px, M) / float(max(1, self.image_size))
        key5_out = np.nan_to_num(key5_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        result['align5pts'] = torch.from_numpy(key5_out)
        result['align5pts_valid'] = torch.from_numpy(key5_valid)
        result['align_applied'] = torch.tensor(1.0, dtype=torch.float32)

        return result


def fast_collate_fn(batch):
    """Custom collate function."""
    collated = {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'basecolor': torch.stack([item['basecolor'] for item in batch]),
        'basecolor_valid': torch.stack([item['basecolor_valid'] for item in batch]),
        'landmarks': torch.stack([item['landmarks'] for item in batch]),
        'landmark_found_mask': torch.stack([item['landmark_found_mask'] for item in batch]),
        'landmark_weights': torch.stack([item['landmark_weights'] for item in batch]),
        'mesh': torch.stack([item['mesh'] for item in batch]),
        'mesh_found_mask': torch.stack([item['mesh_found_mask'] for item in batch]),
        'mesh_weights': torch.stack([item['mesh_weights'] for item in batch]),
        'mesh_texture': torch.stack([item['mesh_texture'] for item in batch]),
        'mesh_texture_valid': torch.stack([item['mesh_texture_valid'] for item in batch]),
        'geo_gt': torch.stack([item['geo_gt'] for item in batch]),
        'geo_valid': torch.stack([item['geo_valid'] for item in batch]),
    }

    if all('mesh_depth' in item for item in batch):
        collated['mesh_depth'] = torch.stack([item['mesh_depth'] for item in batch])
    if all('align5pts' in item for item in batch):
        collated['align5pts'] = torch.stack([item['align5pts'] for item in batch])
    if all('align5pts_valid' in item for item in batch):
        collated['align5pts_valid'] = torch.stack([item['align5pts_valid'] for item in batch])
    if all('align_applied' in item for item in batch):
        collated['align_applied'] = torch.stack([item['align_applied'] for item in batch])

    if 'image_path' in batch[0]:
        collated['image_path'] = [item['image_path'] for item in batch]

    return collated


def create_fast_geometry_dataloaders(
    data_roots,
    batch_size: int = 24,
    num_workers: int = 8,
    image_size: int = 512,
    train_ratio: float = 0.8,
    prefetch_factor: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized train and validation dataloaders."""
    train_dataset = FastGeometryDataset(
        data_roots=data_roots,
        split='train',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=True,
    )
    
    val_dataset = FastGeometryDataset(
        data_roots=data_roots,
        split='val',
        image_size=image_size,
        train_ratio=train_ratio,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=fast_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=fast_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    return train_loader, val_loader








