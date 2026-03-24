"""
Fast RGB+Geometry Dataset for Geometry Training (Landmark + Mesh)
Loads RGB images, landmarks, and mesh vertices with aligned normalization
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, Optional
import cv2

from align_5pt_helper import Align5PtHelper
from alignment_io import (
    apply_alignment_to_geometry,
    build_alignment_metadata,
    compute_alignment_transform,
    warp_mask_image,
    warp_rgb_image,
)
from image_io import (
    apply_face_mask_to_modalities,
    apply_occlusion_mask_to_weights,
    load_exr_as_float32 as _shared_load_exr_as_float32,
    load_optional_mask,
    load_rgb_image,
    load_rgb_image_or_default,
    resize_float3_or_default,
    resize_mask_or_default,
    resize_rgb_or_default,
)
from mesh_io import (
    GeometryTemplateSet,
    apply_geometry_indices,
    attach_depth_channels,
    compute_geometry_found_mask,
    ensure_geometry_with_fallback,
    load_geometry_template_set,
    load_geometry_txt,
    load_landmark_pixels as _shared_load_landmark_pixels,
)
from sample_index import collect_geometry_sample_records
from tex_pack_helper import TexturePackHelper

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
    """Compatibility wrapper for shared landmark pixel loading."""
    return _shared_load_landmark_pixels(filepath)


def _load_exr_as_float32(path: str) -> Optional[np.ndarray]:
    """Compatibility wrapper for shared EXR loading."""
    return _shared_load_exr_as_float32(path)


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
        load_basecolor: bool = True,
        load_mesh_texture: bool = True,
        load_geo_gt: bool = True,
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
            load_basecolor: Load warped basecolor GT maps
            load_mesh_texture: Load packed mesh texture GT maps
            load_geo_gt: Load warped geo GT maps

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
        self.load_basecolor = bool(load_basecolor)
        self.load_mesh_texture = bool(load_mesh_texture)
        self.load_geo_gt = bool(load_geo_gt)

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
        self.template_landmark_depth = None
        self.template_mesh_depth = None
        self._load_default_templates()

        self.samples = collect_geometry_sample_records(
            data_roots=self.data_roots,
            split=split,
            train_ratio=train_ratio,
        )

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
        templates: GeometryTemplateSet = load_geometry_template_set(model_dir="model")
        self.landmark_indices = templates.landmark_indices
        self.mesh_indices = templates.mesh_indices
        self.default_landmarks = templates.default_landmarks
        self.default_mesh = templates.default_mesh
        self.template_landmark_depth = templates.template_landmark_depth
        self.template_mesh_depth = templates.template_mesh_depth

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
        """Compatibility wrapper for shared geometry loading."""
        return load_geometry_txt(filepath, return_raw_xyz=return_raw_xyz)

    @staticmethod
    def _compute_geometry_found_mask(geom: Optional[np.ndarray]) -> np.ndarray:
        return compute_geometry_found_mask(geom)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        data_root = record.data_root
        color_path = record.color_path
        landmark_path = record.landmark_path
        mesh_path = record.mesh_path
        basecolor_path = record.basecolor_path
        sample_id = record.sample_id

        # Load RGB
        image_np = load_rgb_image_or_default(color_path, image_size=self.image_size)

        h, w = image_np.shape[:2]

        basecolor_np = None
        basecolor_valid = 0.0
        if self.load_basecolor:
            basecolor_np = load_rgb_image(basecolor_path) if basecolor_path is not None else None
            if basecolor_np is not None:
                basecolor_valid = 1.0
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

        if self.load_mesh_texture:
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

        mask_alpha = load_optional_mask(record.mask_path)
        landmark_weights = apply_occlusion_mask_to_weights(
            geometry=landmarks,
            weights=landmark_weights,
            mask_alpha=mask_alpha,
            width=w,
            height=h,
        )
        mesh_weights = apply_occlusion_mask_to_weights(
            geometry=mesh,
            weights=mesh_weights,
            mask_alpha=mask_alpha,
            width=w,
            height=h,
        )

        # Load face mask before alignment so it can be warped alongside the GT images.
        facemask_np = None
        if self.load_basecolor or self.load_geo_gt:
            facemask_np = load_optional_mask(record.facemask_path)

        # Load geo GT before alignment so warpAffine can be applied uniformly.
        geo_np = None
        geo_valid = 0.0
        if self.load_geo_gt:
            geo_np = _load_exr_as_float32(record.geo_path)
            if geo_np is not None:
                geo_valid = 1.0

        # MediaPipe face alignment with GT landmark fallback (same as dense_image_dataset).
        lm6, detection_source, M = compute_alignment_transform(
            image_np=image_np,
            landmark_path=landmark_path,
            align_helper=self.align_helper,
            split=self.split,
        )
        image_np = warp_rgb_image(image_np, M, self.image_size)
        if basecolor_np is not None:
            basecolor_np = warp_rgb_image(basecolor_np, M, self.image_size)
        if geo_np is not None:
            geo_np = warp_rgb_image(geo_np, M, self.image_size)
        if facemask_np is not None:
            facemask_np = warp_mask_image(facemask_np, M, self.image_size)
        landmarks = apply_alignment_to_geometry(self.align_helper, landmarks, M, src_w=w, src_h=h)
        mesh = apply_alignment_to_geometry(self.align_helper, mesh, M, src_w=w, src_h=h)
        h = self.image_size
        w = self.image_size
        # Filter using indices
        landmarks, landmarks_raw_xyz, landmark_found_mask, landmark_weights = apply_geometry_indices(
            geometry=landmarks,
            raw_xyz=landmarks_raw_xyz,
            found_mask=landmark_found_mask,
            weights=landmark_weights,
            indices=self.landmark_indices,
        )
        mesh, mesh_raw_xyz, mesh_found_mask, mesh_weights = apply_geometry_indices(
            geometry=mesh,
            raw_xyz=mesh_raw_xyz,
            found_mask=mesh_found_mask,
            weights=mesh_weights,
            indices=self.mesh_indices,
        )

        # Guard against malformed per-sample geometry lengths.
        landmarks, landmarks_raw_xyz, landmark_found_mask, landmark_weights = ensure_geometry_with_fallback(
            geometry=landmarks,
            raw_xyz=landmarks_raw_xyz,
            found_mask=landmark_found_mask,
            weights=landmark_weights,
            fallback_geometry=self.default_landmarks,
        )
        mesh, mesh_raw_xyz, mesh_found_mask, mesh_weights = ensure_geometry_with_fallback(
            geometry=mesh,
            raw_xyz=mesh_raw_xyz,
            found_mask=mesh_found_mask,
            weights=mesh_weights,
            fallback_geometry=self.default_mesh,
        )

        # Image-only augmentation path (used with fixed 5-point alignment).
        if self.image_only_augment is not None and ALBUMENTATIONS_AVAILABLE:
            image_np = self.image_only_augment(image=image_np)['image']
        elif self.image_only_augment_tv is not None:
            img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            img_tensor = self.image_only_augment_tv(img_tensor)
            image_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)

        # Resize image if no augmentation was applied (val split)
        image_np = resize_rgb_or_default(image_np, self.image_size)
        if self.load_basecolor:
            basecolor_np = resize_rgb_or_default(basecolor_np, self.image_size)
        if self.load_geo_gt:
            geo_np = resize_float3_or_default(geo_np, self.image_size)
        if self.load_basecolor or self.load_geo_gt:
            facemask_np = resize_mask_or_default(facemask_np, self.image_size, default_value=255)
            basecolor_np, geo_np = apply_face_mask_to_modalities(
                facemask=facemask_np,
                basecolor=basecolor_np,
                geo=geo_np,
            )

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        # Load matrix data and compute z-depth for mesh
        landmarks, mesh, mesh_depth = attach_depth_channels(
            landmarks=landmarks,
            landmarks_raw_xyz=landmarks_raw_xyz,
            mesh=mesh,
            mesh_raw_xyz=mesh_raw_xyz,
            mat_path=record.mat_path,
            template_landmark_depth=self.template_landmark_depth,
            template_mesh_depth=self.template_mesh_depth,
            sample_id=sample_id,
            color_path=color_path,
        )

        result = {
            'rgb': rgb_tensor,
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

        if self.load_basecolor:
            basecolor_tensor = torch.from_numpy(basecolor_np).permute(2, 0, 1).float() / 255.0
            result['basecolor'] = basecolor_tensor
            result['basecolor_valid'] = torch.tensor(1.0 if float(basecolor_valid) > 0.5 else 0.0, dtype=torch.float32)

        if self.load_mesh_texture:
            if mesh_texture is None:
                mesh_texture = np.zeros((int(self._mesh_texture_size), int(self._mesh_texture_size), 3), dtype=np.float32)
                if mesh_texture_valid is None:
                    mesh_texture_valid = 0.0
            result['mesh_texture'] = torch.from_numpy(mesh_texture).permute(2, 0, 1).float()
            result['mesh_texture_valid'] = torch.tensor(
                1.0 if float(mesh_texture_valid or 0.0) > 0.5 else 0.0,
                dtype=torch.float32,
            )

        if self.load_geo_gt:
            result['geo_gt'] = torch.from_numpy(geo_np.astype(np.float32, copy=False)).permute(2, 0, 1)
            result['geo_valid'] = torch.tensor(1.0 if geo_valid > 0.5 else 0.0, dtype=torch.float32)

        key5_out, key5_valid = build_alignment_metadata(
            align_helper=self.align_helper,
            lm6=lm6,
            transform=M,
            image_size=self.image_size,
        )
        result['align5pts'] = torch.from_numpy(key5_out)
        result['align5pts_valid'] = torch.from_numpy(key5_valid)
        result['align_applied'] = torch.tensor(1.0, dtype=torch.float32)
        result['detection_source'] = detection_source

        return result


def fast_collate_fn(batch):
    """Custom collate function."""
    collated = {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'landmarks': torch.stack([item['landmarks'] for item in batch]),
        'landmark_found_mask': torch.stack([item['landmark_found_mask'] for item in batch]),
        'landmark_weights': torch.stack([item['landmark_weights'] for item in batch]),
        'mesh': torch.stack([item['mesh'] for item in batch]),
        'mesh_found_mask': torch.stack([item['mesh_found_mask'] for item in batch]),
        'mesh_weights': torch.stack([item['mesh_weights'] for item in batch]),
    }

    if all('basecolor' in item for item in batch):
        collated['basecolor'] = torch.stack([item['basecolor'] for item in batch])
    if all('basecolor_valid' in item for item in batch):
        collated['basecolor_valid'] = torch.stack([item['basecolor_valid'] for item in batch])
    if all('mesh_texture' in item for item in batch):
        collated['mesh_texture'] = torch.stack([item['mesh_texture'] for item in batch])
    if all('mesh_texture_valid' in item for item in batch):
        collated['mesh_texture_valid'] = torch.stack([item['mesh_texture_valid'] for item in batch])
    if all('geo_gt' in item for item in batch):
        collated['geo_gt'] = torch.stack([item['geo_gt'] for item in batch])
    if all('geo_valid' in item for item in batch):
        collated['geo_valid'] = torch.stack([item['geo_valid'] for item in batch])
    if all('mesh_depth' in item for item in batch):
        collated['mesh_depth'] = torch.stack([item['mesh_depth'] for item in batch])
    if all('align5pts' in item for item in batch):
        collated['align5pts'] = torch.stack([item['align5pts'] for item in batch])
    if all('align5pts_valid' in item for item in batch):
        collated['align5pts_valid'] = torch.stack([item['align5pts_valid'] for item in batch])
    if all('align_applied' in item for item in batch):
        collated['align_applied'] = torch.stack([item['align_applied'] for item in batch])
    if all('detection_source' in item for item in batch):
        collated['detection_source'] = [item['detection_source'] for item in batch]

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








