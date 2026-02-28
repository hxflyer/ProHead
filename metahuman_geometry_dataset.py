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
from typing import List, Tuple, Optional
import cv2

from align_5pt_helper import Align5PtHelper
from tex_pack_helper import TexturePackHelper

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("Warning: Albumentations not installed. Using basic augmentation. Install with: pip install albumentations")


class FastGeometryDataset(Dataset):
    """
    Fast dataset for RGB geometry training (landmark + mesh).
    Loads RGB images, landmarks, and mesh vertices.
    """

    # 5-point alignment is always enabled with fixed settings.
    ALIGN_5PT_INDICES = (2219, 2194, 1993, 1896, 1839)
    ALIGN_5PT_JITTER_STD = 0.003
    ALIGN_5PT_OUTPUT_SCALE = 0.75
    ALIGN_5PT_DIRECTION_SHIFT = 0.08
    
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
        """
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = data_roots
            
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        self.texture_root = texture_root
        self.align_5pt_indices = np.asarray(list(self.ALIGN_5PT_INDICES), dtype=np.int64)
        if self.align_5pt_indices.shape[0] != 5:
            raise ValueError("align_5pt_indices must contain exactly 5 indices: [eye_r, eye_l, nose, mouth_r, mouth_l]")
        self.align_5pt_jitter_std = float(max(0.0, self.ALIGN_5PT_JITTER_STD))
        self.align_5pt_output_scale = float(np.clip(self.ALIGN_5PT_OUTPUT_SCALE, 0.1, 2.0))
        self.align_5pt_direction_shift = float(np.clip(self.ALIGN_5PT_DIRECTION_SHIFT, 0.0, 0.5))
        # Canonical 5-point targets in normalized image space [0, 1]
        # Order: [eye_r, eye_l, nose, mouth_r, mouth_l]
        self.align_5pt_target_norm = np.array([
            [0.35, 0.38],
            [0.65, 0.38],
            [0.50, 0.56],
            [0.40, 0.72],
            [0.60, 0.72],
        ], dtype=np.float32)

        self.align_helper = Align5PtHelper(
            image_size=self.image_size,
            indices=self.align_5pt_indices,
            target_norm=self.align_5pt_target_norm,
            jitter_std=self.align_5pt_jitter_std,
            output_scale=self.align_5pt_output_scale,
            direction_shift=self.align_5pt_direction_shift,
        )
        self.texture_packer = TexturePackHelper(
            texture_root=self.texture_root,
            texture_png_cache_max_items=texture_png_cache_max_items,
            combined_texture_cache_max_items=combined_texture_cache_max_items,
        )
        self._mesh_texture_size = int(self.texture_packer.mesh_texture_size)
        
        # Find all samples
        self.samples = []  # List of (data_root, color_path, landmark_path, mesh_path)
        
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                print(f"Warning: Data directory {data_root} does not exist, skipping...")
                continue
            
            # Find Color files
            color_files = glob.glob(os.path.join(data_root, "Color_*"))
            color_files = [f for f in color_files if os.path.basename(f).startswith("Color_")]
            
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
                
                # Find corresponding files
                landmark_path = self._find_file(data_root, "landmark", base_id)
                mesh_path = self._find_file(data_root, "mesh", base_id)
                
                # Only include samples that have both landmark and mesh
                if landmark_path and os.path.exists(landmark_path) and mesh_path and os.path.exists(mesh_path):
                    self.samples.append((data_root, color_file, landmark_path, mesh_path))
        
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
        
        print(f"FastGeometryDataset [{split}]: {len(self.samples)} samples")
        
        # Augmentation transforms (image-only; geometric augment is disabled).
        self.image_only_augment = None
        self.image_only_augment_tv = None

        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.image_only_augment = A.Compose([
                A.ISONoise(
                    color_shift=(0.01, 0.05),
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

        print(
            f"FastGeometryDataset [{split}]: fixed 5-point alignment "
            f"(jitter_std={self.align_5pt_jitter_std}, output_scale={self.align_5pt_output_scale}, "
            f"direction_shift={self.align_5pt_direction_shift}). Geometric aug disabled."
        )

    def _get_texture_root(self, data_root: str) -> Optional[str]:
        return self.texture_packer.get_texture_root(data_root)

    def _find_mats_file(self, data_root: str, sample_id: str) -> Optional[str]:
        return self.texture_packer.find_mats_file(data_root, sample_id)

    def _load_mesh_texture_map(self, data_root: str, sample_id: str) -> Optional[np.ndarray]:
        return self.texture_packer.load_mesh_texture_map(data_root, sample_id)

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
    
    def _load_geometry(self, filepath: str) -> Optional[np.ndarray]:
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
                return None

            # Instance Normalization (aligned via build_template)
            xyz = geom[:, 0:3]
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            center = (min_xyz + max_xyz) / 2
            scale = np.max(max_xyz - min_xyz) / 2
            
            if scale > 0:
                geom[:, 0:3] = (xyz - center) / scale
            
            # Normalize 2D coords
            geom[:, 3:5] = geom[:, 3:5] / 1024.0
            
            return geom
        except Exception as e:
            print(f"Error loading geometry {filepath}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)

    def _extract_five_points_px(self, landmarks: Optional[np.ndarray], w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.align_helper.extract_five_points_px(landmarks, w=w, h=h)

    def _compute_head_center_px(self, landmarks: Optional[np.ndarray], w: int, h: int) -> Optional[np.ndarray]:
        return self.align_helper.compute_head_center_px(landmarks, w=w, h=h)

    def _is_extreme_pose(self, five_pts_px: np.ndarray, five_valid: np.ndarray) -> bool:
        return self.align_helper.is_extreme_pose(five_pts_px, five_valid)

    def _estimate_face_direction(self, five_pts_px: np.ndarray, five_valid: np.ndarray) -> np.ndarray:
        return self.align_helper.estimate_face_direction(five_pts_px, five_valid)

    def _estimate_alignment_matrix(
        self,
        five_pts_px: np.ndarray,
        five_valid: np.ndarray,
        src_w: int,
        src_h: int,
        head_center_px: Optional[np.ndarray] = None,
        is_extreme_pose: bool = False,
        face_direction: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        return self.align_helper.estimate_alignment_matrix(
            five_pts_px=five_pts_px,
            five_valid=five_valid,
            src_w=src_w,
            src_h=src_h,
            split=self.split,
            head_center_px=head_center_px,
            is_extreme_pose=is_extreme_pose,
            face_direction=face_direction,
        )

    def _transform_points_px(self, points_px: np.ndarray, m: np.ndarray) -> np.ndarray:
        return self.align_helper.transform_points_px(points_px, m)

    def _apply_alignment_to_geometry(self, geom: Optional[np.ndarray], M: np.ndarray, src_w: int, src_h: int) -> Optional[np.ndarray]:
        return self.align_helper.apply_alignment_to_geometry(geom=geom, m=M, src_w=src_w, src_h=src_h)

    def __getitem__(self, idx: int):
        data_root, color_path, landmark_path, mesh_path = self.samples[idx]
        
        # Load indices if not already loaded
        if not hasattr(self, 'landmark_indices'):
            path = "model/landmark_indices.npy"
            self.landmark_indices = np.load(path) if os.path.exists(path) else None
        
        if not hasattr(self, 'mesh_indices'):
            path = "model/mesh_indices.npy"
            self.mesh_indices = np.load(path) if os.path.exists(path) else None

        # Load RGB
        try:
            image_np = cv2.imread(color_path)
            if image_np is None:
                raise ValueError(f"Could not read {color_path}")
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {color_path}: {e}")
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        h, w = image_np.shape[:2]
        
        # Load geometry
        landmarks = self._load_geometry(landmark_path)
        mesh = self._load_geometry(mesh_path)
        mesh_texture = None
        mesh_texture_valid = None
        five_points_px = None
        five_points_valid = None
        head_center_px = None
        is_extreme_pose = False
        face_direction = None
        align_applied = False
        five_points_px, five_points_valid = self._extract_five_points_px(landmarks, w=w, h=h)
        head_center_px = self._compute_head_center_px(landmarks, w=w, h=h)
        is_extreme_pose = self._is_extreme_pose(five_points_px, five_points_valid)
        face_direction = self._estimate_face_direction(five_points_px, five_points_valid)

        # Mask-based weights
        landmark_weights = None
        mesh_weights = None
        
        if landmarks is not None:
            landmark_weights = np.ones((landmarks.shape[0],), dtype=np.float32)
        if mesh is not None:
            mesh_weights = np.ones((mesh.shape[0],), dtype=np.float32)
        
        # Load mask
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

        if mesh is not None:
            mesh_texture_valid = 1.0
            try:
                mesh_texture = self._load_mesh_texture_map(
                    data_root=data_root,
                    sample_id=sample_id,
                )
                if mesh_texture is None:
                    mesh_texture_valid = 0.0
            except Exception as e:
                print(
                    f"[MeshTextureDebug] EXCEPTION sample_id={sample_id} "
                    f"data_root={data_root} during mesh texture load: {e}"
                )
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
                    
                    # Apply to landmarks
                    if landmarks is not None and landmark_weights is not None:
                        pts_norm = landmarks[:, 3:5]
                        pts_x = (pts_norm[:, 0] * w).astype(np.int32)
                        pts_y = (pts_norm[:, 1] * h).astype(np.int32)
                        
                        for i in range(len(landmarks)):
                            px, py = pts_x[i], pts_y[i]
                            if 0 <= px < w and 0 <= py < h:
                                if mask_dilated[py, px] > 0:
                                    landmark_weights[i] = 0.0
                    
                    # Apply to mesh
                    if mesh is not None and mesh_weights is not None:
                        pts_norm = mesh[:, 3:5]
                        pts_x = (pts_norm[:, 0] * w).astype(np.int32)
                        pts_y = (pts_norm[:, 1] * h).astype(np.int32)
                        
                        for i in range(len(mesh)):
                            px, py = pts_x[i], pts_y[i]
                            if 0 <= px < w and 0 <= py < h:
                                if mask_dilated[py, px] > 0:
                                    mesh_weights[i] = 0.0
            except Exception as e:
                print(f"Error processing mask {mask_path}: {e}")

        # 5-point alignment before filtering/augmentation so all geometry stays coherent.
        if landmarks is not None and five_points_px is not None:
            M = self._estimate_alignment_matrix(
                five_points_px,
                five_points_valid,
                src_w=w,
                src_h=h,
                head_center_px=head_center_px,
                is_extreme_pose=is_extreme_pose,
                face_direction=face_direction,
            )
            if M is not None:
                image_np = cv2.warpAffine(
                    image_np,
                    M,
                    (self.image_size, self.image_size),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                landmarks = self._apply_alignment_to_geometry(landmarks, M=M, src_w=w, src_h=h)
                mesh = self._apply_alignment_to_geometry(mesh, M=M, src_w=w, src_h=h)
                five_points_px = self._transform_points_px(five_points_px, M)
                h = self.image_size
                w = self.image_size
                align_applied = True
        
        # Filter using indices
        if landmarks is not None and self.landmark_indices is not None:
            if self.landmark_indices.max() < len(landmarks):
                landmarks = landmarks[self.landmark_indices]
                if landmark_weights is not None:
                    landmark_weights = landmark_weights[self.landmark_indices]
        
        if mesh is not None and self.mesh_indices is not None:
            if self.mesh_indices.max() < len(mesh):
                mesh = mesh[self.mesh_indices]
                if mesh_weights is not None:
                    mesh_weights = mesh_weights[self.mesh_indices]
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
                    
        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        result = {
            'rgb': rgb_tensor,
            'image_path': color_path,
        }
        
        if landmarks is not None:
            result['landmarks'] = torch.from_numpy(landmarks).float()
            if landmark_weights is not None:
                result['landmark_weights'] = torch.from_numpy(landmark_weights).float()
        
        if mesh is not None:
            result['mesh'] = torch.from_numpy(mesh).float()
            if mesh_weights is not None:
                result['mesh_weights'] = torch.from_numpy(mesh_weights).float()
            if mesh_texture is None:
                mesh_texture = np.zeros((int(self._mesh_texture_size), int(self._mesh_texture_size), 3), dtype=np.float32)
                if mesh_texture_valid is None:
                    mesh_texture_valid = 0.0
            result['mesh_texture'] = torch.from_numpy(mesh_texture).permute(2, 0, 1).float()
            result['mesh_texture_valid'] = torch.tensor(
                1.0 if float(mesh_texture_valid or 0.0) > 0.5 else 0.0,
                dtype=torch.float32,
            )

        if five_points_px is not None and five_points_valid is not None:
            if align_applied:
                denom = np.array([float(self.image_size), float(self.image_size)], dtype=np.float32)
            else:
                denom = np.array([float(max(1, w)), float(max(1, h))], dtype=np.float32)
            five_points_norm = five_points_px / denom[None, :]
            result['align5pts'] = torch.from_numpy(five_points_norm.astype(np.float32))
            result['align5pts_valid'] = torch.from_numpy(five_points_valid.astype(np.float32))
            result['align_applied'] = torch.tensor(1.0 if align_applied else 0.0, dtype=torch.float32)
            result['align_is_extreme'] = torch.tensor(1.0 if is_extreme_pose else 0.0, dtype=torch.float32)
        
        return result


def fast_collate_fn(batch):
    """Custom collate function."""
    collated = {'rgb': torch.stack([item['rgb'] for item in batch])}
    
    if all('landmarks' in item for item in batch):
        collated['landmarks'] = torch.stack([item['landmarks'] for item in batch])
    if all('landmark_weights' in item for item in batch):
        collated['landmark_weights'] = torch.stack([item['landmark_weights'] for item in batch])
    
    if all('mesh' in item for item in batch):
        collated['mesh'] = torch.stack([item['mesh'] for item in batch])
    if all('mesh_weights' in item for item in batch):
        collated['mesh_weights'] = torch.stack([item['mesh_weights'] for item in batch])
    if all('mesh_texture' in item for item in batch):
        collated['mesh_texture'] = torch.stack([item['mesh_texture'] for item in batch])
    if all('mesh_texture_valid' in item for item in batch):
        collated['mesh_texture_valid'] = torch.stack([item['mesh_texture_valid'] for item in batch])

    if all('align5pts' in item for item in batch):
        collated['align5pts'] = torch.stack([item['align5pts'] for item in batch])
    if all('align5pts_valid' in item for item in batch):
        collated['align5pts_valid'] = torch.stack([item['align5pts_valid'] for item in batch])
    if all('align_applied' in item for item in batch):
        collated['align_applied'] = torch.stack([item['align_applied'] for item in batch])
    if all('align_is_extreme' in item for item in batch):
        collated['align_is_extreme'] = torch.stack([item['align_is_extreme'] for item in batch])
    
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








