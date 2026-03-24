import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from align_5pt_helper import Align5PtHelper
from alignment_io import (
    apply_alignment_to_geometry,
    build_alignment_metadata,
    compute_alignment_transform,
    warp_rgb_image,
)
from image_io import (
    apply_occlusion_mask_to_weights,
    load_optional_mask,
    load_rgb_image_or_default,
    resize_rgb_or_default,
)
from mesh_io import (
    apply_geometry_indices,
    attach_depth_channels,
    compute_geometry_found_mask,
    ensure_geometry_with_fallback,
    load_geometry_template_set,
    load_geometry_txt,
)
from sample_index import collect_geometry_sample_records

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def _should_log_dataset_init() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        pass
    return True


class Dense2GeometryDataset(Dataset):
    """Minimal dense2geometry dataset using shared sample/image/mesh helpers."""

    ALIGN_SCALE_JITTER = 0.08
    ALIGN_TRANSLATE_JITTER = 0.03

    def __init__(
        self,
        data_roots,
        split: str = "train",
        image_size: int = 512,
        train_ratio: float = 0.95,
        augment: bool = True,
        texture_root: str | None = None,
    ):
        del texture_root
        self.split = str(split)
        self.image_size = int(image_size)
        self.augment = bool(augment and split == "train")
        self.samples = collect_geometry_sample_records(
            data_roots=data_roots,
            split=split,
            train_ratio=train_ratio,
        )

        self.align_helper = Align5PtHelper(
            image_size=self.image_size,
            scale_jitter=float(max(0.0, self.ALIGN_SCALE_JITTER)),
            translate_jitter=float(max(0.0, self.ALIGN_TRANSLATE_JITTER)),
        )

        templates = load_geometry_template_set(model_dir="model")
        self.mesh_indices = templates.mesh_indices
        self.default_mesh = templates.default_mesh
        self.template_landmark_depth = templates.template_landmark_depth
        self.template_mesh_depth = templates.template_mesh_depth

        self.image_only_augment = None
        self.image_only_augment_tv = None
        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.image_only_augment = A.Compose(
                [
                    A.ISONoise(color_shift=(0.0, 0.02), intensity=(0.1, 0.5), p=0.3),
                    A.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.05,
                        p=0.8,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5,
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 7)),
                            A.MotionBlur(blur_limit=5),
                            A.MedianBlur(blur_limit=5),
                        ],
                        p=0.3,
                    ),
                    A.UnsharpMask(
                        blur_limit=(3, 7),
                        sigma_limit=0.0,
                        alpha=(0.2, 0.5),
                        threshold=10,
                        p=0.3,
                    ),
                ]
            )
        elif self.augment:
            self.image_only_augment_tv = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.15,
                        saturation=0.15,
                        hue=0.05,
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                        p=0.3,
                    ),
                ]
            )

        if _should_log_dataset_init():
            print(f"[{self.__class__.__name__}] split={split} total={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        record = self.samples[idx]
        image_np = load_rgb_image_or_default(record.color_path, image_size=self.image_size)
        src_h, src_w = image_np.shape[:2]

        mesh, mesh_raw_xyz = load_geometry_txt(record.mesh_path, return_raw_xyz=True)
        geometry_fallback = mesh is None
        if geometry_fallback:
            mesh = self.default_mesh.copy()
            mesh_raw_xyz = None

        mesh_found_mask = compute_geometry_found_mask(mesh)
        if geometry_fallback:
            mesh_found_mask = np.zeros((mesh.shape[0],), dtype=bool)

        mesh_weights = np.ones((mesh.shape[0],), dtype=np.float32)
        mesh_weights[~mesh_found_mask] = 0.0

        mask_alpha = load_optional_mask(record.mask_path)
        mesh_weights = apply_occlusion_mask_to_weights(
            geometry=mesh,
            weights=mesh_weights,
            mask_alpha=mask_alpha,
            width=src_w,
            height=src_h,
        )

        lm6, detection_source, transform = compute_alignment_transform(
            image_np=image_np,
            landmark_path=record.landmark_path,
            align_helper=self.align_helper,
            split=self.split,
        )
        image_np = warp_rgb_image(image_np, transform, self.image_size)
        mesh = apply_alignment_to_geometry(self.align_helper, mesh, transform, src_w=src_w, src_h=src_h)

        mesh, mesh_raw_xyz, mesh_found_mask, mesh_weights = apply_geometry_indices(
            geometry=mesh,
            raw_xyz=mesh_raw_xyz,
            found_mask=mesh_found_mask,
            weights=mesh_weights,
            indices=self.mesh_indices,
        )
        mesh, mesh_raw_xyz, mesh_found_mask, mesh_weights = ensure_geometry_with_fallback(
            geometry=mesh,
            raw_xyz=mesh_raw_xyz,
            found_mask=mesh_found_mask,
            weights=mesh_weights,
            fallback_geometry=self.default_mesh,
        )

        if self.image_only_augment is not None and ALBUMENTATIONS_AVAILABLE:
            image_np = self.image_only_augment(image=image_np)["image"]
        elif self.image_only_augment_tv is not None:
            img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            img_tensor = self.image_only_augment_tv(img_tensor)
            image_np = (img_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)

        image_np = resize_rgb_or_default(image_np, self.image_size)
        _, mesh, _ = attach_depth_channels(
            landmarks=None,
            landmarks_raw_xyz=None,
            mesh=mesh,
            mesh_raw_xyz=mesh_raw_xyz,
            mat_path=record.mat_path,
            template_landmark_depth=self.template_landmark_depth,
            template_mesh_depth=self.template_mesh_depth,
            sample_id=record.sample_id,
            color_path=record.color_path,
        )

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mesh_tensor = torch.from_numpy(mesh).float()
        mesh_found_tensor = torch.from_numpy(mesh_found_mask.astype(np.float32, copy=False))
        mesh_weights_tensor = torch.from_numpy(mesh_weights.astype(np.float32, copy=False))
        mesh_loss_weights = mesh_found_tensor * mesh_weights_tensor

        key5_out, key5_valid = build_alignment_metadata(
            align_helper=self.align_helper,
            lm6=lm6,
            transform=transform,
            image_size=self.image_size,
        )

        return {
            "rgb": rgb_tensor,
            "mesh": mesh_tensor,
            "mesh_loss_weights": mesh_loss_weights,
            "image_path": record.color_path,
            "align5pts": torch.from_numpy(key5_out),
            "align5pts_valid": torch.from_numpy(key5_valid),
            "align_applied": torch.tensor(1.0, dtype=torch.float32),
            "detection_source": detection_source,
        }


def dense2geometry_collate_fn(batch):
    collated = {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "mesh": torch.stack([item["mesh"] for item in batch]),
        "mesh_loss_weights": torch.stack([item["mesh_loss_weights"] for item in batch]),
    }
    if "image_path" in batch[0]:
        collated["image_path"] = [item["image_path"] for item in batch]
    if all("align5pts" in item for item in batch):
        collated["align5pts"] = torch.stack([item["align5pts"] for item in batch])
    if all("align5pts_valid" in item for item in batch):
        collated["align5pts_valid"] = torch.stack([item["align5pts_valid"] for item in batch])
    if all("align_applied" in item for item in batch):
        collated["align_applied"] = torch.stack([item["align_applied"] for item in batch])
    if all("detection_source" in item for item in batch):
        collated["detection_source"] = [item["detection_source"] for item in batch]
    return collated
