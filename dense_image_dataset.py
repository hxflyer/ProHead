"""
Dense image dataset for basecolor supervision.
Loads RGB input and BaseColor target image pairs.
"""

import glob
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


class DenseImageDataset(Dataset):
    """
    Dataset for dense image prediction (RGB -> BaseColor).
    """

    def __init__(
        self,
        data_roots,
        split: str = "train",
        image_size: int = 512,
        train_ratio: float = 0.95,
        augment: bool = True,
    ):
        if isinstance(data_roots, str):
            self.data_roots = [data_roots]
        else:
            self.data_roots = list(data_roots)

        self.split = str(split)
        self.image_size = int(image_size)
        self.augment = bool(augment and self.split == "train")

        self.samples: list[tuple[str, str, Optional[str]]] = []
        for data_root in self.data_roots:
            if not os.path.exists(data_root):
                continue

            color_files = glob.glob(os.path.join(data_root, "Color_*"))
            color_files = [
                f
                for f in color_files
                if os.path.basename(f).startswith("Color_")
                and not os.path.basename(f).endswith("_mask.png")
            ]

            for color_file in color_files:
                filename = os.path.basename(color_file)
                if not filename.startswith("Color_"):
                    continue
                sample_name = os.path.splitext(filename[6:])[0]
                base_id = sample_name
                for suffix in ["_gemini", "_flux", "_seedream"]:
                    if base_id.endswith(suffix):
                        base_id = base_id[: -len(suffix)]
                        break
                basecolor_path = os.path.join(
                    data_root, "basecolor", f"BaseColor_{base_id}.png"
                )
                self.samples.append(
                    (
                        data_root,
                        color_file,
                        basecolor_path if os.path.exists(basecolor_path) else None,
                    )
                )

        self.samples = sorted(list(set(self.samples)))

        samples_by_root: dict[str, list[tuple[str, str, Optional[str]]]] = {}
        for sample in self.samples:
            samples_by_root.setdefault(sample[0], []).append(sample)

        split_samples: list[tuple[str, str, Optional[str]]] = []
        val_ratio = float(np.clip(1.0 - float(train_ratio), 0.0, 1.0))
        for root in sorted(samples_by_root.keys()):
            root_samples = sorted(samples_by_root[root])
            val_count = int(len(root_samples) * val_ratio)
            if self.split == "train":
                split_samples.extend(root_samples[val_count:])
            else:
                split_samples.extend(root_samples[:val_count])
        self.samples = split_samples

        self.image_only_augment = None
        self.image_only_augment_tv = None
        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.image_only_augment = A.Compose(
                [
                    A.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.25,
                        hue=0.05,
                        p=0.8,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5)),
                            A.MotionBlur(blur_limit=5),
                        ],
                        p=0.2,
                    ),
                ]
            )
        elif self.augment:
            self.image_only_augment_tv = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                        p=0.2,
                    ),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, color_path, basecolor_path = self.samples[idx]

        image_np = cv2.imread(color_path, cv2.IMREAD_COLOR)
        if image_np is None:
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        if basecolor_path is not None and os.path.exists(basecolor_path):
            basecolor_np = cv2.imread(basecolor_path, cv2.IMREAD_COLOR)
            if basecolor_np is None:
                basecolor_np = np.zeros_like(image_np, dtype=np.uint8)
                basecolor_valid = 0.0
            else:
                basecolor_np = cv2.cvtColor(basecolor_np, cv2.COLOR_BGR2RGB)
                basecolor_valid = 1.0
        else:
            basecolor_np = np.zeros_like(image_np, dtype=np.uint8)
            basecolor_valid = 0.0

        if self.image_only_augment is not None:
            image_np = self.image_only_augment(image=image_np)["image"]
        elif self.image_only_augment_tv is not None:
            rgb_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            rgb_t = self.image_only_augment_tv(rgb_t)
            image_np = (rgb_t.permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(
                np.uint8
            )

        if (
            image_np.shape[0] != self.image_size
            or image_np.shape[1] != self.image_size
        ):
            image_np = cv2.resize(
                image_np, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )
        if (
            basecolor_np.shape[0] != self.image_size
            or basecolor_np.shape[1] != self.image_size
        ):
            basecolor_np = cv2.resize(
                basecolor_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        basecolor_tensor = (
            torch.from_numpy(basecolor_np).permute(2, 0, 1).float() / 255.0
        )

        return {
            "rgb": rgb_tensor,
            "basecolor": basecolor_tensor,
            "basecolor_valid": torch.tensor(
                1.0 if float(basecolor_valid) > 0.5 else 0.0, dtype=torch.float32
            ),
            "image_path": color_path,
        }


def dense_image_collate_fn(batch):
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "basecolor": torch.stack([item["basecolor"] for item in batch]),
        "basecolor_valid": torch.stack([item["basecolor_valid"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
    }
