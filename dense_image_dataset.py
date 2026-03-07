"""
Dense image dataset for RGB basecolor + geo + normal + face-mask supervision.
Loads RGB input, three dense 3-channel targets, and an optional face mask.
"""

import glob
import os
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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


def load_exr_image(exr_path):
    """
    Load EXR image with fallback methods.
    Returns numpy array in float32 format.
    """
    img_array = None

    # Method 1: Try OpenEXR
    try:
        import Imath
        import OpenEXR

        exr_file = OpenEXR.InputFile(exr_path)
        header = exr_file.header()

        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        channels_data = {}
        for channel in ["R", "G", "B", "A"]:
            if channel in header["channels"]:
                channel_type = header["channels"][channel].type
                if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                    pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                    channel_array = np.frombuffer(pixel_data, dtype=np.float32).reshape((height, width))
                else:
                    pixel_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.HALF))
                    channel_array = np.frombuffer(pixel_data, dtype=np.float16).reshape((height, width))
                    channel_array = channel_array.astype(np.float32)
                channels_data[channel] = channel_array

        if channels_data:
            img_array = np.stack(
                [
                    channels_data.get(ch, np.zeros((height, width), dtype=np.float32))
                    for ch in ["R", "G", "B", "A"]
                ],
                axis=2,
            )
    except ImportError:
        pass
    except Exception as e:
        print(f"OpenEXR loading failed: {e}")

    # Method 2: Try imageio
    if img_array is None:
        try:
            import imageio.v3 as imageio

            img_array = imageio.imread(exr_path).astype(np.float32)
        except Exception:
            pass

    # Method 3: Try cv2
    if img_array is None:
        try:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            img_array = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img_array is not None:
                img_array = img_array.astype(np.float32)
        except Exception:
            pass

    return img_array


class DenseImageDataset(Dataset):
    """
    Dataset for dense image prediction (RGB -> BaseColor + Geo + Normal + FaceMask).
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
        self.root_stats: dict[str, dict[str, int]] = {}
        self.missing_basecolor_examples: list[str] = []
        self.missing_geo_examples: list[str] = []
        self.missing_normal_examples: list[str] = []
        self.missing_face_mask_examples: list[str] = []

        self.samples: list[
            tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]
        ] = []
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
                geo_path = os.path.join(
                    data_root, "geo", f"Geo_{base_id}.exr"
                )
                normal_path = os.path.join(
                    data_root, "normal", f"ScreenNormal_{base_id}.png"
                )
                face_mask_path = os.path.join(
                    data_root, "facemask", f"Face_Mask_{base_id}.png"
                )
                basecolor_exists = os.path.exists(basecolor_path)
                geo_exists = os.path.exists(geo_path)
                normal_exists = os.path.exists(normal_path)
                face_mask_exists = os.path.exists(face_mask_path)
                root_info = self.root_stats.setdefault(
                    data_root,
                    {
                        "total": 0,
                        "with_basecolor": 0,
                        "with_geo": 0,
                        "with_normal": 0,
                        "with_face_mask": 0,
                    },
                )
                root_info["total"] += 1
                if basecolor_exists:
                    root_info["with_basecolor"] += 1
                elif len(self.missing_basecolor_examples) < 12:
                    self.missing_basecolor_examples.append(
                        f"{os.path.basename(color_file)} -> {basecolor_path}"
                    )
                if geo_exists:
                    root_info["with_geo"] += 1
                elif len(self.missing_geo_examples) < 12:
                    self.missing_geo_examples.append(
                        f"{os.path.basename(color_file)} -> {geo_path}"
                    )
                if normal_exists:
                    root_info["with_normal"] += 1
                elif len(self.missing_normal_examples) < 12:
                    self.missing_normal_examples.append(
                        f"{os.path.basename(color_file)} -> {normal_path}"
                    )
                if face_mask_exists:
                    root_info["with_face_mask"] += 1
                elif len(self.missing_face_mask_examples) < 12:
                    self.missing_face_mask_examples.append(
                        f"{os.path.basename(color_file)} -> {face_mask_path}"
                    )
                self.samples.append(
                    (
                        data_root,
                        color_file,
                        basecolor_path if basecolor_exists else None,
                        geo_path if geo_exists else None,
                        normal_path if normal_exists else None,
                        face_mask_path if face_mask_exists else None,
                    )
                )

        self.samples = sorted(list(set(self.samples)))

        samples_by_root: dict[
            str,
            list[tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]],
        ] = {}
        for sample in self.samples:
            samples_by_root.setdefault(sample[0], []).append(sample)

        split_samples: list[
            tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]
        ] = []
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

    def get_debug_summary(self) -> dict:
        total = len(self.samples)
        basecolor_valid = sum(1 for _, _, bc_path, _, _, _ in self.samples if bc_path is not None)
        geo_valid = sum(1 for _, _, _, geo_path, _, _ in self.samples if geo_path is not None)
        normal_valid = sum(1 for _, _, _, _, normal_path, _ in self.samples if normal_path is not None)
        face_mask_valid = sum(1 for _, _, _, _, _, mask_path in self.samples if mask_path is not None)
        summary = {
            "split": self.split,
            "total": int(total),
            "valid": int(basecolor_valid),
            "valid_ratio": float(basecolor_valid / max(total, 1)),
            "basecolor_valid": int(basecolor_valid),
            "basecolor_valid_ratio": float(basecolor_valid / max(total, 1)),
            "geo_valid": int(geo_valid),
            "geo_valid_ratio": float(geo_valid / max(total, 1)),
            "normal_valid": int(normal_valid),
            "normal_valid_ratio": float(normal_valid / max(total, 1)),
            "face_mask_valid": int(face_mask_valid),
            "face_mask_valid_ratio": float(face_mask_valid / max(total, 1)),
            "roots": dict(self.root_stats),
            "missing_examples": list(self.missing_basecolor_examples),
            "missing_geo_examples": list(self.missing_geo_examples),
            "missing_normal_examples": list(self.missing_normal_examples),
            "missing_face_mask_examples": list(self.missing_face_mask_examples),
        }
        return summary

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, color_path, basecolor_path, geo_path, normal_path, face_mask_path = self.samples[idx]

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

        if geo_path is not None and os.path.exists(geo_path):
            geo_np = load_exr_image(geo_path)
            if geo_np is None:
                geo_np = np.zeros((*image_np.shape[:2], 3), dtype=np.float32)
                geo_valid = 0.0
            else:
                if geo_np.ndim == 2:
                    geo_np = np.repeat(geo_np[..., None], 3, axis=2)
                if geo_np.shape[2] >= 3:
                    geo_np = geo_np[:, :, :3]
                else:
                    geo_np = np.repeat(geo_np[:, :, :1], 3, axis=2)
                geo_np = np.nan_to_num(geo_np.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
                geo_np = np.clip(geo_np, 0.0, 1.0)
                geo_valid = 1.0
        else:
            geo_np = np.zeros((*image_np.shape[:2], 3), dtype=np.float32)
            geo_valid = 0.0

        if normal_path is not None and os.path.exists(normal_path):
            normal_img = cv2.imread(normal_path, cv2.IMREAD_COLOR)
            if normal_img is None:
                normal_np = np.zeros_like(image_np, dtype=np.uint8)
                normal_valid = 0.0
            else:
                normal_np = cv2.cvtColor(normal_img, cv2.COLOR_BGR2RGB)
                normal_valid = 1.0
        else:
            normal_np = np.zeros_like(image_np, dtype=np.uint8)
            normal_valid = 0.0

        if face_mask_path is not None and os.path.exists(face_mask_path):
            face_mask_img = cv2.imread(face_mask_path, cv2.IMREAD_UNCHANGED)
            if face_mask_img is None:
                face_mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
                face_mask_valid = 0.0
            else:
                if face_mask_img.ndim == 3 and face_mask_img.shape[2] == 4:
                    face_mask_np = face_mask_img[:, :, 3]
                elif face_mask_img.ndim == 3:
                    face_mask_np = cv2.cvtColor(face_mask_img, cv2.COLOR_BGR2GRAY)
                else:
                    face_mask_np = face_mask_img
                face_mask_valid = 1.0
        else:
            face_mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
            face_mask_valid = 0.0

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
        if (
            geo_np.shape[0] != self.image_size
            or geo_np.shape[1] != self.image_size
        ):
            geo_np = cv2.resize(
                geo_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        if (
            normal_np.shape[0] != self.image_size
            or normal_np.shape[1] != self.image_size
        ):
            normal_np = cv2.resize(
                normal_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
        if (
            face_mask_np.shape[0] != self.image_size
            or face_mask_np.shape[1] != self.image_size
        ):
            face_mask_np = cv2.resize(
                face_mask_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        basecolor_tensor = (
            torch.from_numpy(basecolor_np).permute(2, 0, 1).float() / 255.0
        )
        geo_tensor = torch.from_numpy(geo_np).permute(2, 0, 1).float().clamp(0.0, 1.0)
        normal_tensor = (
            torch.from_numpy(normal_np).permute(2, 0, 1).float() / 255.0
        )
        face_mask_tensor = torch.from_numpy(face_mask_np).unsqueeze(0).float() / 255.0

        return {
            "rgb": rgb_tensor,
            "basecolor": basecolor_tensor,
            "basecolor_valid": torch.tensor(
                1.0 if float(basecolor_valid) > 0.5 else 0.0, dtype=torch.float32
            ),
            "geo": geo_tensor,
            "geo_valid": torch.tensor(
                1.0 if float(geo_valid) > 0.5 else 0.0, dtype=torch.float32
            ),
            "normal": normal_tensor,
            "normal_valid": torch.tensor(
                1.0 if float(normal_valid) > 0.5 else 0.0, dtype=torch.float32
            ),
            "face_mask": face_mask_tensor.clamp(0.0, 1.0),
            "face_mask_valid": torch.tensor(
                1.0 if float(face_mask_valid) > 0.5 else 0.0, dtype=torch.float32
            ),
            "image_path": color_path,
        }


def dense_image_collate_fn(batch):
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "basecolor": torch.stack([item["basecolor"] for item in batch]),
        "basecolor_valid": torch.stack([item["basecolor_valid"] for item in batch]),
        "geo": torch.stack([item["geo"] for item in batch]),
        "geo_valid": torch.stack([item["geo_valid"] for item in batch]),
        "normal": torch.stack([item["normal"] for item in batch]),
        "normal_valid": torch.stack([item["normal_valid"] for item in batch]),
        "face_mask": torch.stack([item["face_mask"] for item in batch]),
        "face_mask_valid": torch.stack([item["face_mask_valid"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
    }
