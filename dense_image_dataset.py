"""
Dense image dataset for RGB -> basecolor + geo + detail screen normal + geometry normal + face mask.
"""

import glob
import os
from typing import Optional

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from align_5pt_helper import Align5PtHelper

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
    except Exception:
        pass

    if img_array is None:
        try:
            import imageio.v3 as imageio

            img_array = imageio.imread(exr_path).astype(np.float32)
        except Exception:
            pass

    if img_array is None:
        try:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            img_array = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if img_array is not None:
                img_array = img_array.astype(np.float32)
        except Exception:
            pass

    return img_array


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


def _find_first_existing(paths: list[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def _parse_sample_id_from_color_file(color_path: str) -> str:
    filename = os.path.basename(color_path)
    if not filename.startswith("Color_"):
        return os.path.splitext(filename)[0]

    sample_name = os.path.splitext(filename[6:])[0]
    for suffix in ["_gemini", "_flux", "_seedream"]:
        if sample_name.endswith(suffix):
            return sample_name[: -len(suffix)]
    return sample_name


def _normalize_geo_like_image(img_array: np.ndarray) -> np.ndarray:
    if img_array.ndim == 2:
        img_array = np.repeat(img_array[..., None], 3, axis=2)
    if img_array.shape[2] >= 3:
        img_array = img_array[:, :, :3]
    else:
        img_array = np.repeat(img_array[:, :, :1], 3, axis=2)
    return np.clip(np.nan_to_num(img_array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)





def _warp_with_black_border(
    image: np.ndarray,
    m: np.ndarray,
    output_size: int,
    interpolation: int,
) -> np.ndarray:
    if image.ndim == 2:
        border_value = 0
    else:
        border_value = (0,) * image.shape[2]
    return cv2.warpAffine(
        image,
        m,
        (int(output_size), int(output_size)),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _transform_points_px(points_px: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.asarray(points_px, dtype=np.float32).copy()
    finite_mask = np.isfinite(out).all(axis=1)
    if not np.any(finite_mask):
        return out
    pts = out[finite_mask]
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    out[finite_mask] = pts_h @ m.T
    return out


def _build_pre_align_augment():
    """
    Build albumentations pipeline for pre-alignment geometric augmentation.
    Applies affine only (scale + translate + shear, no rotation, no grid distortion)
    so that the exact transform matrix can be recovered for normal map correction.
    All spatial transforms use black (zero) fill so no content is mirrored in.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    # Parameter names differ across albumentations versions; try each in order.
    affine = None
    for kw_fill, kw_fill_mask in [
        ("fill", "fill_mask"),           # albumentations 2.0.x
        ("fill_value", "mask_fill_value"),  # albumentations 2.x early
        ("cval", "cval_mask"),           # albumentations 1.x
    ]:
        try:
            affine = A.Affine(
                scale=(0.90, 1.10),
                translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
                rotate=0,
                shear=(-4, 4),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
                **{kw_fill: 0, kw_fill_mask: 0},
            )
            break
        except TypeError:
            continue
    if affine is None:
        return None

    return A.Compose(
        [affine],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
            check_each_transform=False,
        ),
        additional_targets={
            "basecolor": "image",
            "geo": "image",
            "screen_normal": "image",
            "geometry_normal": "image",
            "face_mask": "mask",
        },
        is_check_shapes=False,
    )


def _correct_normals_for_affine(normal_map: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Correct a screen-space normal map for a 2D affine Jacobian J (2x2).

    Screen-space normals are in camera/view space — their direction is absolute, not
    relative to the surface. Scale does not change normal directions, only the rotation
    component of J does. We therefore extract only the rotation part via SVD and apply
    that rotation to the XY channels.

    Encoding convention (same as _compute_vertex_normals_encoded):
        stored_R = (1 - Nx) / 2  →  Nx = 1 - 2*R
        stored_G = (1 - Ny) / 2  →  Ny = 1 - 2*G
        stored_B = (1 + Nz) / 2  →  Nz = 2*B - 1

    Background pixels (all-zero) are left unchanged.
    Returns corrected map in the same [0, 1] float encoding.
    """
    background = (normal_map[..., 0] == 0.0) & (normal_map[..., 1] == 0.0) & (normal_map[..., 2] == 0.0)
    try:
        # Extract only the rotation component of J via SVD (discard scale).
        # For orthogonal R: R^{-T} = R, so apply R directly to rotate XY.
        U, _, Vt = np.linalg.svd(J.astype(np.float64))
        J_rot = (U @ Vt).astype(np.float32)  # pure rotation, det ≈ ±1
    except np.linalg.LinAlgError:
        return normal_map.copy()
    J_inv_T = J_rot  # for orthogonal R: R^{-T} = R

    # Decode to unit normals in [-1, 1]
    nx = 1.0 - 2.0 * normal_map[..., 0]
    ny = 1.0 - 2.0 * normal_map[..., 1]
    nz = 2.0 * normal_map[..., 2] - 1.0

    # Apply J^{-T} to XY, keep Nz
    nx_new = J_inv_T[0, 0] * nx + J_inv_T[0, 1] * ny
    ny_new = J_inv_T[1, 0] * nx + J_inv_T[1, 1] * ny

    # Renormalize
    inv_norm = 1.0 / np.sqrt(nx_new ** 2 + ny_new ** 2 + nz ** 2 + 1e-8)
    nx_new *= inv_norm
    ny_new *= inv_norm
    nz_new = nz * inv_norm

    out = np.stack([
        np.clip(0.5 * (1.0 - nx_new), 0.0, 1.0),
        np.clip(0.5 * (1.0 - ny_new), 0.0, 1.0),
        np.clip(0.5 * (1.0 + nz_new), 0.0, 1.0),
    ], axis=-1).astype(np.float32)

    # Restore background pixels
    out[background] = 0.0
    return out


class DenseImageDataset(Dataset):
    ALIGN_SCALE_JITTER = 0.08
    ALIGN_TRANSLATE_JITTER = 0.03

    def __init__(
        self,
        data_roots,
        split: str = "train",
        image_size: int = 1024,
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

        self.align_helper = Align5PtHelper(
            image_size=self.image_size,
            scale_jitter=float(max(0.0, self.ALIGN_SCALE_JITTER)),
            translate_jitter=float(max(0.0, self.ALIGN_TRANSLATE_JITTER)),
        )
        self.pre_align_augment = None
        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.pre_align_augment = _build_pre_align_augment()

        self.samples: list[tuple] = []
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
                sample_id = _parse_sample_id_from_color_file(color_file)
                basecolor_path = os.path.join(data_root, "basecolor", f"BaseColor_{sample_id}.png")
                geo_path = os.path.join(data_root, "geo", f"Geo_{sample_id}.exr")
                screen_normal_path = os.path.join(data_root, "normal", f"ScreenNormal_{sample_id}.png")
                face_mask_path = os.path.join(data_root, "facemask", f"Face_Mask_{sample_id}.png")
                landmark_path = _find_first_existing(
                    [
                        os.path.join(data_root, "landmark", f"landmark_{sample_id}.txt"),
                        os.path.join(data_root, "landmark", f"{sample_id}.txt"),
                    ]
                )
                error_mask_path = os.path.join(data_root, f"Color_{sample_id}_mask.png")
                geo_normal_path = os.path.join(data_root, "geo_normal", f"GeoNormal_{sample_id}.png")
                if not os.path.exists(basecolor_path) or not os.path.exists(geo_path) or not os.path.exists(screen_normal_path) or not os.path.exists(face_mask_path) or not os.path.exists(geo_normal_path):
                    continue
                self.samples.append(
                    (
                        data_root,
                        sample_id,
                        color_file,
                        basecolor_path,
                        geo_path,
                        screen_normal_path,
                        face_mask_path,
                        error_mask_path if os.path.exists(error_mask_path) else None,
                        landmark_path,
                        geo_normal_path,
                    )
                )

        self.samples = sorted(list(set(self.samples)))

        samples_by_root: dict[str, list[tuple]] = {}
        for sample in self.samples:
            samples_by_root.setdefault(sample[0], []).append(sample)

        split_samples: list[tuple] = []
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
        if self.augment and ALBUMENTATIONS_AVAILABLE:
            self.image_only_augment = A.Compose(
                [
                    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05, p=0.8),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                    A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.MotionBlur(blur_limit=5)], p=0.2),
                ]
            )

    def _apply_pre_align_aug(
        self,
        image_np: np.ndarray,
        basecolor_np: np.ndarray,
        geo_np: np.ndarray,
        screen_normal_np: np.ndarray,
        geometry_normal_np: np.ndarray,
        face_mask_np: np.ndarray,
    ) -> tuple:
        """Apply albumentations geometric augmentation before dlib alignment.
        Returns (image, basecolor, geo, screen_normal, geometry_normal, face_mask, m_aug).
        m_aug is the 2x3 affine matrix recovered from 4 corner correspondences.
        """
        geo_u8 = np.clip(geo_np * 255.0, 0, 255).astype(np.uint8)
        sn_u8 = np.clip(screen_normal_np * 255.0, 0, 255).astype(np.uint8)
        gn_u8 = np.clip(geometry_normal_np * 255.0, 0, 255).astype(np.uint8)

        h, w = image_np.shape[:2]
        corner_src = [
            (0.0, 0.0), (float(w - 1), 0.0),
            (0.0, float(h - 1)), (float(w - 1), float(h - 1)),
        ]

        result = self.pre_align_augment(
            image=image_np,
            basecolor=basecolor_np,
            geo=geo_u8,
            screen_normal=sn_u8,
            geometry_normal=gn_u8,
            face_mask=face_mask_np,
            keypoints=corner_src,
        )

        aug_image = result["image"]
        aug_basecolor = result["basecolor"]
        aug_geo = result["geo"].astype(np.float32) / 255.0
        aug_screen_normal = result["screen_normal"].astype(np.float32) / 255.0
        aug_geometry_normal = result["geometry_normal"].astype(np.float32) / 255.0
        aug_face_mask = result["face_mask"]

        m_aug = None
        aug_kps = result["keypoints"]
        if len(aug_kps) >= 4:
            corner_dst = np.array([[p[0], p[1]] for p in aug_kps[:4]], dtype=np.float32)
            m_fit, _ = cv2.estimateAffine2D(np.array(corner_src, dtype=np.float32), corner_dst)
            if m_fit is not None and m_fit.shape == (2, 3) and np.isfinite(m_fit).all():
                m_aug = m_fit

        return aug_image, aug_basecolor, aug_geo, aug_screen_normal, aug_geometry_normal, aug_face_mask, m_aug

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        (
            data_root,
            sample_id,
            color_path,
            basecolor_path,
            geo_path,
            screen_normal_path,
            face_mask_path,
            error_mask_path,
            landmark_path,
            geo_normal_path,
        ) = self.samples[idx]

        image_np = cv2.cvtColor(cv2.imread(color_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        src_h, src_w = image_np.shape[:2]
        if src_h != 1024 or src_w != 1024:
            image_np = cv2.resize(image_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            src_h, src_w = 1024, 1024

        basecolor_np = cv2.cvtColor(cv2.imread(basecolor_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if basecolor_np.shape[:2] != (1024, 1024):
            basecolor_np = cv2.resize(basecolor_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        geo_np = _normalize_geo_like_image(load_exr_image(geo_path))
        if geo_np.shape[:2] != (1024, 1024):
            geo_np = cv2.resize(geo_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        screen_normal_np = cv2.cvtColor(cv2.imread(screen_normal_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if screen_normal_np.shape[:2] != (1024, 1024):
            screen_normal_np = cv2.resize(screen_normal_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        if face_mask_path is not None:
            face_mask_img = cv2.imread(face_mask_path, cv2.IMREAD_UNCHANGED)
            if face_mask_img.ndim == 3 and face_mask_img.shape[2] == 4:
                face_mask_np = face_mask_img[:, :, 3]
            elif face_mask_img.ndim == 3:
                face_mask_np = cv2.cvtColor(face_mask_img, cv2.COLOR_BGR2GRAY)
            else:
                face_mask_np = face_mask_img
            if face_mask_np.shape[:2] != (1024, 1024):
                face_mask_np = cv2.resize(face_mask_np, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        else:
            face_mask_np = np.zeros((src_h, src_w), dtype=np.uint8)

        if geo_normal_path is not None:
            geometry_normal_np = cv2.cvtColor(
                cv2.imread(geo_normal_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0
            if geometry_normal_np.shape[:2] != (src_h, src_w):
                geometry_normal_np = cv2.resize(geometry_normal_np, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
        else:
            geometry_normal_np = np.zeros((src_h, src_w, 3), dtype=np.float32)

        m_aug = None
        if self.pre_align_augment is not None:
            (
                image_np, basecolor_np, geo_np, screen_normal_np,
                geometry_normal_np, face_mask_np, m_aug
            ) = self._apply_pre_align_aug(
                image_np, basecolor_np, geo_np, screen_normal_np,
                geometry_normal_np, face_mask_np,
            )

        # Load GT landmark pixels for MediaPipe fallback; transform through m_aug if applied.
        lm_px_gt = _load_landmark_pixels(landmark_path)
        if lm_px_gt is not None and m_aug is not None:
            ones = np.ones((len(lm_px_gt), 1), dtype=np.float32)
            lm_px_gt = (np.concatenate([lm_px_gt, ones], axis=1) @ m_aug.T).astype(np.float32)

        # Run MediaPipe on the (possibly augmented) image to get landmarks for alignment.
        lm68, detection_source = self.align_helper.detect_landmarks(image_np, fallback_lm_px=lm_px_gt)
        align_h, align_w = image_np.shape[:2]
        align_matrix = self.align_helper.estimate_alignment_matrix(lm68, src_w=align_w, src_h=align_h, split=self.split)
        total_m = align_matrix

        image_np = _warp_with_black_border(image_np, total_m, self.image_size, cv2.INTER_LINEAR)
        basecolor_np = _warp_with_black_border(basecolor_np, total_m, self.image_size, cv2.INTER_LINEAR)
        geo_np = _warp_with_black_border(geo_np, total_m, self.image_size, cv2.INTER_LINEAR)
        screen_normal_np = _warp_with_black_border(screen_normal_np, total_m, self.image_size, cv2.INTER_LINEAR)
        geometry_normal_np = _warp_with_black_border(geometry_normal_np, total_m, self.image_size, cv2.INTER_LINEAR)
        face_mask_np = _warp_with_black_border(face_mask_np, total_m, self.image_size, cv2.INTER_NEAREST)

        warp_valid_np = _warp_with_black_border(
            np.full((src_h, src_w), 255, dtype=np.uint8), total_m, self.image_size, cv2.INTER_NEAREST
        )

        if error_mask_path is not None:
            error_mask_img = cv2.imread(error_mask_path, cv2.IMREAD_UNCHANGED)
            if error_mask_img.ndim == 3 and error_mask_img.shape[2] == 4:
                alpha = error_mask_img[:, :, 3]
            elif error_mask_img.ndim == 2:
                alpha = error_mask_img
            else:
                alpha = error_mask_img[:, :, 0]
            alpha = cv2.resize(alpha, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
            alpha = np.where(alpha > 0, np.uint8(0), np.uint8(255))  # alpha>0=bad→0, transparent=good→255
            error_mask_np = _warp_with_black_border(alpha, total_m, self.image_size, cv2.INTER_NEAREST)
        else:
            error_mask_np = np.full((self.image_size, self.image_size), 255, dtype=np.uint8)

        # Correct normal map XY channels for the combined affine transform.
        # J = total_m_linear @ m_aug_linear; normals transform by J^{-T}.
        J_aug = m_aug[:2, :2] if m_aug is not None else np.eye(2, dtype=np.float32)
        J_full = (total_m[:2, :2] @ J_aug).astype(np.float32)
        geometry_normal_np = _correct_normals_for_affine(geometry_normal_np, J_full)
        screen_normal_np = _correct_normals_for_affine(screen_normal_np, J_full)

        detail_screen_normal_np = np.clip(screen_normal_np - geometry_normal_np, -1.0, 1.0)

        if self.image_only_augment is not None:
            image_np = self.image_only_augment(image=image_np)["image"]

        key5_px, key5_valid = self.align_helper.extract_key5_from_lm68(lm68)
        key5_out = _transform_points_px(key5_px, total_m) / float(max(1, self.image_size))
        key5_out = np.nan_to_num(key5_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        if lm68 is not None:
            jaw_px = _transform_points_px(lm68[5:6], total_m) / float(max(1, self.image_size))
            jaw_out = np.nan_to_num(jaw_px[0].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            jaw_valid = 1.0
        else:
            jaw_out = np.zeros(2, dtype=np.float32)
            jaw_valid = 0.0

        rgb_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        basecolor_tensor = torch.from_numpy(basecolor_np).permute(2, 0, 1).float() / 255.0
        geo_tensor = torch.from_numpy(geo_np).permute(2, 0, 1).float().clamp(0.0, 1.0)
        detail_screen_normal_tensor = torch.from_numpy(detail_screen_normal_np).permute(2, 0, 1).float()
        screen_normal_tensor = torch.from_numpy(screen_normal_np).permute(2, 0, 1).float().clamp(0.0, 1.0)
        geometry_normal_tensor = (
            torch.from_numpy(geometry_normal_np).permute(2, 0, 1).float().clamp(0.0, 1.0)
        )
        face_mask_tensor = torch.from_numpy(face_mask_np).unsqueeze(0).float() / 255.0
        error_mask_tensor = torch.from_numpy(((warp_valid_np > 0) & (error_mask_np > 0)).astype(np.float32)).unsqueeze(0)

        return {
            "rgb": rgb_tensor,
            "basecolor": basecolor_tensor,
            "geo": geo_tensor,
            "screen_normal": screen_normal_tensor,
            "detail_normal": detail_screen_normal_tensor.clamp(-1.0, 1.0),
            "geometry_normal": geometry_normal_tensor,
            "normal": detail_screen_normal_tensor.clamp(-1.0, 1.0),
            "face_mask": face_mask_tensor.clamp(0.0, 1.0),
            "error_mask": error_mask_tensor,
            "align5pts": torch.from_numpy(key5_out),
            "align5pts_valid": torch.from_numpy(key5_valid),
            "jaw_pt": torch.from_numpy(jaw_out),
            "jaw_valid": torch.tensor(jaw_valid, dtype=torch.float32),
            "align_applied": torch.tensor(float(align_matrix is not None), dtype=torch.float32),
            "detection_source": detection_source,
            "sample_id": sample_id,
            "data_root": data_root,
            "image_path": color_path,
        }


def dense_image_collate_fn(batch):
    return {
        "rgb": torch.stack([item["rgb"] for item in batch]),
        "basecolor": torch.stack([item["basecolor"] for item in batch]),
        "geo": torch.stack([item["geo"] for item in batch]),
        "screen_normal": torch.stack([item["screen_normal"] for item in batch]),
        "detail_normal": torch.stack([item["detail_normal"] for item in batch]),
        "geometry_normal": torch.stack([item["geometry_normal"] for item in batch]),
        "normal": torch.stack([item["normal"] for item in batch]),
        "face_mask": torch.stack([item["face_mask"] for item in batch]),
        "error_mask": torch.stack([item["error_mask"] for item in batch]),
        "align5pts": torch.stack([item["align5pts"] for item in batch]),
        "align5pts_valid": torch.stack([item["align5pts_valid"] for item in batch]),
        "jaw_pt": torch.stack([item["jaw_pt"] for item in batch]),
        "jaw_valid": torch.stack([item["jaw_valid"] for item in batch]),
        "align_applied": torch.stack([item["align_applied"] for item in batch]),
        "detection_source": [item["detection_source"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "data_root": [item["data_root"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }
