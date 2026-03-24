from __future__ import annotations

import cv2
import numpy as np

from .mesh_io import load_landmark_pixels
from .image_io import warp_affine_black_border


def compute_alignment_transform(
    image_np: np.ndarray,
    landmark_path: str,
    align_helper,
    split: str,
) -> tuple[np.ndarray, str, np.ndarray]:
    height, width = image_np.shape[:2]
    landmark_pixels = load_landmark_pixels(landmark_path)
    lm6, detection_source = align_helper.detect_landmarks(image_np, fallback_lm_px=landmark_pixels)
    transform = align_helper.estimate_alignment_matrix(lm6, src_w=width, src_h=height, split=split)
    return lm6, detection_source, transform


def warp_rgb_image(image: np.ndarray, transform: np.ndarray, output_size: int) -> np.ndarray:
    return warp_affine_black_border(
        image=image,
        transform=transform,
        output_size=output_size,
        interpolation=cv2.INTER_LINEAR,
    )


def warp_mask_image(image: np.ndarray, transform: np.ndarray, output_size: int) -> np.ndarray:
    return warp_affine_black_border(
        image=image,
        transform=transform,
        output_size=output_size,
        interpolation=cv2.INTER_NEAREST,
    )


def apply_alignment_to_geometry(align_helper, geometry: np.ndarray | None, transform: np.ndarray, src_w: int, src_h: int):
    if geometry is None:
        return None
    return align_helper.apply_alignment_to_geometry(geometry, m=transform, src_w=src_w, src_h=src_h)


def build_alignment_metadata(align_helper, lm6: np.ndarray, transform: np.ndarray, image_size: int):
    key5_px, key5_valid = align_helper.extract_key5_from_lm68(lm6)
    key5_out = align_helper.transform_points_px(key5_px, transform) / float(max(1, image_size))
    key5_out = np.nan_to_num(key5_out.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return key5_out, key5_valid
