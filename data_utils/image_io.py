from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False


_WARNED_BAD_IMAGE_PATHS: set[tuple[str, str]] = set()


def _warn_bad_image_once(path: str | None, context: str, exc: Exception | None = None) -> None:
    key = (str(context), str(path or ""))
    if key in _WARNED_BAD_IMAGE_PATHS:
        return
    _WARNED_BAD_IMAGE_PATHS.add(key)
    suffix = f" ({exc})" if exc is not None else ""
    print(f"[ImageIO] Failed to decode {context}: {path}{suffix}")


def load_unchanged_image(path: str | None, context: str = "image") -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None

    if PIL_AVAILABLE:
        try:
            with Image.open(path) as image:
                image.load()
                if image.mode == "P":
                    image = image.convert("RGBA" if "transparency" in image.info else "RGB")
                out = np.array(image)
                if out.ndim == 0:
                    return None
                return out
        except Exception as exc:
            _warn_bad_image_once(path, context, exc)
            return None

    try:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            _warn_bad_image_once(path, context)
            return None
        if image.ndim == 3 and image.shape[2] >= 3:
            if image.shape[2] >= 4:
                rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                alpha = image[:, :, 3:4]
                return np.concatenate([rgb, alpha], axis=2)
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        return image
    except Exception as exc:
        _warn_bad_image_once(path, context, exc)
        return None


def load_rgb_image(path: str | None) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    image = load_unchanged_image(path, context="rgb image")
    if image is None:
        return None
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3]
    return None


def load_rgb_image_or_default(path: str | None, image_size: int) -> np.ndarray:
    image = load_rgb_image(path)
    if image is not None:
        return image
    return np.zeros((int(image_size), int(image_size), 3), dtype=np.uint8)


def load_optional_mask(path: str | None) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    image = load_unchanged_image(path, context="mask")
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3]
    if image.ndim == 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    return image


def load_exr_as_float32(path: str | None) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None

    geo_float = None
    try:
        import Imath
        import OpenEXR as openexr

        exr_file = openexr.InputFile(path)
        header = exr_file.header()
        data_window = header["dataWindow"]
        width = data_window.max.x - data_window.min.x + 1
        height = data_window.max.y - data_window.min.y + 1

        channels = {}
        for channel_name in ["R", "G", "B"]:
            if channel_name not in header["channels"]:
                continue
            channel_type = header["channels"][channel_name].type
            if channel_type == Imath.PixelType(Imath.PixelType.FLOAT):
                raw = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
                channels[channel_name] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            else:
                raw = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.HALF))
                channels[channel_name] = np.frombuffer(raw, dtype=np.float16).reshape(height, width).astype(np.float32)

        if channels:
            geo_float = np.stack(
                [channels.get(channel_name, np.zeros((height, width), dtype=np.float32)) for channel_name in ["R", "G", "B"]],
                axis=-1,
            )
    except ImportError:
        pass
    except Exception:
        pass

    if geo_float is None:
        try:
            import imageio.v3 as imageio

            loaded = imageio.imread(path).astype(np.float32)
            if loaded.ndim == 3 and loaded.shape[2] >= 3:
                geo_float = loaded[..., :3]
        except Exception:
            pass

    if geo_float is None:
        try:
            os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
            loaded = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if loaded is not None:
                geo_float = cv2.cvtColor(loaded.astype(np.float32), cv2.COLOR_BGR2RGB)
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


def warp_affine_black_border(
    image: np.ndarray,
    transform: np.ndarray,
    output_size: int,
    interpolation: int,
):
    if image.ndim == 2:
        border_value = 0
    else:
        border_value = (0,) * image.shape[2]
    return cv2.warpAffine(
        image,
        transform,
        (int(output_size), int(output_size)),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def apply_occlusion_mask_to_weights(
    geometry: np.ndarray | None,
    weights: np.ndarray | None,
    mask_alpha: np.ndarray | None,
    width: int,
    height: int,
) -> np.ndarray | None:
    if geometry is None or weights is None or mask_alpha is None:
        return weights

    mh, mw = mask_alpha.shape[:2]
    if mh != height or mw != width:
        mask_alpha = cv2.resize(mask_alpha, (width, height), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask_alpha > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)

    weights = weights.copy()
    pts_norm = geometry[:, 3:5]
    pts_x = (pts_norm[:, 0] * width).astype(np.int32)
    pts_y = (pts_norm[:, 1] * height).astype(np.int32)
    for idx in range(len(geometry)):
        px, py = pts_x[idx], pts_y[idx]
        if 0 <= px < width and 0 <= py < height and mask_dilated[py, px] > 0:
            weights[idx] = 0.0
    return weights


def resize_rgb_or_default(image: np.ndarray | None, image_size: int) -> np.ndarray:
    if image is None:
        return np.zeros((int(image_size), int(image_size), 3), dtype=np.uint8)
    if image.shape[0] != image_size or image.shape[1] != image_size:
        return cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
    return image


def resize_float3_or_default(image: np.ndarray | None, image_size: int) -> np.ndarray:
    if image is None:
        return np.zeros((int(image_size), int(image_size), 3), dtype=np.float32)
    if image.shape[0] != image_size or image.shape[1] != image_size:
        return cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_LINEAR)
    return image


def resize_mask_or_default(mask: np.ndarray | None, image_size: int, default_value: int = 255) -> np.ndarray:
    if mask is None:
        return np.full((int(image_size), int(image_size)), int(default_value), dtype=np.uint8)
    if mask.shape[0] != image_size or mask.shape[1] != image_size:
        return cv2.resize(mask, (int(image_size), int(image_size)), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_face_mask_to_modalities(
    facemask: np.ndarray | None,
    basecolor: np.ndarray | None = None,
    geo: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if facemask is None:
        return basecolor, geo

    mask_bin = (facemask > 127).astype(np.uint8)
    mask3 = mask_bin[:, :, None]
    if basecolor is not None:
        basecolor = basecolor * mask3
    if geo is not None:
        geo = geo * mask3.astype(np.float32, copy=False)
    return basecolor, geo
