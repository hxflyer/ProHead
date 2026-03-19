"""
Prepare geometry training dataset from FFHQ images.
Runs dense image prediction and saves outputs in the format expected by metahuman_geometry_dataset.py.

Searches only 2nd level subfolders (e.g., ffhq/0, ffhq/1) and saves outputs directly there.

Output structure per subfolder:
- basecolor/BaseColor_{id}.png (masked basecolor with soft mask)
- geo/Geo_{id}.exr (masked geometry in EXR format with soft mask)
- normal/ScreenNormal_{id}.png (masked normal with soft mask)
- facemask/Face_Mask_{id}.png (predicted mask)
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from dense_image_transformer import DenseImageTransformer, compute_dense_output_channels

# Try to import OpenEXR for EXR writing
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not installed. Will use OpenCV for EXR writing (limited support).")


SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def _normalize_imagenet(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _split_dense_prediction(
    pred: torch.Tensor,
    predict_basecolor: bool = True,
    predict_geo: bool = True,
    predict_normal: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Split dense prediction into basecolor, geo, detail normal, geometry normal, and mask."""
    if pred.ndim != 4:
        raise ValueError(f"Expected model output [B, C, H, W], got {tuple(pred.shape)}")
    expected_channels = compute_dense_output_channels(
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )
    if int(pred.shape[1]) != expected_channels:
        raise ValueError(
            f"Expected model output with {expected_channels} channels, got {tuple(pred.shape)}"
        )

    channel_idx = 0
    pred_basecolor = None
    pred_geo = None
    pred_detail_normal = None
    pred_geometry_normal = None

    if predict_basecolor:
        pred_basecolor = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_geo:
        pred_geo = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_normal:
        pred_geometry_normal = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
        pred_detail_normal = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    pred_mask_logits = pred[:, channel_idx : channel_idx + 1]

    return pred_basecolor, pred_geo, pred_detail_normal, pred_geometry_normal, pred_mask_logits


def _infer_model_config(
    state_dict: dict[str, torch.Tensor],
    fallback_nhead: int,
    checkpoint_meta: dict | None = None,
) -> dict[str, int | bool]:
    if "fusion_proj.weight" not in state_dict:
        raise KeyError("Checkpoint is missing fusion_proj.weight; cannot infer d_model.")

    d_model = int(state_dict["fusion_proj.weight"].shape[0])
    layer_pattern = re.compile(r"^transformer_layers\.(\d+)\.")
    layer_indices = []
    for key in state_dict.keys():
        match = layer_pattern.match(key)
        if match is not None:
            layer_indices.append(int(match.group(1)))
    num_layers = max(layer_indices) + 1 if layer_indices else 4

    # Check for both DenseImageDecoder (*_head) and MultiTaskFPNDecoder (*_branch) naming conventions
    has_mask_head = (
        "decoder.mask_head.block.1.weight" in state_dict
        or "decoder.mask_head.weight" in state_dict
        or "decoder.mask_branch.refine_and_predict.0.block.1.weight" in state_dict
    )
    has_geo_head = (
        "decoder.geo_head.block.1.weight" in state_dict
        or "decoder.geo_head.weight" in state_dict
        or "decoder.geo_branch.refine_and_predict.0.block.1.weight" in state_dict
    )
    has_normal_head = (
        "decoder.normal_head.block.1.weight" in state_dict
        or "decoder.normal_head.weight" in state_dict
        or "decoder.normal_branch.refine_and_predict.0.block.1.weight" in state_dict
    )
    has_basecolor_head = (
        "decoder.rgb_head.block.1.weight" in state_dict
        or "decoder.rgb_head.weight" in state_dict
        or "decoder.rgb_branch.refine_and_predict.0.block.1.weight" in state_dict
    )

    if checkpoint_meta is None:
        checkpoint_meta = {}

    legacy_shared_dense_channels = None
    if (
        "decoder.rgb_head.weight" in state_dict
        and not has_geo_head
        and not has_normal_head
    ):
        legacy_shared_dense_channels = int(state_dict["decoder.rgb_head.weight"].shape[0])

    if "predict_basecolor" in checkpoint_meta:
        predict_basecolor = bool(checkpoint_meta["predict_basecolor"])
    else:
        predict_basecolor = bool(has_basecolor_head)

    if "predict_geo" in checkpoint_meta:
        predict_geo = bool(checkpoint_meta["predict_geo"])
    else:
        predict_geo = bool(has_geo_head)

    if "predict_normal" in checkpoint_meta:
        predict_normal = bool(checkpoint_meta["predict_normal"])
    else:
        predict_normal = bool(has_normal_head)

    if legacy_shared_dense_channels is not None and not any(
        key in checkpoint_meta for key in ("predict_basecolor", "predict_geo", "predict_normal")
    ):
        if legacy_shared_dense_channels >= 3:
            predict_basecolor = True
        if legacy_shared_dense_channels >= 6:
            predict_geo = True
        if legacy_shared_dense_channels >= 9:
            predict_normal = True

    if not has_mask_head:
        raise ValueError("Checkpoint is missing the mandatory mask head.")

    output_channels = compute_dense_output_channels(
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )

    if d_model % max(int(fallback_nhead), 1) != 0:
        raise ValueError(
            f"d_model={d_model} is not divisible by nhead={fallback_nhead}. "
            "Pass a compatible --nhead value."
        )

    return {
        "d_model": d_model,
        "num_layers": num_layers,
        "output_channels": output_channels,
        "predict_basecolor": predict_basecolor,
        "predict_geo": predict_geo,
        "predict_normal": predict_normal,
    }


def _load_matching_state_dict(
    model: DenseImageTransformer,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in model_state or model_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        filtered_state[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    return skipped_keys, list(missing_keys) + list(unexpected_keys)


def _load_model(args, device: torch.device) -> DenseImageTransformer:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    checkpoint_meta = checkpoint if isinstance(checkpoint, dict) else {}
    inferred = _infer_model_config(state_dict, fallback_nhead=args.nhead, checkpoint_meta=checkpoint_meta)
    model = DenseImageTransformer(
        d_model=inferred["d_model"],
        nhead=int(args.nhead),
        num_layers=inferred["num_layers"],
        predict_basecolor=bool(inferred["predict_basecolor"]),
        predict_geo=bool(inferred["predict_geo"]),
        predict_normal=bool(inferred["predict_normal"]),
        output_size=int(args.image_size),
        transformer_map_size=int(args.transformer_map_size),
        backbone_weights=str(args.backbone_weights),
    ).to(device)

    skipped_keys, load_notes = _load_matching_state_dict(model, state_dict)
    if skipped_keys:
        print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
    if load_notes:
        print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")

    model.eval()
    return model


def _prepare_input(image_bgr: np.ndarray, image_size: int, device: torch.device) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(
        image_rgb,
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    rgb = rgb.to(device=device, non_blocking=True)
    return _normalize_imagenet(rgb)


def _save_output_image(data: np.ndarray, out_path: str, is_rgb: bool = True) -> str:
    """Save a numpy array as an image file."""
    data = np.clip(data, 0.0, 1.0)
    data_u8 = (data * 255.0).astype(np.uint8)
    if is_rgb and data_u8.ndim == 3:
        cv2.imwrite(out_path, cv2.cvtColor(data_u8, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(out_path, data_u8)
    return out_path


def _save_exr_opencv(data: np.ndarray, out_path: str) -> str:
    """Save a numpy array as EXR using OpenCV."""
    # OpenCV expects BGR for 3-channel images
    if data.ndim == 3 and data.shape[2] == 3:
        data_bgr = cv2.cvtColor(np.clip(data, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2BGR)
    else:
        data_bgr = np.clip(data, 0.0, 1.0).astype(np.float32)
    cv2.imwrite(out_path, data_bgr)
    return out_path


def _save_exr_openexr(data: np.ndarray, out_path: str) -> str:
    """Save a numpy array as EXR using OpenEXR library (higher precision)."""
    h, w = data.shape[:2]
    header = OpenEXR.Header(w, h)

    # Set float pixel type (32-bit float)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header['channels'] = {
        'R': Imath.Channel(pt),
        'G': Imath.Channel(pt),
        'B': Imath.Channel(pt)
    }

    # Clip and convert to float32
    data = np.clip(data, 0.0, 1.0).astype(np.float32)

    # Split channels and convert to bytes
    r = data[:, :, 0].tobytes()
    g = data[:, :, 1].tobytes()
    b = data[:, :, 2].tobytes()

    exr_file = OpenEXR.OutputFile(out_path, header)
    exr_file.writePixels({'R': r, 'G': g, 'B': b})
    exr_file.close()

    return out_path


def _save_exr(data: np.ndarray, out_path: str) -> str:
    """Save a numpy array as EXR format."""
    if HAS_OPENEXR:
        return _save_exr_openexr(data, out_path)
    else:
        return _save_exr_opencv(data, out_path)


# Folders that are generated outputs (skip when searching for input images)
OUTPUT_FOLDERS = {"geo", "basecolor", "normal", "facemask"}


def _collect_image_files(input_path: str, max_images: int = -1) -> list[tuple[str, str, str]]:
    """
    Collect image files from 2nd level subfolders only (e.g., ffhq/0, ffhq/1).
    Does not search deeper nested folders and skips generated output folders.

    Returns:
        List of (image_path, subfolder_name, image_id) tuples.
        subfolder_name is the 2nd level folder name (e.g., "0", "1")
        image_id is the filename without extension (e.g., "00000")
    """
    input_path = os.path.abspath(input_path)
    image_files = []

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTS:
            parent = os.path.basename(os.path.dirname(input_path))
            image_id = os.path.splitext(os.path.basename(input_path))[0]
            return [(input_path, parent, image_id)]
        return []

    # Only search 2nd level subfolders (ffhq/0, ffhq/1, etc.)
    # List immediate subdirectories, excluding output folders
    subfolders = []
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path) and item not in OUTPUT_FOLDERS:
            subfolders.append(item)

    subfolders.sort()

    # Search each subfolder for images (but not deeper)
    for subfolder in subfolders:
        subfolder_path = os.path.join(input_path, subfolder)

        for file_name in sorted(os.listdir(subfolder_path)):
            # Skip directories (like generated output folders if they exist)
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isdir(file_path):
                continue

            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                image_id = os.path.splitext(file_name)[0]
                image_files.append((file_path, subfolder, image_id))

    image_files.sort()

    if max_images > 0:
        image_files = image_files[:max_images]

    return image_files


# ---------------------------------------------------------------------------
# Multi-GPU batch inference helpers
# ---------------------------------------------------------------------------

def _load_one(item_and_size: tuple) -> tuple:
    """Load and preprocess one image to a normalized CHW float32 array.
    Returns (arr_chw | None, orig_h, orig_w, image_path, subfolder, image_id).
    Called from a ThreadPoolExecutor — no GPU ops here.
    """
    (image_path, subfolder, image_id), image_size = item_and_size
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None, 0, 0, image_path, subfolder, image_id
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    arr = ((arr - mean) / std).transpose(2, 0, 1)  # CHW
    return arr, orig_h, orig_w, image_path, subfolder, image_id


def _tensor_to_numpy_hwc(t: torch.Tensor, orig_h: int, orig_w: int) -> np.ndarray:
    """Resize [1, C, H, W] tensor to original size and return [H, W, C] float32 on CPU."""
    if t.shape[-2] != orig_h or t.shape[-1] != orig_w:
        t = F.interpolate(t.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    return t[0].permute(1, 2, 0).float().cpu().numpy()


def _save_item(
    out_dir: str,
    image_id: str,
    bc_np: np.ndarray | None,
    geo_np: np.ndarray | None,
    norm_np: np.ndarray | None,
    mask_np: np.ndarray,
) -> None:
    """Write all outputs for one image. Safe to call from a ThreadPoolExecutor."""
    basecolor_dir = os.path.join(out_dir, "basecolor")
    geo_dir       = os.path.join(out_dir, "geo")
    normal_dir    = os.path.join(out_dir, "normal")
    facemask_dir  = os.path.join(out_dir, "facemask")
    os.makedirs(basecolor_dir, exist_ok=True)
    os.makedirs(geo_dir,       exist_ok=True)
    os.makedirs(normal_dir,    exist_ok=True)
    os.makedirs(facemask_dir,  exist_ok=True)

    mask_2d   = mask_np[:, :, 0]
    mask_3ch  = np.stack([mask_2d, mask_2d, mask_2d], axis=-1)

    _save_output_image(mask_2d, os.path.join(facemask_dir, f"Face_Mask_{image_id}.png"), is_rgb=False)

    if bc_np is not None:
        _save_output_image(
            np.clip(bc_np, 0.0, 1.0) * mask_3ch,
            os.path.join(basecolor_dir, f"BaseColor_{image_id}.png"),
            is_rgb=True,
        )
    if geo_np is not None:
        _save_exr(
            np.clip(geo_np, 0.0, 1.0) * mask_3ch,
            os.path.join(geo_dir, f"Geo_{image_id}.exr"),
        )
    if norm_np is not None:
        neutral = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        _save_output_image(
            np.clip(norm_np, 0.0, 1.0) * mask_3ch + neutral * (1.0 - mask_3ch),
            os.path.join(normal_dir, f"ScreenNormal_{image_id}.png"),
            is_rgb=True,
        )


def _outputs_exist(out_dir: str, image_id: str, predict_basecolor: bool, predict_geo: bool, predict_normal: bool) -> bool:
    """Return True if all expected output files already exist for this image."""
    if predict_geo and not os.path.exists(os.path.join(out_dir, "geo", f"Geo_{image_id}.exr")):
        return False
    if predict_normal and not os.path.exists(os.path.join(out_dir, "normal", f"ScreenNormal_{image_id}.png")):
        return False
    if predict_basecolor and not os.path.exists(os.path.join(out_dir, "basecolor", f"BaseColor_{image_id}.png")):
        return False
    return True


def _gpu_worker(rank: int, num_gpus: int, args, image_files: list) -> None:
    """Per-GPU worker: processes a round-robin shard of image_files on cuda:{rank}."""
    my_files = image_files[rank::num_gpus]
    if not my_files:
        return

    device = torch.device(f"cuda:{rank}")
    model  = _load_model(args, device)
    model.eval()

    predict_basecolor = bool(getattr(model, "predict_basecolor", True))
    predict_geo       = bool(getattr(model, "predict_geo",       True))
    predict_normal    = bool(getattr(model, "predict_normal",    True))

    input_root = os.path.abspath(args.input_path)
    batch_size = int(args.batch_size)
    image_size = int(args.image_size)
    io_workers = int(args.io_workers)

    pbar = tqdm(total=len(my_files), desc=f"GPU {rank}", position=rank, leave=True)

    # Max number of batches worth of numpy buffers allowed to sit in the save queue.
    # Keeps RAM bounded: each image is ~8 MB of numpy, so 4 batches * 8 imgs = ~256 MB per GPU.
    max_pending_batches = 4

    with ThreadPoolExecutor(max_workers=io_workers) as save_pool:
        pending_saves = []

        for batch_start in range(0, len(my_files), batch_size):
            batch_items = my_files[batch_start : batch_start + batch_size]

            # ---- Skip already-processed images ----
            filtered_items = []
            skip_count = 0
            for item in batch_items:
                image_path, subfolder, image_id = item
                out_dir = os.path.join(input_root, subfolder) if subfolder else input_root
                if _outputs_exist(out_dir, image_id, predict_basecolor, predict_geo, predict_normal):
                    skip_count += 1
                else:
                    filtered_items.append(item)
            if skip_count:
                pbar.update(skip_count)
            batch_items = filtered_items
            if not batch_items:
                continue

            # ---- Load batch in parallel (CPU threads) ----
            with ThreadPoolExecutor(max_workers=min(len(batch_items), io_workers)) as load_pool:
                load_results = list(load_pool.map(
                    _load_one,
                    [(item, image_size) for item in batch_items],
                ))

            valid_arrs = []
            valid_meta = []
            for arr, orig_h, orig_w, image_path, subfolder, image_id in load_results:
                if arr is None:
                    print(f"[Warn GPU{rank}] Skipping unreadable: {image_path}")
                    continue
                valid_arrs.append(arr)
                valid_meta.append((subfolder, image_id, orig_h, orig_w))

            if not valid_arrs:
                continue

            # ---- Batch GPU inference ----
            batch_tensor = torch.from_numpy(np.stack(valid_arrs, axis=0)).to(device, non_blocking=True)
            with torch.no_grad():
                preds = model(batch_tensor)

            pred_bc, pred_geo, pred_detail_norm, pred_geometry_norm, pred_mask_logits = _split_dense_prediction(
                preds, predict_basecolor, predict_geo, predict_normal
            )
            pred_mask = torch.sigmoid(pred_mask_logits)

            # ---- Dispatch saves to thread pool ----
            for i, (subfolder, image_id, orig_h, orig_w) in enumerate(valid_meta):
                out_dir  = os.path.join(input_root, subfolder) if subfolder else input_root
                mask_np  = _tensor_to_numpy_hwc(pred_mask[i:i+1],     orig_h, orig_w)
                bc_np    = _tensor_to_numpy_hwc(pred_bc[i:i+1],       orig_h, orig_w) if pred_bc   is not None else None
                geo_np   = _tensor_to_numpy_hwc(pred_geo[i:i+1],      orig_h, orig_w) if pred_geo  is not None else None
                if pred_detail_norm is not None and pred_geometry_norm is not None:
                    detail_np = _tensor_to_numpy_hwc(pred_detail_norm[i:i+1], orig_h, orig_w)
                    geometry_np = _tensor_to_numpy_hwc(pred_geometry_norm[i:i+1], orig_h, orig_w)
                    norm_np = np.clip(detail_np + geometry_np, 0.0, 1.0)
                else:
                    norm_np = None
                pending_saves.append(
                    save_pool.submit(_save_item, out_dir, image_id, bc_np, geo_np, norm_np, mask_np)
                )

            pbar.update(len(valid_meta))

            # ---- Drain completed futures to free numpy buffers ----
            # If save queue is too deep, wait for the oldest batch before continuing.
            if len(pending_saves) >= max_pending_batches * batch_size:
                drain_count = batch_size  # wait for one batch worth
                for fut in pending_saves[:drain_count]:
                    fut.result()
                pending_saves = pending_saves[drain_count:]

        # Wait for remaining saves to complete before exiting
        for fut in pending_saves:
            fut.result()

    pbar.close()


@torch.no_grad()
def run_inference(args) -> None:
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    image_files = _collect_image_files(args.input_path, max_images=args.max_images)
    if not image_files:
        raise RuntimeError(f"No image files found under: {args.input_path}")

    num_gpus = torch.cuda.device_count()
    if args.num_gpus > 0:
        num_gpus = min(args.num_gpus, num_gpus)
    num_gpus = max(num_gpus, 1)

    print(f"Input path:  {os.path.abspath(args.input_path)}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Images:      {len(image_files)}")
    print(f"GPUs:        {num_gpus}  |  batch/GPU: {args.batch_size}  |  I/O threads/GPU: {args.io_workers}")

    if num_gpus == 1:
        _gpu_worker(0, 1, args, image_files)
    else:
        mp.spawn(
            _gpu_worker,
            args=(num_gpus, args, image_files),
            nprocs=num_gpus,
            join=True,
        )

    print(f"\nDone! Processed {len(image_files)} images.")
    print(f"Output saved to: {os.path.abspath(args.input_path)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare geometry training dataset from FFHQ images")
    parser.add_argument("--input_path", type=str, default="/hy-tmp/ffhq",
                        help="Input path to FFHQ images folder")
    parser.add_argument("--output_dir", type=str, default="/hy-tmp/ffhq_geometry_dataset",
                        help="Output directory for geometry dataset (informational only)")
    parser.add_argument("--checkpoint", type=str, default="best_dense_image_transformer_ch10.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Model input image size")
    parser.add_argument("--nhead", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--transformer_map_size", type=int, default=32,
                        help="Transformer token map size")
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "dinov3"],
        help="Backbone pretrained weights type"
    )
    parser.add_argument("--device", type=str, default="",
                        help="Ignored in multi-GPU mode; GPUs are assigned automatically")
    parser.add_argument("--max_images", type=int, default=-1,
                        help="Maximum number of images to process (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Images per forward pass per GPU")
    parser.add_argument("--num_gpus", type=int, default=-1,
                        help="Number of GPUs to use (-1 = use all available)")
    parser.add_argument("--io_workers", type=int, default=4,
                        help="I/O threads per GPU worker for parallel image loading/saving")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
