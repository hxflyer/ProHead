import argparse
import os
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dense_image_transformer import DenseImageTransformer, compute_dense_output_channels


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
) -> tuple[torch.Tensor | None, torch.Tensor]:
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
    pred_rgb = None
    if predict_basecolor:
        pred_rgb = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_geo:
        channel_idx += 3
    if predict_normal:
        channel_idx += 3
    return pred_rgb, pred[:, channel_idx : channel_idx + 1]


def _collect_image_files(input_path: str, output_dir: str) -> list[str]:
    input_path = os.path.abspath(input_path)
    output_dir = os.path.abspath(output_dir)

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        return [input_path] if ext in SUPPORTED_IMAGE_EXTS else []

    image_files: list[str] = []
    output_dir_norm = os.path.normcase(output_dir)
    for root, dirs, files in os.walk(input_path):
        root_norm = os.path.normcase(os.path.abspath(root))
        if root_norm.startswith(output_dir_norm):
            continue

        dirs[:] = [
            d
            for d in dirs
            if not os.path.normcase(os.path.abspath(os.path.join(root, d))).startswith(output_dir_norm)
        ]

        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTS:
                image_files.append(os.path.join(root, file_name))

    image_files.sort()
    return image_files


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

    has_mask_head = "decoder.mask_head.block.1.weight" in state_dict or "decoder.mask_head.weight" in state_dict
    has_geo_head = "decoder.geo_head.block.1.weight" in state_dict or "decoder.geo_head.weight" in state_dict
    has_normal_head = "decoder.normal_head.block.1.weight" in state_dict or "decoder.normal_head.weight" in state_dict
    has_basecolor_head = "decoder.rgb_head.block.1.weight" in state_dict or "decoder.rgb_head.weight" in state_dict

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


def _save_triptych(
    image_path: str,
    input_path: str,
    output_dir: str,
    src_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    pred_mask: np.ndarray,
) -> str:
    src_rgb = np.clip(src_rgb, 0.0, 1.0)
    pred_rgb = np.clip(pred_rgb, 0.0, 1.0)
    pred_mask = np.clip(pred_mask, 0.0, 1.0)
    pred_mask_rgb = np.repeat(pred_mask[..., None], 3, axis=2)
    pred_masked_rgb = pred_rgb * pred_mask[..., None]
    canvas = np.concatenate([src_rgb, pred_rgb, pred_mask_rgb, pred_masked_rgb], axis=1)
    canvas_u8 = (canvas * 255.0).astype(np.uint8)

    input_abs = os.path.abspath(input_path)
    image_abs = os.path.abspath(image_path)
    if os.path.isdir(input_abs):
        rel_path = os.path.relpath(image_abs, input_abs)
    else:
        rel_path = os.path.basename(image_abs)

    rel_base, _ = os.path.splitext(rel_path)
    out_path = os.path.join(output_dir, f"{rel_base}_dense_pred.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(canvas_u8, cv2.COLOR_RGB2BGR))
    return out_path


@torch.no_grad()
def run_inference(args) -> None:
    device_str = str(args.device).strip() if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        if os.path.isdir(args.input_path):
            output_dir = os.path.abspath(os.path.join(args.input_path, "dense_inference"))
        else:
            output_dir = os.path.abspath(
                os.path.join(os.path.dirname(args.input_path), "dense_inference")
            )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    model = _load_model(args, device=device)
    image_files = _collect_image_files(args.input_path, output_dir)
    if not image_files:
        raise RuntimeError(f"No image files found under: {args.input_path}")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input path: {os.path.abspath(args.input_path)}")
    print(f"Output dir: {output_dir}")
    print(f"Images found: {len(image_files)}")

    for image_path in tqdm(image_files, desc="Dense Inference"):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[Warn] Skipping unreadable file: {image_path}")
            continue

        orig_h, orig_w = image_bgr.shape[:2]
        rgb_in = _prepare_input(image_bgr, image_size=int(args.image_size), device=device)
        pred = model(rgb_in)
        pred_rgb, pred_mask_logits = _split_dense_prediction(
            pred,
            predict_basecolor=bool(getattr(model, "predict_basecolor", True)),
            predict_geo=bool(getattr(model, "predict_geo", True)),
            predict_normal=bool(getattr(model, "predict_normal", True)),
        )
        if pred_rgb is None:
            raise ValueError("This checkpoint does not predict basecolor/RGB, so RGB preview export is unavailable.")

        pred_rgb = F.interpolate(
            pred_rgb,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = torch.sigmoid(pred_mask_logits)
        pred_mask = F.interpolate(
            pred_mask,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        pred_rgb_np = pred_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
        pred_mask_np = pred_mask[0, 0].detach().cpu().numpy()
        src_rgb_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        _save_triptych(
            image_path=image_path,
            input_path=args.input_path,
            output_dir=output_dir,
            src_rgb=src_rgb_np,
            pred_rgb=pred_rgb_np,
            pred_mask=pred_mask_np,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense image RGB+mask inference")
    parser.add_argument("--input_path", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="best_dense_image_transformer_ch10.pth")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--transformer_map_size", type=int, default=32)
    parser.add_argument(
        "--backbone_weights",
        type=str,
        default="imagenet",
        choices=["imagenet", "dinov3"],
    )
    parser.add_argument("--device", type=str, default="")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
