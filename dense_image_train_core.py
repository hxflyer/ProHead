import argparse
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dense_image_dataset import DenseImageDataset, dense_image_collate_fn
from dense_image_transformer import DenseImageTransformer


@dataclass
class DataConfig:
    data_roots: list[str]
    batch_size: int = 2
    num_workers: int = 4
    prefetch_factor: int = 1
    persistent_workers: bool = True
    image_size: int = 512
    train_ratio: float = 0.95
    max_train_samples: int = 0


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    output_channels: int = 3
    transformer_map_size: int = 32
    backbone_weights: str = "imagenet"


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 50
    amp_dtype: str = "fp16"
    load_model: str = ""
    save_path: str = "best_dense_image_transformer_ch3.pth"
    save_preview_every: int = 1
    basecolor_fg_weight: float = 8.0
    basecolor_fg_threshold: float = 0.02


def _parse_bool_arg(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def create_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument(
        "--persistent_workers",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--train_ratio", type=float, default=0.95)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_path", type=str, default="best_dense_image_transformer_ch3.pth")
    parser.add_argument("--save_preview_every", type=int, default=1)
    parser.add_argument("--basecolor_fg_weight", type=float, default=8.0)
    parser.add_argument("--basecolor_fg_threshold", type=float, default=0.02)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--output_channels", type=int, default=3)
    parser.add_argument("--transformer_map_size", type=int, default=32)
    parser.add_argument(
        "--backbone_weights", type=str, default="imagenet", choices=["imagenet", "dinov3"]
    )

    parser.add_argument("--data_roots", nargs="*", default=None)
    return parser


def _default_data_roots(platform_name: str) -> list[str]:
    if platform_name == "linux":
        return [
            "/hy-tmp/CapturedFrames_final1_processed",
            "/hy-tmp/CapturedFrames_final7_processed",
            "/hy-tmp/CapturedFrames_final8_processed",
            "/hy-tmp/CapturedFrames_final9_processed",
        ]
    return [
        "G:/CapturedFrames_final1_processed",
        "G:/CapturedFrames_final7_processed",
        "G:/CapturedFrames_final8_processed",
        "G:/CapturedFrames_final9_processed",
    ]


def build_configs_from_args(args, platform_name: str) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_roots = list(args.data_roots) if args.data_roots else _default_data_roots(platform_name)
    data_cfg = DataConfig(
        data_roots=data_roots,
        batch_size=int(args.batch_size),
        max_train_samples=int(args.max_train_samples),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        image_size=int(args.image_size),
        train_ratio=float(args.train_ratio),
    )
    model_cfg = ModelConfig(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        output_channels=int(args.output_channels),
        transformer_map_size=int(args.transformer_map_size),
        backbone_weights=str(args.backbone_weights),
    )
    train_cfg = TrainConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        amp_dtype=str(args.amp_dtype),
        load_model=str(args.load_model),
        save_path=str(args.save_path),
        save_preview_every=max(1, int(args.save_preview_every)),
        basecolor_fg_weight=float(args.basecolor_fg_weight),
        basecolor_fg_threshold=float(args.basecolor_fg_threshold),
    )
    return data_cfg, model_cfg, train_cfg


def create_dataloaders(data_cfg: DataConfig):
    train_dataset = DenseImageDataset(
        data_roots=data_cfg.data_roots,
        split="train",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=True,
    )
    val_dataset = DenseImageDataset(
        data_roots=data_cfg.data_roots,
        split="val",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=False,
    )

    if int(data_cfg.max_train_samples) > 0 and len(train_dataset) > int(data_cfg.max_train_samples):
        train_dataset = Subset(train_dataset, list(range(int(data_cfg.max_train_samples))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=dense_image_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=dense_image_collate_fn,
    )
    return train_loader, val_loader


def _normalize_imagenet(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _basecolor_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid: torch.Tensor | None,
    fg_weight: float = 8.0,
    fg_threshold: float = 0.02,
) -> torch.Tensor:
    """
    Weighted basecolor L1:
    - background pixels keep weight 1
    - foreground (non-dark) pixels get higher weight
    This avoids the trivial all-black solution on sparse basecolor targets.
    """
    l1 = (pred - gt).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    fg_mask = (gt.mean(dim=1, keepdim=True) > float(fg_threshold)).to(dtype=l1.dtype)
    pix_w = 1.0 + (float(max(1.0, fg_weight)) - 1.0) * fg_mask

    if valid is None:
        denom = pix_w.sum()
        if denom.detach().item() <= 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return (l1 * pix_w).sum() / (denom + 1e-6)

    sample_w = valid.float().view(-1, 1, 1, 1).to(dtype=l1.dtype, device=l1.device)
    pix_w = pix_w * sample_w
    denom = pix_w.sum()
    if denom.detach().item() <= 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return (l1 * pix_w).sum() / (denom + 1e-6)


def _save_preview(epoch: int, batch, pred: torch.Tensor, output_dir: str = "training_samples_dense"):
    os.makedirs(output_dir, exist_ok=True)
    rgb = batch["rgb"][:4].detach().cpu().numpy()
    gt = batch["basecolor"][:4].detach().cpu().numpy()
    pred_np = pred[:4].detach().cpu().numpy()
    valid = batch["basecolor_valid"][:4].detach().cpu().numpy()

    rows = []
    for i in range(rgb.shape[0]):
        rgb_i = np.clip(np.transpose(rgb[i], (1, 2, 0)), 0.0, 1.0)
        gt_i = np.clip(np.transpose(gt[i], (1, 2, 0)), 0.0, 1.0)
        pred_i = np.clip(np.transpose(pred_np[i], (1, 2, 0)), 0.0, 1.0)
        diff_i = np.abs(pred_i - gt_i)
        if float(valid[i]) <= 0.5:
            gt_i = np.zeros_like(gt_i)
            diff_i = np.zeros_like(diff_i)
        row = np.concatenate([rgb_i, gt_i, pred_i, diff_i], axis=1)
        rows.append((row * 255.0).astype(np.uint8))

    if rows:
        canvas = np.concatenate(rows, axis=0)
        out_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}_dense_preview.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def train_one_epoch(
    model: DenseImageTransformer,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_dtype,
    device: torch.device,
    train_cfg: TrainConfig,
):
    model.train()
    total_loss = 0.0
    n = 0
    preview_batch = None
    preview_pred = None

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["basecolor"].to(device, non_blocking=True)
        valid = batch["basecolor_valid"].to(device, non_blocking=True)

        rgb_in = F.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=True)
        rgb_in = _normalize_imagenet(rgb_in)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(rgb_in)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            loss = _basecolor_loss(
                pred.float(),
                gt.float(),
                valid,
                fg_weight=train_cfg.basecolor_fg_weight,
                fg_threshold=train_cfg.basecolor_fg_threshold,
            )

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += float(loss.item())
        n += 1
        pbar.set_postfix({"loss": f"{(total_loss / max(n, 1)):.4f}"})

        if preview_batch is None:
            preview_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            preview_pred = pred.detach().clone()

    return total_loss / max(n, 1), preview_batch, preview_pred


@torch.no_grad()
def validate_one_epoch(
    model: DenseImageTransformer,
    loader: DataLoader,
    device: torch.device,
    train_cfg: TrainConfig,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["basecolor"].to(device, non_blocking=True)
        valid = batch["basecolor_valid"].to(device, non_blocking=True)

        rgb_in = F.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=True)
        rgb_in = _normalize_imagenet(rgb_in)
        pred = model(rgb_in)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        loss = _basecolor_loss(
            pred.float(),
            gt.float(),
            valid,
            fg_weight=train_cfg.basecolor_fg_weight,
            fg_threshold=train_cfg.basecolor_fg_threshold,
        )
        if torch.isfinite(loss):
            total_loss += float(loss.item())
            n += 1
    return total_loss / max(n, 1)


def run_training(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, device: torch.device):
    train_loader, val_loader = create_dataloaders(data_cfg)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = DenseImageTransformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        output_channels=model_cfg.output_channels,
        output_size=data_cfg.image_size,
        transformer_map_size=model_cfg.transformer_map_size,
        backbone_weights=model_cfg.backbone_weights,
    ).to(device)

    if train_cfg.load_model and os.path.exists(train_cfg.load_model):
        ckpt = torch.load(train_cfg.load_model, map_location="cpu")
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {train_cfg.load_model}")

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_cfg.epochs), eta_min=1e-6
    )

    amp_dtype = torch.float16 if train_cfg.amp_dtype == "fp16" else torch.bfloat16
    if device.type != "cuda":
        amp_dtype = torch.float32
    elif amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    run_name = f"dense_img_{model_cfg.backbone_weights}_{int(time.time())}"
    writer = SummaryWriter(os.path.join("runs", run_name))

    best_val = float("inf")
    for epoch in range(train_cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)

        train_loss, preview_batch, preview_pred = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            train_cfg=train_cfg,
        )
        val_loss = validate_one_epoch(model=model, loader=val_loader, device=device, train_cfg=train_cfg)
        scheduler.step()

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            writer.add_scalar("GPU/PeakGB", peak_gb, epoch)
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        if preview_batch is not None and preview_pred is not None and ((epoch + 1) % train_cfg.save_preview_every == 0):
            _save_preview(epoch, preview_batch, preview_pred)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "output_channels": model_cfg.output_channels,
                },
                train_cfg.save_path,
            )
            print(f"Saved best model: {train_cfg.save_path}")

    writer.close()


def launch_windows(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args, platform_name="windows")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = bool(device.type == "cuda")
    torch.set_float32_matmul_precision("high")
    print(f"Training device: {device}")
    run_training(data_cfg, model_cfg, train_cfg, device=device)


def launch_linux(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args, platform_name="linux")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = bool(device.type == "cuda")
    torch.set_float32_matmul_precision("high")
    print(f"Training device: {device}")
    run_training(data_cfg, model_cfg, train_cfg, device=device)
