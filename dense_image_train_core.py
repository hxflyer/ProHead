import argparse
import datetime
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dense_image_dataset import DenseImageDataset, dense_image_collate_fn
from dense_image_transformer import DenseImageTransformer, compute_dense_output_channels


class UncertaintyWeightedLoss(nn.Module):
    """
    Learnable uncertainty-based task weighting for multi-task learning.
    
    Based on: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., CVPR 2018)
    
    Each task's loss is weighted by learned uncertainty (log-variance):
        weighted_loss = L / (2 * exp(log_sigma)) + log_sigma
    
    The network automatically learns optimal task weights through gradient descent.
    """
    
    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters (initialized to 0, so sigma^2 = 1)
        self.log_var_rgb = nn.Parameter(torch.zeros(1))
        self.log_var_geo = nn.Parameter(torch.zeros(1))
        self.log_var_normal = nn.Parameter(torch.zeros(1))
        self.log_var_mask = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        rgb_loss: torch.Tensor,
        geo_loss: torch.Tensor,
        normal_loss: torch.Tensor,
        mask_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply uncertainty weighting to combine multiple task losses.
        
        Args:
            rgb_loss: Basecolor prediction loss
            geo_loss: Geometry prediction loss
            normal_loss: Normal prediction loss
            mask_loss: Mask prediction loss
            
        Returns:
            Combined weighted loss
        """
        # Apply uncertainty weighting: L / (2*sigma^2) + log(sigma^2)
        weighted_rgb = rgb_loss / (2 * self.log_var_rgb.exp()) + self.log_var_rgb
        weighted_geo = geo_loss / (2 * self.log_var_geo.exp()) + self.log_var_geo
        weighted_normal = normal_loss / (2 * self.log_var_normal.exp()) + self.log_var_normal
        weighted_mask = mask_loss / (2 * self.log_var_mask.exp()) + self.log_var_mask
        
        return weighted_rgb + weighted_geo + weighted_normal + weighted_mask
    
    def get_uncertainties(self) -> dict[str, float]:
        """Get current learned uncertainties (sigma^2) for each task."""
        return {
            'rgb_sigma2': self.log_var_rgb.exp().item(),
            'geo_sigma2': self.log_var_geo.exp().item(),
            'normal_sigma2': self.log_var_normal.exp().item(),
            'mask_sigma2': self.log_var_mask.exp().item(),
        }
    
    def get_weights(self) -> dict[str, float]:
        """Get effective task weights (1 / 2*sigma^2) for monitoring."""
        return {
            'rgb_weight': 1.0 / (2 * self.log_var_rgb.exp().item()),
            'geo_weight': 1.0 / (2 * self.log_var_geo.exp().item()),
            'normal_weight': 1.0 / (2 * self.log_var_normal.exp().item()),
            'mask_weight': 1.0 / (2 * self.log_var_mask.exp().item()),
        }


@dataclass
class DataConfig:
    data_roots: list[str]
    batch_size: int = 2
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = True
    image_size: int = 1024
    train_ratio: float = 0.95
    max_train_samples: int = 0


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    predict_basecolor: bool = True
    predict_geo: bool = True
    predict_normal: bool = True
    output_channels: int = 13
    transformer_map_size: int = 32
    backbone_weights: str = "imagenet"


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 50
    amp_dtype: str = "fp16"
    load_model: str = ""
    save_path: str = "best_dense_image_transformer_ch13.pth"
    save_preview_every: int = 1
    basecolor_fg_weight: float = 8.0
    basecolor_fg_threshold: float = 0.02
    basecolor_bg_weight: float = 0.1
    mask_bce_lambda: float = 1.0
    mask_dice_lambda: float = 1.0
    debug_print_every: int = 200
    master_port: str = "12356"


def setup_distributed(rank: int, world_size: int, master_port: str, backend: str) -> None:
    """Initialize distributed process group."""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    if world_size <= 1:
        return

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    try:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


def _unwrap_model(model):
    """Unwrap DDP model to get underlying model."""
    return model.module if isinstance(model, DDP) else model


def _load_matching_state_dict(model, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
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


def worker_init_fn(_worker_id):
    """Initialize DataLoader workers - disable OpenCV threading."""
    import cv2
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--persistent_workers",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--train_ratio", type=float, default=0.95)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_path", type=str, default="best_dense_image_transformer_ch13.pth")
    parser.add_argument("--save_preview_every", type=int, default=1)
    parser.add_argument("--basecolor_fg_weight", type=float, default=8.0)
    parser.add_argument("--basecolor_fg_threshold", type=float, default=0.02)
    parser.add_argument("--basecolor_bg_weight", type=float, default=0.1)
    parser.add_argument("--mask_bce_lambda", type=float, default=1.0)
    parser.add_argument("--mask_dice_lambda", type=float, default=1.0)
    parser.add_argument("--debug_print_every", type=int, default=200)
    parser.add_argument("--master_port", type=str, default="12356")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--basecolor", action="store_true")
    parser.add_argument("--geo", action="store_true")
    parser.add_argument("--normal", action="store_true")
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


def _resolve_prediction_targets(args) -> tuple[bool, bool, bool]:
    predict_basecolor = bool(getattr(args, "basecolor", False))
    predict_geo = bool(getattr(args, "geo", False))
    predict_normal = bool(getattr(args, "normal", False))
    if not (predict_basecolor or predict_geo or predict_normal):
        return True, True, True
    return predict_basecolor, predict_geo, predict_normal


def _target_summary(predict_basecolor: bool, predict_geo: bool, predict_normal: bool) -> str:
    names = []
    if predict_basecolor:
        names.append("basecolor")
    if predict_geo:
        names.append("geo")
    if predict_normal:
        names.append("geometry_normal")
        names.append("detail_normal")
    names.append("mask")
    return "+".join(names)


def build_configs_from_args(args, platform_name: str) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_roots = list(args.data_roots) if args.data_roots else _default_data_roots(platform_name)
    predict_basecolor, predict_geo, predict_normal = _resolve_prediction_targets(args)
    output_channels = compute_dense_output_channels(
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )
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
        predict_basecolor=bool(predict_basecolor),
        predict_geo=bool(predict_geo),
        predict_normal=bool(predict_normal),
        output_channels=int(output_channels),
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
        basecolor_bg_weight=float(args.basecolor_bg_weight),
        mask_bce_lambda=float(args.mask_bce_lambda),
        mask_dice_lambda=float(args.mask_dice_lambda),
        debug_print_every=max(1, int(args.debug_print_every)),
        master_port=str(args.master_port),
    )
    return data_cfg, model_cfg, train_cfg


def create_dataloaders(data_cfg: DataConfig):
    """Create non-distributed dataloaders (for single GPU/CPU)."""
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
        num_workers=max(1, data_cfg.num_workers // 4),  # Use half the workers for validation
        pin_memory=True,
        drop_last=False,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=dense_image_collate_fn,
    )
    return train_loader, val_loader


def create_distributed_dataloaders(data_cfg: DataConfig, rank: int, world_size: int):
    """Create distributed dataloaders with DistributedSampler for multi-GPU training."""
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
        if rank == 0:
            print(f"[Info] Limiting train dataset for test run: {int(data_cfg.max_train_samples)} samples")
        train_dataset = Subset(train_dataset, list(range(int(data_cfg.max_train_samples))))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=dense_image_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        num_workers=max(1, data_cfg.num_workers // 2),  # Use half the workers for validation
        pin_memory=True,
        drop_last=False,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=dense_image_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, train_sampler


def _normalize_imagenet(rgb: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
    return (rgb - mean) / std


def _zero_scalar_like(reference: torch.Tensor | None) -> torch.Tensor:
    if reference is None:
        return torch.tensor(0.0, dtype=torch.float32)
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _model_prediction_targets(model) -> tuple[bool, bool, bool]:
    base_model = _unwrap_model(model)
    return (
        bool(getattr(base_model, "predict_basecolor", True)),
        bool(getattr(base_model, "predict_geo", True)),
        bool(getattr(base_model, "predict_normal", True)),
    )


def _split_dense_prediction(
    pred: torch.Tensor,
    predict_basecolor: bool = True,
    predict_geo: bool = True,
    predict_normal: bool = True,
) -> dict[str, torch.Tensor | None]:
    if pred.ndim != 4:
        raise ValueError(f"Expected dense prediction [B, C, H, W], got {tuple(pred.shape)}")

    expected_channels = compute_dense_output_channels(
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )
    channels = int(pred.shape[1])
    if channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} output channels for "
            f"{_target_summary(predict_basecolor, predict_geo, predict_normal)}, got {channels}"
        )

    channel_idx = 0
    pred_rgb = None
    pred_geo = None
    pred_detail_normal = None
    pred_geometry_normal = None
    if predict_basecolor:
        pred_rgb = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_geo:
        pred_geo = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
    if predict_normal:
        pred_geometry_normal = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3
        pred_detail_normal = pred[:, channel_idx : channel_idx + 3]
        channel_idx += 3

    return {
        "rgb": pred_rgb,
        "geo": pred_geo,
        "geometry_normal": pred_geometry_normal,
        "detail_normal": pred_detail_normal,
        "normal": pred_detail_normal,
        "mask_logits": pred[:, channel_idx : channel_idx + 1],
    }


def _prepare_face_mask(
    face_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    if face_mask is None:
        return None
    mask = face_mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask, size=reference.shape[-2:], mode="nearest")
    return mask


def _build_feature_loss_weight(
    reference: torch.Tensor,
    face_mask: torch.Tensor | None,
    error_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_weight = _prepare_face_mask(face_mask, reference)
    if loss_weight is None:
        loss_weight = torch.ones(
            (reference.shape[0], 1, reference.shape[-2], reference.shape[-1]),
            device=reference.device,
            dtype=reference.dtype,
        )

    if error_mask is not None:
        loss_weight = loss_weight * error_mask.to(device=reference.device, dtype=reference.dtype)

    return loss_weight


def _masked_feature_loss(
    pred_feat: torch.Tensor | None,
    gt_feat: torch.Tensor | None,
    face_mask: torch.Tensor | None,
    error_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred_feat is None or gt_feat is None:
        reference = pred_feat if pred_feat is not None else gt_feat
        return _zero_scalar_like(reference)

    loss_weight = _build_feature_loss_weight(
        reference=pred_feat,
        face_mask=face_mask,
        error_mask=error_mask,
    )
    loss_weight = loss_weight.expand_as(pred_feat)
    denom = loss_weight.sum()
    if denom.detach().item() <= 0:
        return torch.zeros((), device=pred_feat.device, dtype=pred_feat.dtype)
    return ((pred_feat - gt_feat).abs() * loss_weight).sum() / (denom + 1e-6)


def _masked_normal_cosine_loss(
    pred_normal: torch.Tensor | None,
    gt_normal: torch.Tensor | None,
    face_mask: torch.Tensor | None,
    error_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_normal is None or gt_normal is None:
        reference = pred_normal if pred_normal is not None else gt_normal
        return _zero_scalar_like(reference)

    loss_weight = _build_feature_loss_weight(
        reference=pred_normal,
        face_mask=face_mask,
        error_mask=error_mask,
    )
    denom = loss_weight.sum()
    if denom.detach().item() <= 0:
        return torch.zeros((), device=pred_normal.device, dtype=pred_normal.dtype)

    # Model output is already normalized (unit length) in [0,1] range
    # Just transform to [-1, 1] - no need to normalize again
    pred_vec = pred_normal * 2.0 - 1.0  # Already unit length from model
    
    # GT needs to be normalized
    gt_vec = F.normalize(gt_normal * 2.0 - 1.0, dim=1, eps=eps)
    
    cosine = (pred_vec * gt_vec).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    loss_map = 1.0 - cosine
    return (loss_map * loss_weight).sum() / (denom + 1e-6)


def _mask_bce_loss(
    mask_logits: torch.Tensor | None,
    gt_mask: torch.Tensor | None,
    error_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask_logits is None or gt_mask is None:
        reference = mask_logits if mask_logits is not None else gt_mask
        return _zero_scalar_like(reference)

    target = _prepare_face_mask(gt_mask, mask_logits)
    valid_weight = torch.ones_like(mask_logits, dtype=mask_logits.dtype, device=mask_logits.device)
    if error_mask is not None:
        valid_weight = valid_weight * error_mask.to(device=mask_logits.device, dtype=mask_logits.dtype)

    denom = valid_weight.sum()
    if denom.detach().item() <= 0:
        return torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)

    raw = F.binary_cross_entropy_with_logits(mask_logits, target, reduction="none")
    return (raw * valid_weight).sum() / (denom + 1e-6)


def _mask_dice_loss(
    mask_logits: torch.Tensor | None,
    gt_mask: torch.Tensor | None,
    error_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if mask_logits is None or gt_mask is None:
        reference = mask_logits if mask_logits is not None else gt_mask
        return _zero_scalar_like(reference)

    target = _prepare_face_mask(gt_mask, mask_logits)
    probs = torch.sigmoid(mask_logits)

    if error_mask is not None:
        em = error_mask.to(device=mask_logits.device, dtype=mask_logits.dtype)
        probs = probs * em
        target = target * em

    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def _combined_mask_loss(
    mask_logits: torch.Tensor | None,
    gt_mask: torch.Tensor | None,
    bce_lambda: float,
    dice_lambda: float,
    error_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_logits is None:
        zero = _zero_scalar_like(gt_mask)
        return zero, zero, zero

    mask_bce = _mask_bce_loss(mask_logits, gt_mask, error_mask=error_mask)
    mask_dice = _mask_dice_loss(mask_logits, gt_mask, error_mask=error_mask)
    mask_total = float(bce_lambda) * mask_bce + float(dice_lambda) * mask_dice
    return mask_total, mask_bce, mask_dice


def _get_detail_normal_batch_tensors(batch, device: torch.device) -> torch.Tensor | None:
    detail_normal = batch.get("detail_normal", batch.get("normal"))
    return detail_normal.to(device, non_blocking=True) if detail_normal is not None else None


def _get_geometry_normal_batch_tensors(batch, device: torch.device) -> torch.Tensor | None:
    geometry_normal = batch.get("geometry_normal")
    return geometry_normal.to(device, non_blocking=True) if geometry_normal is not None else None


def _to_vis_map(arr: np.ndarray, signed: bool = False) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if signed:
        arr = arr * 0.5 + 0.5
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr[:, :, :3]
    return np.clip(arr, 0.0, 1.0)


def _save_preview(
    epoch: int,
    batch,
    pred: torch.Tensor,
    output_dir: str = "training_samples_dense",
    predict_basecolor: bool = True,
    predict_geo: bool = True,
    predict_normal: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    rgb = batch["rgb"][:4].detach().cpu().numpy()
    gt_rgb = batch["basecolor"][:4].detach().cpu().numpy()
    gt_geo = batch["geo"][:4].detach().cpu().numpy()
    gt_detail_normal = batch.get("detail_normal", batch["normal"])[:4].detach().cpu().numpy()
    gt_geometry_normal = batch["geometry_normal"][:4].detach().cpu().numpy()
    gt_mask = batch["face_mask"][:4].detach().cpu().numpy()

    pred_parts = _split_dense_prediction(
        pred[:4],
        predict_basecolor=predict_basecolor,
        predict_geo=predict_geo,
        predict_normal=predict_normal,
    )
    pred_rgb_np = (
        pred_parts["rgb"].detach().cpu().numpy()
        if pred_parts["rgb"] is not None
        else np.zeros_like(gt_rgb, dtype=np.float32)
    )
    pred_geo_np = (
        pred_parts["geo"].detach().cpu().numpy()
        if pred_parts["geo"] is not None
        else np.zeros_like(gt_geo, dtype=np.float32)
    )
    pred_detail_normal_np = (
        pred_parts["detail_normal"].detach().cpu().numpy()
        if pred_parts["detail_normal"] is not None
        else np.zeros_like(gt_detail_normal, dtype=np.float32)
    )
    pred_geometry_normal_np = (
        pred_parts["geometry_normal"].detach().cpu().numpy()
        if pred_parts["geometry_normal"] is not None
        else np.zeros_like(gt_geometry_normal, dtype=np.float32)
    )
    if pred_parts["mask_logits"] is not None:
        pred_mask_np = torch.sigmoid(pred_parts["mask_logits"]).detach().cpu().numpy()
    else:
        pred_mask_np = np.zeros_like(gt_mask, dtype=np.float32)

    rows = []
    for i in range(rgb.shape[0]):
        rgb_i = np.clip(np.transpose(rgb[i], (1, 2, 0)), 0.0, 1.0)
        gt_rgb_i = np.clip(np.transpose(gt_rgb[i], (1, 2, 0)), 0.0, 1.0)
        gt_geo_i = np.clip(np.transpose(gt_geo[i], (1, 2, 0)), 0.0, 1.0)
        gt_detail_normal_i = np.clip(np.transpose(gt_detail_normal[i], (1, 2, 0)), -1.0, 1.0)
        gt_geometry_normal_i = np.clip(np.transpose(gt_geometry_normal[i], (1, 2, 0)), 0.0, 1.0)
        pred_rgb_i = np.clip(np.transpose(pred_rgb_np[i], (1, 2, 0)), 0.0, 1.0)
        pred_geo_i = np.clip(np.transpose(pred_geo_np[i], (1, 2, 0)), 0.0, 1.0)
        pred_detail_normal_i = np.clip(np.transpose(pred_detail_normal_np[i], (1, 2, 0)), -1.0, 1.0)
        pred_geometry_normal_i = np.clip(np.transpose(pred_geometry_normal_np[i], (1, 2, 0)), 0.0, 1.0)
        gt_mask_i = np.clip(gt_mask[i, 0], 0.0, 1.0)
        pred_mask_i = np.clip(pred_mask_np[i, 0], 0.0, 1.0)
        pred_masked_rgb_i = pred_rgb_i * pred_mask_i[..., None]
        rgb_diff_i = np.abs(pred_rgb_i - gt_rgb_i) * gt_mask_i[..., None]
        pred_masked_geo_i = pred_geo_i * pred_mask_i[..., None]
        geo_diff_i = np.abs(pred_geo_i - gt_geo_i) * gt_mask_i[..., None]
        geometry_normal_diff_i = np.abs(pred_geometry_normal_i - gt_geometry_normal_i) * gt_mask_i[..., None]
        pred_masked_detail_normal_i = pred_detail_normal_i * pred_mask_i[..., None]
        detail_normal_diff_i = np.abs(pred_detail_normal_i - gt_detail_normal_i) * 0.5 * gt_mask_i[..., None]
        gt_mask_vis = np.repeat(gt_mask_i[..., None], 3, axis=2)
        pred_mask_vis = np.repeat(pred_mask_i[..., None], 3, axis=2)
        row = np.concatenate(
            [
                _to_vis_map(rgb_i),                                         # 1  input rgb
                _to_vis_map(gt_rgb_i),                                      # 2  gt basecolor
                _to_vis_map(pred_masked_rgb_i),                             # 3  pred masked basecolor
                _to_vis_map(rgb_diff_i),                                    # 4  basecolor diff
                _to_vis_map(gt_geo_i),                                      # 5  gt geo
                _to_vis_map(pred_masked_geo_i),                             # 6  pred masked geo
                _to_vis_map(geo_diff_i),                                    # 7  geo diff
                _to_vis_map(gt_geometry_normal_i),                          # 8  gt geometry normal
                _to_vis_map(pred_geometry_normal_i),                        # 9  pred geometry normal
                _to_vis_map(geometry_normal_diff_i),                        # 10 geometry normal diff
                _to_vis_map(gt_detail_normal_i, signed=True),               # 11 gt detail normal
                _to_vis_map(pred_masked_detail_normal_i, signed=True),      # 12 pred masked detail normal
                _to_vis_map(detail_normal_diff_i),                          # 13 detail normal diff
                _to_vis_map(gt_mask_vis),                                   # 14 gt face mask
                _to_vis_map(pred_mask_vis),                                 # 15 pred face mask
            ],
            axis=1,
        )
        rows.append((row * 255.0).astype(np.uint8))

    if rows:
        canvas = np.concatenate(rows, axis=0)
        out_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}_dense_preview.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Saved preview: {out_path}")


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_dtype,
    device: torch.device,
    train_cfg: TrainConfig,
    rank: int = 0,
    world_size: int = 1,
    train_sampler: DistributedSampler = None,
    epoch: int = 0,
):
    model.train()
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_geo_loss = 0.0
    total_detail_normal_loss = 0.0
    total_geometry_normal_loss = 0.0
    total_normal_loss = 0.0
    total_mask_loss = 0.0
    total_mask_bce = 0.0
    total_mask_dice = 0.0
    n = 0
    visualization_batch = None

    # Set epoch for distributed sampler to ensure proper shuffling
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    pbar = tqdm(loader, desc="Training") if rank == 0 else loader
    predict_basecolor, predict_geo, predict_normal = _model_prediction_targets(model)
    for step, batch in enumerate(pbar):
        # Capture first batch for visualization (only rank 0, move to CPU immediately)
        if rank == 0 and visualization_batch is None:
            visualization_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        rgb = batch["rgb"].to(device, non_blocking=True)
        
        gt_rgb = batch["basecolor"].to(device, non_blocking=True) if predict_basecolor else None
        gt_geo = batch["geo"].to(device, non_blocking=True) if predict_geo else None
        gt_detail_normal = _get_detail_normal_batch_tensors(batch, device) if predict_normal else None
        gt_geometry_normal = _get_geometry_normal_batch_tensors(batch, device) if predict_normal else None
        face_mask = batch["face_mask"].to(device, non_blocking=True)
        error_mask = batch["error_mask"].to(device, non_blocking=True)

        rgb_in = _normalize_imagenet(rgb)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            pred = model(rgb_in)
            target_size = face_mask.shape[-2:]
            if pred.shape[-2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
        pred_parts = _split_dense_prediction(
            pred,
            predict_basecolor=predict_basecolor,
            predict_geo=predict_geo,
            predict_normal=predict_normal,
        )
        rgb_loss = _masked_feature_loss(
            pred_parts["rgb"].float() if pred_parts["rgb"] is not None else None,
            gt_rgb.float() if gt_rgb is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        geo_loss = _masked_feature_loss(
            pred_parts["geo"].float() if pred_parts["geo"] is not None else None,
            gt_geo.float() if gt_geo is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        detail_normal_loss = _masked_feature_loss(
            pred_parts["detail_normal"].float() if pred_parts["detail_normal"] is not None else None,
            gt_detail_normal.float() if gt_detail_normal is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        geometry_normal_loss = _masked_normal_cosine_loss(
            pred_parts["geometry_normal"].float() if pred_parts["geometry_normal"] is not None else None,
            gt_geometry_normal.float() if gt_geometry_normal is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        normal_loss = detail_normal_loss + geometry_normal_loss
        mask_loss, mask_bce, mask_dice = _combined_mask_loss(
            pred_parts["mask_logits"].float() if pred_parts["mask_logits"] is not None else None,
            face_mask.float(),
            bce_lambda=train_cfg.mask_bce_lambda,
            dice_lambda=train_cfg.mask_dice_lambda,
            error_mask=error_mask.float(),
        )
        loss = rgb_loss + geo_loss + normal_loss + mask_loss

        # Check for NaN/inf before backward - ALL RANKS must agree and follow same path
        loss_finite_local = torch.isfinite(loss).to(device=device, dtype=torch.int32)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_finite_local, op=dist.ReduceOp.MIN)
        loss_finite_global = bool(loss_finite_local.item() > 0)

        if not loss_finite_global:
            # All ranks skip this iteration together - just clear gradients and continue
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
        total_rgb_loss += float(rgb_loss.item())
        total_geo_loss += float(geo_loss.item())
        total_detail_normal_loss += float(detail_normal_loss.item())
        total_geometry_normal_loss += float(geometry_normal_loss.item())
        total_normal_loss += float(normal_loss.item())
        total_mask_loss += float(mask_loss.item())
        total_mask_bce += float(mask_bce.item())
        total_mask_dice += float(mask_dice.item())
        n += 1
        
        if rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix(
                {
                    "loss": f"{(total_loss / max(n, 1)):.4f}",
                    "rgb": f"{(total_rgb_loss / max(n, 1)):.4f}",
                    "geo": f"{(total_geo_loss / max(n, 1)):.4f}",
                    "nrm": f"{(total_normal_loss / max(n, 1)):.4f}",
                    "mask": f"{(total_mask_loss / max(n, 1)):.4f}",
                }
            )

    # Aggregate loss across all ranks for distributed training
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(
            [
                total_loss,
                total_rgb_loss,
                total_geo_loss,
                total_detail_normal_loss,
                total_geometry_normal_loss,
                total_normal_loss,
                total_mask_loss,
                total_mask_bce,
                total_mask_dice,
                n,
            ],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        total_rgb_loss = loss_tensor[1].item()
        total_geo_loss = loss_tensor[2].item()
        total_detail_normal_loss = loss_tensor[3].item()
        total_geometry_normal_loss = loss_tensor[4].item()
        total_normal_loss = loss_tensor[5].item()
        total_mask_loss = loss_tensor[6].item()
        total_mask_bce = loss_tensor[7].item()
        total_mask_dice = loss_tensor[8].item()
        n = loss_tensor[9].item()

    denom = max(n, 1)
    return {
        "avg_loss": total_loss / denom,
        "avg_rgb_loss": total_rgb_loss / denom,
        "avg_geo_loss": total_geo_loss / denom,
        "avg_detail_normal_loss": total_detail_normal_loss / denom,
        "avg_geometry_normal_loss": total_geometry_normal_loss / denom,
        "avg_normal_loss": total_normal_loss / denom,
        "avg_mask_loss": total_mask_loss / denom,
        "avg_mask_bce": total_mask_bce / denom,
        "avg_mask_dice": total_mask_dice / denom,
        "visualization_batch": visualization_batch,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    train_cfg: TrainConfig,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    model.eval()
    predict_basecolor, predict_geo, predict_normal = _model_prediction_targets(model)
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_geo_loss = 0.0
    total_detail_normal_loss = 0.0
    total_geometry_normal_loss = 0.0
    total_normal_loss = 0.0
    total_mask_loss = 0.0
    total_mask_bce = 0.0
    total_mask_dice = 0.0
    n = 0
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        
        gt_rgb = batch["basecolor"].to(device, non_blocking=True) if predict_basecolor else None
        gt_geo = batch["geo"].to(device, non_blocking=True) if predict_geo else None
        gt_detail_normal = _get_detail_normal_batch_tensors(batch, device) if predict_normal else None
        gt_geometry_normal = _get_geometry_normal_batch_tensors(batch, device) if predict_normal else None
        face_mask = batch["face_mask"].to(device, non_blocking=True)
        error_mask = batch["error_mask"].to(device, non_blocking=True)

        rgb_in = _normalize_imagenet(rgb)
        pred = model(rgb_in)
        target_size = face_mask.shape[-2:]
        if pred.shape[-2:] != target_size:
            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
        pred_parts = _split_dense_prediction(
            pred,
            predict_basecolor=predict_basecolor,
            predict_geo=predict_geo,
            predict_normal=predict_normal,
        )
        rgb_loss = _masked_feature_loss(
            pred_parts["rgb"].float() if pred_parts["rgb"] is not None else None,
            gt_rgb.float() if gt_rgb is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        geo_loss = _masked_feature_loss(
            pred_parts["geo"].float() if pred_parts["geo"] is not None else None,
            gt_geo.float() if gt_geo is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        detail_normal_loss = _masked_feature_loss(
            pred_parts["detail_normal"].float() if pred_parts["detail_normal"] is not None else None,
            gt_detail_normal.float() if gt_detail_normal is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        geometry_normal_loss = _masked_normal_cosine_loss(
            pred_parts["geometry_normal"].float() if pred_parts["geometry_normal"] is not None else None,
            gt_geometry_normal.float() if gt_geometry_normal is not None else None,
            face_mask=face_mask.float(),
            error_mask=error_mask.float(),
        )
        normal_loss = detail_normal_loss + geometry_normal_loss
        mask_loss, mask_bce, mask_dice = _combined_mask_loss(
            pred_parts["mask_logits"].float() if pred_parts["mask_logits"] is not None else None,
            face_mask.float(),
            bce_lambda=train_cfg.mask_bce_lambda,
            dice_lambda=train_cfg.mask_dice_lambda,
            error_mask=error_mask.float(),
        )
        loss = rgb_loss + geo_loss + normal_loss + mask_loss
        if torch.isfinite(loss):
            total_loss += float(loss.item())
            total_rgb_loss += float(rgb_loss.item())
            total_geo_loss += float(geo_loss.item())
            total_detail_normal_loss += float(detail_normal_loss.item())
            total_geometry_normal_loss += float(geometry_normal_loss.item())
            total_normal_loss += float(normal_loss.item())
            total_mask_loss += float(mask_loss.item())
            total_mask_bce += float(mask_bce.item())
            total_mask_dice += float(mask_dice.item())
            n += 1

    # Aggregate loss across all ranks for distributed training
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor(
            [
                total_loss,
                total_rgb_loss,
                total_geo_loss,
                total_detail_normal_loss,
                total_geometry_normal_loss,
                total_normal_loss,
                total_mask_loss,
                total_mask_bce,
                total_mask_dice,
                n,
            ],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = loss_tensor[0].item()
        total_rgb_loss = loss_tensor[1].item()
        total_geo_loss = loss_tensor[2].item()
        total_detail_normal_loss = loss_tensor[3].item()
        total_geometry_normal_loss = loss_tensor[4].item()
        total_normal_loss = loss_tensor[5].item()
        total_mask_loss = loss_tensor[6].item()
        total_mask_bce = loss_tensor[7].item()
        total_mask_dice = loss_tensor[8].item()
        n = loss_tensor[9].item()

    denom = max(n, 1)
    return {
        "avg_loss": total_loss / denom,
        "avg_rgb_loss": total_rgb_loss / denom,
        "avg_geo_loss": total_geo_loss / denom,
        "avg_detail_normal_loss": total_detail_normal_loss / denom,
        "avg_geometry_normal_loss": total_geometry_normal_loss / denom,
        "avg_normal_loss": total_normal_loss / denom,
        "avg_mask_loss": total_mask_loss / denom,
        "avg_mask_bce": total_mask_bce / denom,
        "avg_mask_dice": total_mask_dice / denom,
    }


def run_training(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, device: torch.device):
    train_loader, val_loader = create_dataloaders(data_cfg)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = DenseImageTransformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        predict_basecolor=model_cfg.predict_basecolor,
        predict_geo=model_cfg.predict_geo,
        predict_normal=model_cfg.predict_normal,
        output_size=data_cfg.image_size,
        transformer_map_size=model_cfg.transformer_map_size,
        backbone_weights=model_cfg.backbone_weights,
    ).to(device)

    resumed_ckpt = None
    if train_cfg.load_model and os.path.exists(train_cfg.load_model):
        ckpt = torch.load(train_cfg.load_model, map_location="cpu")
        if "model_state_dict" in ckpt:
            skipped_keys, load_notes = _load_matching_state_dict(model, ckpt["model_state_dict"])
            resumed_ckpt = ckpt
        else:
            skipped_keys, load_notes = _load_matching_state_dict(model, ckpt)
        print(f"Loaded checkpoint: {train_cfg.load_model}")
        if skipped_keys:
            print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
        if load_notes:
            print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")

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

    # Restore training state from checkpoint if available
    start_epoch = 0
    best_val = float("inf")
    if resumed_ckpt is not None:
        if "optimizer_state_dict" in resumed_ckpt:
            try:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
                print("[Resume] Restored optimizer state")
            except Exception as e:
                print(f"[Resume] Could not restore optimizer state: {e}")
        if "scheduler_state_dict" in resumed_ckpt:
            try:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
                print("[Resume] Restored scheduler state")
            except Exception as e:
                print(f"[Resume] Could not restore scheduler state: {e}")
        if "scaler_state_dict" in resumed_ckpt:
            try:
                scaler.load_state_dict(resumed_ckpt["scaler_state_dict"])
                print("[Resume] Restored scaler state")
            except Exception as e:
                print(f"[Resume] Could not restore scaler state: {e}")
        if "epoch" in resumed_ckpt:
            start_epoch = int(resumed_ckpt["epoch"]) + 1
            print(f"[Resume] Resuming from epoch {start_epoch + 1}")
        if "best_val" in resumed_ckpt:
            best_val = float(resumed_ckpt["best_val"])
        elif "val_loss" in resumed_ckpt:
            best_val = float(resumed_ckpt["val_loss"])

    run_name = f"dense_img_{model_cfg.backbone_weights}_{int(time.time())}"
    writer = SummaryWriter(os.path.join("runs", run_name))

    for epoch in range(start_epoch, train_cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=device)

        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            train_cfg=train_cfg,
        )
        val_out = validate_one_epoch(model=model, loader=val_loader, device=device, train_cfg=train_cfg)
        scheduler.step()

        print(
            f"Train Loss: {train_out['avg_loss']:.6f} | Val Loss: {val_out['avg_loss']:.6f} "
            f"| Train RGB: {train_out['avg_rgb_loss']:.6f} | Train Geo: {train_out['avg_geo_loss']:.6f} "
            f"| Train Normal: {train_out['avg_normal_loss']:.6f} "
            f"(detail {train_out['avg_detail_normal_loss']:.6f} + geom {train_out['avg_geometry_normal_loss']:.6f}) "
            f"| Train Mask: {train_out['avg_mask_loss']:.6f} "
            f"| Val RGB: {val_out['avg_rgb_loss']:.6f} | Val Geo: {val_out['avg_geo_loss']:.6f} "
            f"| Val Normal: {val_out['avg_normal_loss']:.6f} "
            f"(detail {val_out['avg_detail_normal_loss']:.6f} + geom {val_out['avg_geometry_normal_loss']:.6f}) "
            f"| Val Mask: {val_out['avg_mask_loss']:.6f}"
        )
        writer.add_scalar("Loss/Train", train_out["avg_loss"], epoch)
        writer.add_scalar("Loss/Val", val_out["avg_loss"], epoch)
        writer.add_scalar("Loss/RGB_Train", train_out["avg_rgb_loss"], epoch)
        writer.add_scalar("Loss/RGB_Val", val_out["avg_rgb_loss"], epoch)
        writer.add_scalar("Loss/Geo_Train", train_out["avg_geo_loss"], epoch)
        writer.add_scalar("Loss/Geo_Val", val_out["avg_geo_loss"], epoch)
        writer.add_scalar("Loss/DetailNormal_Train", train_out["avg_detail_normal_loss"], epoch)
        writer.add_scalar("Loss/DetailNormal_Val", val_out["avg_detail_normal_loss"], epoch)
        writer.add_scalar("Loss/GeometryNormal_Train", train_out["avg_geometry_normal_loss"], epoch)
        writer.add_scalar("Loss/GeometryNormal_Val", val_out["avg_geometry_normal_loss"], epoch)
        writer.add_scalar("Loss/Normal_Train", train_out["avg_normal_loss"], epoch)
        writer.add_scalar("Loss/Normal_Val", val_out["avg_normal_loss"], epoch)
        writer.add_scalar("Loss/Mask_Train", train_out["avg_mask_loss"], epoch)
        writer.add_scalar("Loss/Mask_Val", val_out["avg_mask_loss"], epoch)
        writer.add_scalar("Loss/MaskBCE_Train", train_out["avg_mask_bce"], epoch)
        writer.add_scalar("Loss/MaskBCE_Val", val_out["avg_mask_bce"], epoch)
        writer.add_scalar("Loss/MaskDice_Train", train_out["avg_mask_dice"], epoch)
        writer.add_scalar("Loss/MaskDice_Val", val_out["avg_mask_dice"], epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device=device) / (1024**3)
            writer.add_scalar("GPU/PeakGB", peak_gb, epoch)
            print(f"Peak GPU memory: {peak_gb:.2f} GB")

        visualization_batch = train_out["visualization_batch"]
        if visualization_batch is not None and ((epoch + 1) % train_cfg.save_preview_every == 0):
            model.eval()
            with torch.no_grad():
                rgb = visualization_batch["rgb"].to(device, non_blocking=True)
                rgb_in = _normalize_imagenet(rgb)
                preview_pred = model(rgb_in)
                if preview_pred.shape[-2:] != visualization_batch["basecolor"].shape[-2:]:
                    preview_pred = F.interpolate(
                        preview_pred,
                        size=visualization_batch["basecolor"].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                _save_preview(
                    epoch,
                    visualization_batch,
                    preview_pred,
                    predict_basecolor=model_cfg.predict_basecolor,
                    predict_geo=model_cfg.predict_geo,
                    predict_normal=model_cfg.predict_normal,
                )
            model.train()

        if val_out["avg_loss"] < best_val:
            best_val = val_out["avg_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": val_out["avg_loss"],
                    "best_val": best_val,
                    "output_channels": model_cfg.output_channels,
                    "predict_basecolor": model_cfg.predict_basecolor,
                    "predict_geo": model_cfg.predict_geo,
                    "predict_normal": model_cfg.predict_normal,
                },
                train_cfg.save_path,
            )
            print(f"Saved best model: {train_cfg.save_path}")

    writer.close()


def train_worker(rank: int, world_size: int, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, backend: str):
    """Worker function for distributed training - one process per GPU."""
    setup_distributed(rank, world_size, train_cfg.master_port, backend)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{rank}")

    # Create distributed dataloaders
    train_loader, val_loader, train_sampler = create_distributed_dataloaders(data_cfg, rank, world_size)

    if rank == 0:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    model = DenseImageTransformer(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        predict_basecolor=model_cfg.predict_basecolor,
        predict_geo=model_cfg.predict_geo,
        predict_normal=model_cfg.predict_normal,
        output_size=data_cfg.image_size,
        transformer_map_size=model_cfg.transformer_map_size,
        backbone_weights=model_cfg.backbone_weights,
    ).to(device)

    # Load checkpoint
    resumed_ckpt = None
    if train_cfg.load_model and os.path.exists(train_cfg.load_model):
        ckpt = torch.load(train_cfg.load_model, map_location="cpu")
        if "model_state_dict" in ckpt:
            # Full checkpoint with training state
            skipped_keys, load_notes = _load_matching_state_dict(model, ckpt["model_state_dict"])
            resumed_ckpt = ckpt
        else:
            # Weights-only checkpoint
            skipped_keys, load_notes = _load_matching_state_dict(model, ckpt)
        if rank == 0:
            print(f"Loaded checkpoint: {train_cfg.load_model}")
            if skipped_keys:
                print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
            if load_notes:
                print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")

    # Wrap with DDP for multi-GPU
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[rank])

    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_cfg.epochs), eta_min=1e-6
    )

    amp_dtype = torch.float16 if train_cfg.amp_dtype == "fp16" else torch.bfloat16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # Restore training state from checkpoint if available
    start_epoch = 0
    best_val = float("inf")
    if resumed_ckpt is not None:
        if "optimizer_state_dict" in resumed_ckpt:
            try:
                optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
                if rank == 0:
                    print("[Resume] Restored optimizer state")
            except Exception as e:
                if rank == 0:
                    print(f"[Resume] Could not restore optimizer state: {e}")
        if "scheduler_state_dict" in resumed_ckpt:
            try:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
                if rank == 0:
                    print("[Resume] Restored scheduler state")
            except Exception as e:
                if rank == 0:
                    print(f"[Resume] Could not restore scheduler state: {e}")
        if "scaler_state_dict" in resumed_ckpt:
            try:
                scaler.load_state_dict(resumed_ckpt["scaler_state_dict"])
                if rank == 0:
                    print("[Resume] Restored scaler state")
            except Exception as e:
                if rank == 0:
                    print(f"[Resume] Could not restore scaler state: {e}")
        if "epoch" in resumed_ckpt:
            start_epoch = int(resumed_ckpt["epoch"]) + 1
            if rank == 0:
                print(f"[Resume] Resuming from epoch {start_epoch + 1}")
        if "best_val" in resumed_ckpt:
            best_val = float(resumed_ckpt["best_val"])
        elif "val_loss" in resumed_ckpt:
            best_val = float(resumed_ckpt["val_loss"])

    # Tensorboard writer only on rank 0
    writer = None
    if rank == 0:
        run_name = f"dense_img_{model_cfg.backbone_weights}_{int(time.time())}"
        writer = SummaryWriter(os.path.join("runs", run_name))

    for epoch in range(start_epoch, train_cfg.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")

        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            amp_dtype=amp_dtype,
            device=device,
            train_cfg=train_cfg,
            rank=rank,
            world_size=world_size,
            train_sampler=train_sampler,
            epoch=epoch,
        )

        val_out = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            train_cfg=train_cfg,
            rank=rank,
            world_size=world_size,
        )
        scheduler.step()

        # Only rank 0 logs and saves
        if rank == 0:
            print(
                f"Train Loss: {train_out['avg_loss']:.6f} | Val Loss: {val_out['avg_loss']:.6f} "
                f"| Train RGB: {train_out['avg_rgb_loss']:.6f} | Train Geo: {train_out['avg_geo_loss']:.6f} "
                f"| Train Normal: {train_out['avg_normal_loss']:.6f} "
                f"(detail {train_out['avg_detail_normal_loss']:.6f} + geom {train_out['avg_geometry_normal_loss']:.6f}) "
                f"| Train Mask: {train_out['avg_mask_loss']:.6f} "
                f"| Val RGB: {val_out['avg_rgb_loss']:.6f} | Val Geo: {val_out['avg_geo_loss']:.6f} "
                f"| Val Normal: {val_out['avg_normal_loss']:.6f} "
                f"(detail {val_out['avg_detail_normal_loss']:.6f} + geom {val_out['avg_geometry_normal_loss']:.6f}) "
                f"| Val Mask: {val_out['avg_mask_loss']:.6f}"
            )
            writer.add_scalar("Loss/Train", train_out["avg_loss"], epoch)
            writer.add_scalar("Loss/Val", val_out["avg_loss"], epoch)
            writer.add_scalar("Loss/RGB_Train", train_out["avg_rgb_loss"], epoch)
            writer.add_scalar("Loss/RGB_Val", val_out["avg_rgb_loss"], epoch)
            writer.add_scalar("Loss/Geo_Train", train_out["avg_geo_loss"], epoch)
            writer.add_scalar("Loss/Geo_Val", val_out["avg_geo_loss"], epoch)
            writer.add_scalar("Loss/DetailNormal_Train", train_out["avg_detail_normal_loss"], epoch)
            writer.add_scalar("Loss/DetailNormal_Val", val_out["avg_detail_normal_loss"], epoch)
            writer.add_scalar("Loss/GeometryNormal_Train", train_out["avg_geometry_normal_loss"], epoch)
            writer.add_scalar("Loss/GeometryNormal_Val", val_out["avg_geometry_normal_loss"], epoch)
            writer.add_scalar("Loss/Normal_Train", train_out["avg_normal_loss"], epoch)
            writer.add_scalar("Loss/Normal_Val", val_out["avg_normal_loss"], epoch)
            writer.add_scalar("Loss/Mask_Train", train_out["avg_mask_loss"], epoch)
            writer.add_scalar("Loss/Mask_Val", val_out["avg_mask_loss"], epoch)
            writer.add_scalar("Loss/MaskBCE_Train", train_out["avg_mask_bce"], epoch)
            writer.add_scalar("Loss/MaskBCE_Val", val_out["avg_mask_bce"], epoch)
            writer.add_scalar("Loss/MaskDice_Train", train_out["avg_mask_dice"], epoch)
            writer.add_scalar("Loss/MaskDice_Val", val_out["avg_mask_dice"], epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # Generate preview visualization if we have a batch
            visualization_batch = train_out["visualization_batch"]
            if visualization_batch is not None and ((epoch + 1) % train_cfg.save_preview_every == 0):
                model.eval()
                with torch.no_grad():
                    # Move batch back to device for inference
                    rgb = visualization_batch["rgb"].to(device, non_blocking=True)
                    rgb_in = _normalize_imagenet(rgb)
                    preview_pred = model(rgb_in)
                    if preview_pred.shape[-2:] != visualization_batch["basecolor"].shape[-2:]:
                        preview_pred = F.interpolate(
                            preview_pred,
                            size=visualization_batch["basecolor"].shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    _save_preview(
                        epoch,
                        visualization_batch,
                        preview_pred,
                        predict_basecolor=model_cfg.predict_basecolor,
                        predict_geo=model_cfg.predict_geo,
                        predict_normal=model_cfg.predict_normal,
                    )
                model.train()

            if val_out["avg_loss"] < best_val:
                best_val = val_out["avg_loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": _unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "val_loss": val_out["avg_loss"],
                        "best_val": best_val,
                        "output_channels": model_cfg.output_channels,
                        "predict_basecolor": model_cfg.predict_basecolor,
                        "predict_geo": model_cfg.predict_geo,
                        "predict_normal": model_cfg.predict_normal,
                    },
                    train_cfg.save_path,
                )
                print(f"Saved best model: {train_cfg.save_path}")

    if writer:
        writer.close()
    cleanup_distributed()


def launch_linux(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args, platform_name="linux")
    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")
    print(f"Spawning {world_size} processes (linux/nccl)")
    mp.spawn(train_worker, args=(world_size, data_cfg, model_cfg, train_cfg, "nccl"), nprocs=world_size, join=True)


def launch_windows(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args, platform_name="windows")
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("No CUDA devices found")
    print("Launching single-process training (windows/gloo)")
    train_worker(0, 1, data_cfg, model_cfg, train_cfg, "gloo")
