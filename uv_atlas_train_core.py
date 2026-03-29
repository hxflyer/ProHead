from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from uv_atlas_dataset import UVAtlasDataset, uv_atlas_collate_fn
from uv_atlas_unet import UVAtlasUNet


@dataclass
class DataConfig:
    cache_root: str = "artifacts/uv_atlas_cache"
    batch_size: int = 1
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    max_train_samples: int = 0
    validate_cache: bool = False


@dataclass
class ModelConfig:
    input_channels: int = 16
    base_channels: int = 32


@dataclass
class TrainConfig:
    lr: float = 2e-4
    epochs: int = 50
    amp_dtype: str = "fp16"
    load_model: str = ""
    save_path: str = "artifacts/checkpoints/best_uv_atlas_unet.pth"
    save_preview_every: int = 1
    master_port: str = "12359"


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
    parser.add_argument("--cache_root", type=str, default="artifacts/uv_atlas_cache")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--validate_cache",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--persistent_workers",
        type=_parse_bool_arg,
        nargs="?",
        const=True,
        default=True,
    )

    parser.add_argument("--input_channels", type=int, default=16)
    parser.add_argument("--base_channels", type=int, default=32)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_path", type=str, default="artifacts/checkpoints/best_uv_atlas_unet.pth")
    parser.add_argument("--save_preview_every", type=int, default=1)
    parser.add_argument("--master_port", type=str, default="12359")
    return parser


def build_configs_from_args(args) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_cfg = DataConfig(
        cache_root=str(args.cache_root),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
        max_train_samples=int(args.max_train_samples),
        validate_cache=bool(args.validate_cache),
    )
    model_cfg = ModelConfig(
        input_channels=int(args.input_channels),
        base_channels=int(args.base_channels),
    )
    train_cfg = TrainConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        amp_dtype=str(args.amp_dtype),
        load_model=str(args.load_model),
        save_path=str(args.save_path),
        save_preview_every=max(1, int(args.save_preview_every)),
        master_port=str(args.master_port),
    )
    return data_cfg, model_cfg, train_cfg


def setup_distributed(rank: int, world_size: int, master_port: str, backend: str) -> None:
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
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _load_matching_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
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
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _shutdown_dataloader(loader) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is None:
        return
    try:
        shutdown()
    except Exception:
        pass


def _loader_kwargs(data_cfg: DataConfig) -> dict:
    kwargs = {
        "num_workers": int(max(0, data_cfg.num_workers)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": uv_atlas_collate_fn,
        "worker_init_fn": worker_init_fn,
    }
    if kwargs["num_workers"] > 0:
        kwargs["prefetch_factor"] = int(max(1, data_cfg.prefetch_factor))
        kwargs["persistent_workers"] = bool(data_cfg.persistent_workers)
    return kwargs


def create_distributed_dataloaders(data_cfg: DataConfig, rank: int, world_size: int):
    train_dataset = UVAtlasDataset(
        cache_root=data_cfg.cache_root,
        split="train",
        validate_samples=bool(data_cfg.validate_cache),
    )
    val_dataset = UVAtlasDataset(
        cache_root=data_cfg.cache_root,
        split="val",
        validate_samples=bool(data_cfg.validate_cache),
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No train UV atlas cache samples found under {data_cfg.cache_root!r}. Run scripts/data/precompute_uv_atlas_cache.py first.")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No val UV atlas cache samples found under {data_cfg.cache_root!r}. Run scripts/data/precompute_uv_atlas_cache.py first.")
    if int(data_cfg.max_train_samples) > 0:
        max_n = min(int(data_cfg.max_train_samples), len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(max_n)))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    loader_kwargs = _loader_kwargs(data_cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.batch_size),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.batch_size),
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def _resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    return torch.float16 if str(amp_dtype).lower() == "fp16" else torch.bfloat16


def _to_vis_map(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr, 0.0, 1.0)


def _save_preview(
    epoch: int,
    batch,
    pred_out: dict[str, torch.Tensor],
    output_dir: str = "artifacts/training_samples_uv_atlas",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    src_color = batch["src_color_uv"][:4].detach().cpu().numpy()
    pred_basecolor_in = batch["pred_basecolor_uv"][:4].detach().cpu().numpy()
    pred_geo_in = batch["pred_geo_uv"][:4].detach().cpu().numpy()
    pred_geometry_normal_in = batch["pred_geometry_normal_uv"][:4].detach().cpu().numpy()
    pred_detail_normal_in = batch["pred_detail_normal_uv"][:4].detach().cpu().numpy()
    uv_valid_mask = batch["uv_valid_mask"][:4].detach().cpu().numpy()
    gt_basecolor = batch["basecolor_atlas"][:4].detach().cpu().numpy()
    gt_detail_normal = batch["detail_normal_atlas"][:4].detach().cpu().numpy()

    pred_basecolor = pred_out["basecolor"][:4].detach().cpu().numpy()
    pred_detail_normal = pred_out["detail_normal"][:4].detach().cpu().numpy()

    rows = []
    for i in range(src_color.shape[0]):
        src_i = np.transpose(src_color[i], (1, 2, 0))
        base_in_i = np.transpose(pred_basecolor_in[i], (1, 2, 0))
        geo_in_i = np.transpose(pred_geo_in[i], (1, 2, 0))
        geom_nrm_i = np.transpose(pred_geometry_normal_in[i], (1, 2, 0))
        detail_in_i = np.transpose(pred_detail_normal_in[i], (1, 2, 0))
        mask_i = np.transpose(uv_valid_mask[i], (1, 2, 0))
        gt_base_i = np.transpose(gt_basecolor[i], (1, 2, 0))
        pred_base_i = np.transpose(pred_basecolor[i], (1, 2, 0))
        gt_normal_i = np.transpose(gt_detail_normal[i], (1, 2, 0))
        pred_normal_i = np.transpose(pred_detail_normal[i], (1, 2, 0))
        base_diff_i = np.abs(pred_base_i - gt_base_i)
        normal_diff_i = np.abs(pred_normal_i - gt_normal_i)

        row = np.concatenate(
            [
                _to_vis_map(src_i),
                _to_vis_map(base_in_i),
                _to_vis_map(geo_in_i),
                _to_vis_map(geom_nrm_i),
                _to_vis_map(detail_in_i),
                _to_vis_map(mask_i),
                _to_vis_map(gt_base_i),
                _to_vis_map(pred_base_i),
                _to_vis_map(base_diff_i),
                _to_vis_map(gt_normal_i),
                _to_vis_map(pred_normal_i),
                _to_vis_map(normal_diff_i),
            ],
            axis=1,
        )
        rows.append((row * 255.0).astype(np.uint8))

    if rows:
        canvas = np.concatenate(rows, axis=0)
        out_path = os.path.join(output_dir, f"epoch_{epoch + 1:03d}_uv_atlas_preview.png")
        cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"Saved UV atlas preview: {out_path}")


def _compute_losses(pred_out: dict[str, torch.Tensor], batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_basecolor = batch["basecolor_atlas"].to(device, non_blocking=True)
    gt_detail_normal = batch["detail_normal_atlas"].to(device, non_blocking=True)
    basecolor_loss = F.l1_loss(pred_out["basecolor"].float(), gt_basecolor.float())
    detail_normal_loss = F.l1_loss(pred_out["detail_normal"].float(), gt_detail_normal.float())
    total_loss = basecolor_loss + detail_normal_loss
    return total_loss, basecolor_loss, detail_normal_loss


def _reduce_metric_sums(total_loss: float, total_basecolor_loss: float, total_detail_normal_loss: float, sample_count: int, device: torch.device):
    tensor = torch.tensor(
        [total_loss, total_basecolor_loss, total_detail_normal_loss, float(sample_count)],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denom = max(float(tensor[3].item()), 1.0)
    return {
        "avg_loss": float(tensor[0].item() / denom),
        "avg_basecolor_loss": float(tensor[1].item() / denom),
        "avg_detail_normal_loss": float(tensor[2].item() / denom),
        "num_samples": int(tensor[3].item()),
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype,
    device: torch.device,
    rank: int = 0,
    train_sampler: DistributedSampler | None = None,
    epoch: int = 0,
):
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    total_loss = 0.0
    total_basecolor_loss = 0.0
    total_detail_normal_loss = 0.0
    sample_count = 0
    pbar = tqdm(loader, desc="Training") if rank == 0 else loader

    for batch in pbar:
        uv_input = batch["uv_input"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred_out = model(uv_input)
            loss, basecolor_loss, detail_normal_loss = _compute_losses(pred_out, batch, device)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(uv_input.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_basecolor_loss += float(basecolor_loss.item()) * batch_size
        total_detail_normal_loss += float(detail_normal_loss.item()) * batch_size
        sample_count += batch_size

        if rank == 0 and hasattr(pbar, "set_postfix"):
            denom = max(sample_count, 1)
            pbar.set_postfix({
                "loss": f"{total_loss / denom:.4f}",
                "base": f"{total_basecolor_loss / denom:.4f}",
                "nrm": f"{total_detail_normal_loss / denom:.4f}",
            })

    return _reduce_metric_sums(total_loss, total_basecolor_loss, total_detail_normal_loss, sample_count, device)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    amp_dtype: torch.dtype,
    device: torch.device,
    rank: int = 0,
):
    model.eval()
    total_loss = 0.0
    total_basecolor_loss = 0.0
    total_detail_normal_loss = 0.0
    sample_count = 0
    visualization_batch = None

    for batch in loader:
        if rank == 0 and visualization_batch is None:
            visualization_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        uv_input = batch["uv_input"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            pred_out = model(uv_input)
            loss, basecolor_loss, detail_normal_loss = _compute_losses(pred_out, batch, device)

        batch_size = int(uv_input.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_basecolor_loss += float(basecolor_loss.item()) * batch_size
        total_detail_normal_loss += float(detail_normal_loss.item()) * batch_size
        sample_count += batch_size

    out = _reduce_metric_sums(total_loss, total_basecolor_loss, total_detail_normal_loss, sample_count, device)
    out["visualization_batch"] = visualization_batch if rank == 0 else None
    return out


def log_and_checkpoint(
    epoch: int,
    rank: int,
    built,
    train_out,
    val_out,
    writer,
    train_cfg: TrainConfig,
):
    if rank != 0:
        return

    optimizer = built["optimizer"]
    print(
        f"Train Loss: {train_out['avg_loss']:.6f} | TrainBase: {train_out['avg_basecolor_loss']:.6f} | "
        f"TrainNormal: {train_out['avg_detail_normal_loss']:.6f} | "
        f"Val Loss: {val_out['avg_loss']:.6f} | ValBase: {val_out['avg_basecolor_loss']:.6f} | "
        f"ValNormal: {val_out['avg_detail_normal_loss']:.6f}"
    )

    if writer is not None:
        writer.add_scalar("Loss/Train", train_out["avg_loss"], epoch)
        writer.add_scalar("Loss/Val", val_out["avg_loss"], epoch)
        writer.add_scalar("Loss/Basecolor_Train", train_out["avg_basecolor_loss"], epoch)
        writer.add_scalar("Loss/Basecolor_Val", val_out["avg_basecolor_loss"], epoch)
        writer.add_scalar("Loss/DetailNormal_Train", train_out["avg_detail_normal_loss"], epoch)
        writer.add_scalar("Loss/DetailNormal_Val", val_out["avg_detail_normal_loss"], epoch)
        writer.add_scalar("LR/main", optimizer.param_groups[0]["lr"], epoch)

    if val_out["avg_loss"] < built["best_loss"]:
        built["best_loss"] = val_out["avg_loss"]
        print(f"New best UV atlas model (Val Loss: {val_out['avg_loss']:.6f})")

    save_dir = os.path.dirname(train_cfg.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    save_dict = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(built["model"]).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": built["scheduler"].state_dict(),
        "scaler_state_dict": built["scaler"].state_dict(),
        "val_loss": val_out["avg_loss"],
        "best_loss": built["best_loss"],
    }
    torch.save(save_dict, train_cfg.save_path)
    print(f"Saved checkpoint to {train_cfg.save_path} (epoch {epoch + 1}, Val Loss: {val_out['avg_loss']:.6f})")


def train_worker(rank: int, world_size: int, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, backend: str):
    setup_distributed(rank, world_size, train_cfg.master_port, backend)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = _resolve_amp_dtype(train_cfg.amp_dtype)

    train_loader = val_loader = None
    writer = None
    try:
        if rank == 0:
            print(
                f"[UVAtlasTrain] rank=0 distributed ready, building datasets "
                f"(validate_cache={bool(data_cfg.validate_cache)})"
            )
        train_loader, val_loader, train_sampler, _ = create_distributed_dataloaders(data_cfg, rank, world_size)
        model = UVAtlasUNet(
            input_channels=int(model_cfg.input_channels),
            base_channels=int(model_cfg.base_channels),
        ).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[rank], output_device=rank)

        optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg.lr), betas=(0.9, 0.95), weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(train_cfg.epochs), 1))
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

        built = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "best_loss": float("inf"),
        }
        start_epoch = 0

        if train_cfg.load_model and os.path.exists(train_cfg.load_model):
            checkpoint = torch.load(train_cfg.load_model, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            skipped_keys, load_notes = _load_matching_state_dict(_unwrap_model(model), state_dict)
            if rank == 0:
                print(f"Loaded checkpoint: {train_cfg.load_model}")
                if skipped_keys:
                    print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
                if load_notes:
                    print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")
            if isinstance(checkpoint, dict):
                start_epoch = int(checkpoint.get("epoch", -1)) + 1
                built["best_loss"] = float(checkpoint.get("best_loss", checkpoint.get("val_loss", float("inf"))))
                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if rank == 0:
                        print("[Resume] Restored optimizer state")
                if "scheduler_state_dict" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if rank == 0:
                        print("[Resume] Restored scheduler state")
                if "scaler_state_dict" in checkpoint:
                    try:
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])
                        if rank == 0:
                            print("[Resume] Restored scaler state")
                    except Exception:
                        if rank == 0:
                            print("[Resume] Failed to restore scaler state; continuing with fresh scaler")

        if rank == 0:
            run_name = f"uv_atlas_unet_{time.strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(os.path.join("artifacts", "runs", run_name))

        for epoch in range(start_epoch, int(train_cfg.epochs)):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")

            train_out = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                amp_dtype=amp_dtype,
                device=device,
                rank=rank,
                train_sampler=train_sampler,
                epoch=epoch,
            )
            val_out = evaluate(
                model=model,
                loader=val_loader,
                amp_dtype=amp_dtype,
                device=device,
                rank=rank,
            )
            scheduler.step()
            log_and_checkpoint(epoch, rank, built, train_out, val_out, writer, train_cfg)

            if rank == 0 and val_out.get("visualization_batch") is not None and ((epoch + 1) % train_cfg.save_preview_every == 0):
                preview_batch = val_out["visualization_batch"]
                preview_input = preview_batch["uv_input"].to(device, non_blocking=True)
                preview_model = _unwrap_model(model)
                preview_model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                        preview_out = preview_model(preview_input)
                _save_preview(epoch, preview_batch, preview_out)

    finally:
        _shutdown_dataloader(train_loader)
        _shutdown_dataloader(val_loader)
        if writer is not None:
            writer.close()
        cleanup_distributed()


def launch_linux(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)
    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found")
    print(f"Spawning {world_size} processes (linux/nccl)")
    mp.spawn(train_worker, args=(world_size, data_cfg, model_cfg, train_cfg, "nccl"), nprocs=world_size, join=True)


def launch_windows(args):
    data_cfg, model_cfg, train_cfg = build_configs_from_args(args)
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("No CUDA devices found")
    print("Launching single-process training (windows/gloo)")
    train_worker(0, 1, data_cfg, model_cfg, train_cfg, "gloo")


def main() -> None:
    parser = create_arg_parser("Train UV Atlas UNet")
    args = parser.parse_args()
    if sys.platform.startswith("win"):
        launch_windows(args)
    else:
        launch_linux(args)


if __name__ == "__main__":
    main()
