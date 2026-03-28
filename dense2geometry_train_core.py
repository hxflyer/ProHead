import argparse
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import cv2
from pathlib import Path

from dense2geometry import Dense2Geometry, DenseStageConfig
from dense2geometry_dataset import Dense2GeometryDataset, dense2geometry_collate_fn
from train_loss_helper import MetricAccumulator, compute_weighted_l1
from train_visualize_helper import load_mesh_topology, save_dense2geometry_visualizations, create_combined_overlay


@dataclass
class DataConfig:
    data_roots: list[str]
    texture_root: str = ""
    batch_size: int = 2
    num_workers: int = 4
    prefetch_factor: int = 1
    persistent_workers: bool = True
    image_size: int = 1024
    train_ratio: float = 0.95
    max_train_samples: int = 0


@dataclass
class ModelConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    image_memory_size: int = 16
    search_size: int = 512
    landmark_search_size: int = 256
    local_search_radius: int = 12
    local_min_candidates: int = 9
    search_chunk_size: int = 1024
    search_distance_threshold: float = 0.05
    search_distance_floor: float = 0.02
    search_mad_scale: float = 3.0
    search_mask_threshold: float = 0.30
    search_min_geo_magnitude: float = 0.02
    freeze_dense_stage: bool = True
    dense_d_model: int = 256
    dense_nhead: int = 8
    dense_num_layers: int = 4
    dense_transformer_map_size: int = 32
    dense_backbone_weights: str = "imagenet"
    dense_decoder_type: str = "multitask"


@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 50
    amp_dtype: str = "fp16"
    load_model: str = ""
    load_dense_model: str = "artifacts/checkpoints/best_dense_image_transformer_ch13.pth"
    save_path: str = "artifacts/checkpoints/best_dense2geometry.pth"
    master_port: str = "12358"


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
    parser.add_argument("--data_roots", nargs="*", default=None)
    parser.add_argument("--texture_root", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", type=_parse_bool_arg, nargs="?", const=True, default=True)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--train_ratio", type=float, default=0.95)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--load_dense_model", type=str, default="artifacts/checkpoints/best_dense_image_transformer_ch13.pth")
    parser.add_argument("--save_path", type=str, default="artifacts/checkpoints/best_dense2geometry.pth")
    parser.add_argument("--master_port", type=str, default="12358")

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--image_memory_size", type=int, default=16)
    parser.add_argument("--search_size", type=int, default=512)
    parser.add_argument("--landmark_search_size", type=int, default=256)
    parser.add_argument("--local_search_radius", type=int, default=12)
    parser.add_argument("--local_min_candidates", type=int, default=9)
    parser.add_argument("--search_chunk_size", type=int, default=1024)
    parser.add_argument("--search_distance_threshold", type=float, default=0.05)
    parser.add_argument("--search_distance_floor", type=float, default=0.02)
    parser.add_argument("--search_mad_scale", type=float, default=3.0)
    parser.add_argument("--search_mask_threshold", type=float, default=0.30)
    parser.add_argument("--search_min_geo_magnitude", type=float, default=0.02)
    parser.add_argument("--freeze_dense_stage", type=_parse_bool_arg, nargs="?", const=True, default=True)

    parser.add_argument("--dense_d_model", type=int, default=256)
    parser.add_argument("--dense_nhead", type=int, default=8)
    parser.add_argument("--dense_num_layers", type=int, default=4)
    parser.add_argument("--dense_transformer_map_size", type=int, default=32)
    parser.add_argument("--dense_backbone_weights", type=str, default="imagenet", choices=["imagenet", "dinov3"])
    parser.add_argument("--dense_decoder_type", type=str, default="multitask", choices=["multitask", "shared"])
    return parser


def build_configs_from_args(args) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_cfg = DataConfig(
        data_roots=list(args.data_roots) if args.data_roots else [],
        texture_root=str(args.texture_root or ""),
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
        image_memory_size=int(args.image_memory_size),
        search_size=int(args.search_size),
        landmark_search_size=int(args.landmark_search_size),
        local_search_radius=int(args.local_search_radius),
        local_min_candidates=int(args.local_min_candidates),
        search_chunk_size=int(args.search_chunk_size),
        search_distance_threshold=float(args.search_distance_threshold),
        search_distance_floor=float(args.search_distance_floor),
        search_mad_scale=float(args.search_mad_scale),
        search_mask_threshold=float(args.search_mask_threshold),
        search_min_geo_magnitude=float(args.search_min_geo_magnitude),
        freeze_dense_stage=bool(args.freeze_dense_stage),
        dense_d_model=int(args.dense_d_model),
        dense_nhead=int(args.dense_nhead),
        dense_num_layers=int(args.dense_num_layers),
        dense_transformer_map_size=int(args.dense_transformer_map_size),
        dense_backbone_weights=str(args.dense_backbone_weights),
        dense_decoder_type=str(args.dense_decoder_type),
    )
    train_cfg = TrainConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        amp_dtype=str(args.amp_dtype),
        load_model=str(args.load_model),
        load_dense_model=str(args.load_dense_model),
        save_path=str(args.save_path),
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
    import cv2
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


def create_distributed_dataloaders(data_cfg: DataConfig, rank: int, world_size: int):
    train_dataset = Dense2GeometryDataset(
        data_roots=data_cfg.data_roots,
        split="train",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=True,
        texture_root=data_cfg.texture_root,
    )
    val_dataset = Dense2GeometryDataset(
        data_roots=data_cfg.data_roots,
        split="val",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=False,
        texture_root=data_cfg.texture_root,
    )

    if int(data_cfg.max_train_samples) > 0 and len(train_dataset) > int(data_cfg.max_train_samples):
        if rank == 0:
            print(f"[Info] Limiting train dataset for test run: {int(data_cfg.max_train_samples)} samples")
        train_dataset = Subset(train_dataset, list(range(int(data_cfg.max_train_samples))))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_num_workers = int(max(0, data_cfg.num_workers))
    val_num_workers = 0 if train_num_workers <= 0 else max(1, train_num_workers // 4)
    use_persistent_workers = bool(data_cfg.persistent_workers)

    if rank == 0:
        print(
            f"[DataLoader] train_workers={train_num_workers} "
            f"val_workers={val_num_workers} persistent_workers={use_persistent_workers}"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=train_num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=data_cfg.prefetch_factor if train_num_workers > 0 else None,
        persistent_workers=use_persistent_workers if train_num_workers > 0 else False,
        collate_fn=dense2geometry_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        num_workers=val_num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=data_cfg.prefetch_factor if val_num_workers > 0 else None,
        persistent_workers=use_persistent_workers if val_num_workers > 0 else False,
        collate_fn=dense2geometry_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, train_sampler


def build_model_and_optim(rank: int, world_size: int, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig):
    device = torch.device(f"cuda:{rank}")
    resume_model_exists = bool(train_cfg.load_model and os.path.exists(train_cfg.load_model))
    dense_override_after_resume = bool(resume_model_exists and train_cfg.load_dense_model)
    effective_dense_backbone_weights = (
        "none" if dense_override_after_resume else str(model_cfg.dense_backbone_weights)
    )
    if rank == 0 and dense_override_after_resume:
        print(
            "[Init] Skipping dense backbone pretrained init because dense_stage "
            f"will be reloaded from {train_cfg.load_dense_model} after resume."
        )
    dense_stage_cfg = DenseStageConfig(
        d_model=model_cfg.dense_d_model,
        nhead=model_cfg.dense_nhead,
        num_layers=model_cfg.dense_num_layers,
        output_size=data_cfg.image_size,
        transformer_map_size=model_cfg.dense_transformer_map_size,
        backbone_weights=effective_dense_backbone_weights,
        decoder_type=model_cfg.dense_decoder_type,
    )
    model = Dense2Geometry(
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        dense_stage_cfg=dense_stage_cfg,
        dense_checkpoint="" if dense_override_after_resume else train_cfg.load_dense_model,
        freeze_dense_stage=model_cfg.freeze_dense_stage,
        image_memory_size=model_cfg.image_memory_size,
        search_size=model_cfg.search_size,
        landmark_search_size=model_cfg.landmark_search_size,
        local_search_radius=model_cfg.local_search_radius,
        local_min_candidates=model_cfg.local_min_candidates,
        search_chunk_size=model_cfg.search_chunk_size,
        search_distance_threshold=model_cfg.search_distance_threshold,
        search_distance_floor=model_cfg.search_distance_floor,
        search_mad_scale=model_cfg.search_mad_scale,
        search_mask_threshold=model_cfg.search_mask_threshold,
        search_min_geo_magnitude=model_cfg.search_min_geo_magnitude,
    ).to(device)

    resumed_ckpt = None
    if train_cfg.load_model and os.path.exists(train_cfg.load_model):
        ckpt = torch.load(train_cfg.load_model, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        skipped_keys, load_notes = _load_matching_state_dict(model, state_dict)
        resumed_ckpt = ckpt if "model_state_dict" in ckpt else None
        if rank == 0:
            print(f"Loaded checkpoint: {train_cfg.load_model}")
            if skipped_keys:
                print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
            if load_notes:
                print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")

    dense_stage_reloaded = False
    if dense_override_after_resume:
        model.load_dense_stage(train_cfg.load_dense_model)
        dense_stage_reloaded = True
        if rank == 0:
            print(
                "[Resume] Reloaded dense_stage from "
                f"{train_cfg.load_dense_model} after loading {train_cfg.load_model}"
            )

    if world_size > 1 and dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[rank], find_unused_parameters=False, gradient_as_bucket_view=False)

    dense_params = []
    head_params = []
    for name, param in _unwrap_model(model).named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("dense_stage."):
            dense_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if dense_params:
        param_groups.append({"params": dense_params, "lr": train_cfg.lr * 0.1})
    if head_params:
        param_groups.append({"params": head_params, "lr": train_cfg.lr})

    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs, eta_min=1e-6)

    amp_dtype = torch.float16 if train_cfg.amp_dtype == "fp16" else torch.bfloat16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    start_epoch = 0
    best_loss = float("inf")
    skip_resume_optimizer_state = dense_stage_reloaded and not model_cfg.freeze_dense_stage
    if resumed_ckpt is not None:
        if "optimizer_state_dict" in resumed_ckpt:
            if skip_resume_optimizer_state:
                if rank == 0:
                    print(
                        "[Resume] Skipped optimizer state restore because dense_stage was "
                        "reloaded from a different checkpoint and remains trainable."
                    )
            else:
                try:
                    optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
                    if rank == 0:
                        print("[Resume] Restored optimizer state")
                except Exception as exc:
                    if rank == 0:
                        print(f"[Resume] Could not restore optimizer state: {exc}")
        if "scheduler_state_dict" in resumed_ckpt:
            try:
                scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])
                if rank == 0:
                    print("[Resume] Restored scheduler state")
            except Exception as exc:
                if rank == 0:
                    print(f"[Resume] Could not restore scheduler state: {exc}")
        if "scaler_state_dict" in resumed_ckpt:
            try:
                scaler.load_state_dict(resumed_ckpt["scaler_state_dict"])
                if rank == 0:
                    print("[Resume] Restored scaler state")
            except Exception as exc:
                if rank == 0:
                    print(f"[Resume] Could not restore scaler state: {exc}")
        if "epoch" in resumed_ckpt:
            start_epoch = int(resumed_ckpt["epoch"]) + 1
        if "best_loss" in resumed_ckpt:
            best_loss = float(resumed_ckpt["best_loss"])
        elif "val_loss" in resumed_ckpt:
            best_loss = float(resumed_ckpt["val_loss"])

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "amp_dtype": amp_dtype,
        "start_epoch": start_epoch,
        "best_loss": best_loss,
    }


def _mesh_weights_6d(batch, device: torch.device) -> torch.Tensor:
    weights = batch["mesh_loss_weights"].to(device, non_blocking=True).float()
    return weights.unsqueeze(-1).repeat(1, 1, 6)


def _update_mesh_metrics(metrics: MetricAccumulator, mesh_pred: torch.Tensor, mesh_gt: torch.Tensor, mesh_w6: torch.Tensor) -> None:
    err = (mesh_pred - mesh_gt).abs()
    metrics.update_sum_count("mesh_3d", (err[..., :3] * mesh_w6[..., :3]).sum(), mesh_w6[..., :3].sum())
    metrics.update_sum_count("mesh_2d", (err[..., 3:5] * mesh_w6[..., 3:5]).sum(), mesh_w6[..., 3:5].sum())
    metrics.update_sum_count("mesh_depth", (err[..., 5:6] * mesh_w6[..., 5:6]).sum(), mesh_w6[..., 5:6].sum())


def _update_search_metrics(
    metrics: MetricAccumulator,
    searched_uv: torch.Tensor,
    match_mask: torch.Tensor,
    mesh_gt: torch.Tensor,
    mesh_loss_weights: torch.Tensor,
    image_size: int,
) -> None:
    metrics.update_sum_count("search_accept", match_mask.sum(), match_mask.numel())
    valid = (match_mask > 0.5) & (mesh_loss_weights > 0)
    if bool(valid.any().item()):
        px_err = (searched_uv[valid] - mesh_gt[..., 3:5][valid]).abs().mean() * float(image_size)
        metrics.update_sum_count("search_uv_px", px_err, 1.0)


def train_one_epoch(epoch: int, rank: int, world_size: int, data_cfg: DataConfig, built, train_loader, train_sampler):
    model = built["model"]
    optimizer = built["optimizer"]
    scaler = built["scaler"]
    amp_dtype = built["amp_dtype"]
    device = torch.device(f"cuda:{rank}")

    train_sampler.set_epoch(epoch)
    model.train()
    train_loss = 0.0
    num_batches = 0
    metrics = MetricAccumulator()
    visualization_batch = None

    pbar = tqdm(train_loader, desc="Training") if rank == 0 else train_loader
    for batch in pbar:
        if rank == 0 and visualization_batch is None:
            keep_keys = {
                "rgb",
                "mesh",
                "mesh_loss_weights",
                "image_path",
                "align5pts",
                "align5pts_valid",
                "align_applied",
                "detection_source",
            }
            visualization_batch = {
                k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
                if k in keep_keys
            }
        rgb = batch["rgb"].to(device, non_blocking=True)
        mesh_gt = batch["mesh"].to(device, non_blocking=True)
        mesh_w6 = _mesh_weights_6d(batch, device)
        mesh_loss_weights = batch["mesh_loss_weights"].to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(rgb)
            mesh_pred = outputs["mesh"].float()
            loss = compute_weighted_l1(mesh_pred, mesh_gt, mesh_w6)
            # Deep supervision: auxiliary losses from intermediate layers
            aux_weight = 0.3
            for aux_mesh in outputs.get("intermediate_meshes", []):
                loss = loss + aux_weight * compute_weighted_l1(aux_mesh.float(), mesh_gt, mesh_w6)

        loss_finite_local = torch.isfinite(loss).to(device=device, dtype=torch.int32)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_finite_local, op=dist.ReduceOp.MIN)
        if not bool(loss_finite_local.item() > 0):
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        train_loss += float(loss.item())
        num_batches += 1

        with torch.no_grad():
            _update_mesh_metrics(metrics, mesh_pred, mesh_gt, mesh_w6)
            _update_search_metrics(
                metrics,
                outputs["searched_uv"].detach(),
                outputs["match_mask"].detach(),
                mesh_gt,
                mesh_loss_weights,
                data_cfg.image_size,
            )

        if rank == 0 and isinstance(pbar, tqdm):
            postfix = {"loss": f"{train_loss / max(num_batches, 1):.4f}"}
            if metrics.has("mesh_3d"):
                postfix["3D"] = f"{metrics.mean('mesh_3d'):.5f}"
            if metrics.has("mesh_2d"):
                postfix["px"] = f"{metrics.mean('mesh_2d') * data_cfg.image_size:.2f}"
            if metrics.has("search_accept"):
                postfix["match"] = f"{metrics.mean('search_accept') * 100.0:.1f}%"
            pbar.set_postfix(postfix)

    if world_size > 1:
        loss_tensor = torch.tensor([train_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor[0].item()
        num_batches = loss_tensor[1].item()

    metric_tensor = torch.tensor(
        [
            metrics.get_sum("mesh_3d"),
            metrics.get_count("mesh_3d"),
            metrics.get_sum("mesh_2d"),
            metrics.get_count("mesh_2d"),
            metrics.get_sum("mesh_depth"),
            metrics.get_count("mesh_depth"),
            metrics.get_sum("search_accept"),
            metrics.get_count("search_accept"),
            metrics.get_sum("search_uv_px"),
            metrics.get_count("search_uv_px"),
        ],
        device=device,
        dtype=torch.float32,
    )
    if world_size > 1:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    return {
        "avg_train_loss": train_loss / max(num_batches, 1),
        "avg_train_3d_error": metric_tensor[0].item() / max(metric_tensor[1].item(), 1e-6) if metric_tensor[1].item() > 0 else None,
        "avg_train_2d_pixel_error": (metric_tensor[2].item() / max(metric_tensor[3].item(), 1e-6) * data_cfg.image_size) if metric_tensor[3].item() > 0 else None,
        "avg_train_depth_error": metric_tensor[4].item() / max(metric_tensor[5].item(), 1e-6) if metric_tensor[5].item() > 0 else None,
        "avg_search_accept_ratio": metric_tensor[6].item() / max(metric_tensor[7].item(), 1e-6) if metric_tensor[7].item() > 0 else None,
        "avg_search_uv_pixel_error": metric_tensor[8].item() / max(metric_tensor[9].item(), 1e-6) if metric_tensor[9].item() > 0 else None,
        "visualization_batch": visualization_batch,
    }


def validate_one_epoch(rank: int, world_size: int, data_cfg: DataConfig, built, val_loader):
    model = built["model"]
    amp_dtype = built["amp_dtype"]
    device = torch.device(f"cuda:{rank}")
    model.eval()
    val_loss = 0.0
    num_batches = 0
    metrics = MetricAccumulator()

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            mesh_gt = batch["mesh"].to(device, non_blocking=True)
            mesh_w6 = _mesh_weights_6d(batch, device)
            mesh_loss_weights = batch["mesh_loss_weights"].to(device, non_blocking=True).float()

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = model(rgb)
                mesh_pred = outputs["mesh"].float()
                loss = compute_weighted_l1(mesh_pred, mesh_gt, mesh_w6)

            val_loss += float(loss.item())
            num_batches += 1

            _update_mesh_metrics(metrics, mesh_pred, mesh_gt, mesh_w6)
            _update_search_metrics(
                metrics,
                outputs["searched_uv"],
                outputs["match_mask"],
                mesh_gt,
                mesh_loss_weights,
                data_cfg.image_size,
            )

    if world_size > 1:
        loss_tensor = torch.tensor([val_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor[0].item()
        num_batches = loss_tensor[1].item()

    metric_tensor = torch.tensor(
        [
            metrics.get_sum("mesh_3d"),
            metrics.get_count("mesh_3d"),
            metrics.get_sum("mesh_2d"),
            metrics.get_count("mesh_2d"),
            metrics.get_sum("mesh_depth"),
            metrics.get_count("mesh_depth"),
            metrics.get_sum("search_accept"),
            metrics.get_count("search_accept"),
            metrics.get_sum("search_uv_px"),
            metrics.get_count("search_uv_px"),
        ],
        device=device,
        dtype=torch.float32,
    )
    if world_size > 1:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    return {
        "avg_val_loss": val_loss / max(num_batches, 1),
        "avg_val_3d_error": metric_tensor[0].item() / max(metric_tensor[1].item(), 1e-6) if metric_tensor[1].item() > 0 else None,
        "avg_val_2d_pixel_error": (metric_tensor[2].item() / max(metric_tensor[3].item(), 1e-6) * data_cfg.image_size) if metric_tensor[3].item() > 0 else None,
        "avg_val_depth_error": metric_tensor[4].item() / max(metric_tensor[5].item(), 1e-6) if metric_tensor[5].item() > 0 else None,
        "avg_search_accept_ratio": metric_tensor[6].item() / max(metric_tensor[7].item(), 1e-6) if metric_tensor[7].item() > 0 else None,
        "avg_search_uv_pixel_error": metric_tensor[8].item() / max(metric_tensor[9].item(), 1e-6) if metric_tensor[9].item() > 0 else None,
    }


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def predict_test_folder(
    epoch: int,
    model: nn.Module,
    device: torch.device,
    test_folder: str,
    output_dir: str,
    mesh_topology: dict | None,
    mesh_restore_indices: np.ndarray | None,
    image_size: int = 1024,
    amp_dtype: torch.dtype = torch.float16,
) -> None:
    """Run model prediction on test images using the same alignment helper as dense-image training."""
    from align_5pt_helper import Align5PtHelper

    test_path = Path(test_folder)
    if not test_path.exists():
        return
    image_files = sorted(
        p for p in test_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_files:
        return

    # Initialize alignment helper (same helper/algorithm as dense_image_dataset, no train jitter)
    align_helper = Align5PtHelper(
        image_size=image_size,
    )

    os.makedirs(output_dir, exist_ok=True)

    raw_model = _unwrap_model(model)
    raw_model.eval()
    vis_size = 512

    def _restore(values: np.ndarray) -> np.ndarray:
        out = np.asarray(values).copy()
        if mesh_restore_indices is not None:
            out = out[mesh_restore_indices]
        return out

    def _prepare_overlay_points(coords_uv: np.ndarray, w: int, h: int) -> np.ndarray:
        pts = np.asarray(coords_uv, dtype=np.float32).copy()
        invalid = ~np.isfinite(pts).all(axis=1)
        invalid |= (pts[:, 0] < 0.0) | (pts[:, 1] < 0.0)
        pts[invalid] = np.nan
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)
        return pts

    def _approximate_points_from_bbox(w: float, h: float) -> np.ndarray:
        return np.array(
            [
                [0.35 * w, 0.38 * h],
                [0.65 * w, 0.38 * h],
                [0.50 * w, 0.56 * h],
                [0.42 * w, 0.74 * h],
                [0.58 * w, 0.74 * h],
            ],
            dtype=np.float32,
        )

    for img_path in image_files:
        basename = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[TestPredict] Failed to read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Resize to image_size for face detection if needed
        if h_orig != image_size or w_orig != image_size:
            detect_rgb = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        else:
            detect_rgb = img_rgb

        lm6, detection_source = align_helper.detect_landmarks(detect_rgb, fallback_lm_px=None)
        m = align_helper.estimate_alignment_matrix(lm6, src_w=image_size, src_h=image_size, split="val")

        img_aligned = cv2.warpAffine(
            detect_rgb,
            m,
            (image_size, image_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Draw aligned 5 points for visualization when available.
        aligned_5pt_vis = img_aligned.copy()
        _5pt_colors = [(255, 80, 80), (80, 200, 80), (80, 150, 255), (255, 200, 50), (220, 80, 220)]
        key5_px, key5_valid = align_helper.extract_key5_from_lm68(lm6)
        if bool(key5_valid.any()):
            key5_aligned = align_helper.transform_points_px(key5_px, m)
            for pt, valid, color in zip(key5_aligned, key5_valid, _5pt_colors):
                if float(valid) <= 0.5 or not np.isfinite(pt).all():
                    continue
                x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
                cv2.circle(aligned_5pt_vis, (x, y), 8, color, -1, cv2.LINE_AA)
                cv2.circle(aligned_5pt_vis, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
        elif detection_source == "none":
            print(f"[TestPredict] Warning: alignment landmarks not found for {img_path}; using fallback centering.")

        # Convert to tensor [1, 3, H, W] in [0, 1]
        rgb_tensor = torch.from_numpy(
            img_aligned.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0).to(device)

        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outputs = raw_model(rgb_tensor)

        # Extract predictions
        pred_mesh = outputs["mesh"][0].detach().cpu().float().numpy()
        searched_uv = outputs["searched_uv"][0].detach().cpu().float().numpy()
        match_mask = outputs["match_mask"][0].detach().cpu().float().numpy()
        pred_geo = outputs["pred_geo"][0].detach().cpu().float().numpy()
        pred_mask_logits = outputs["pred_mask_logits"][0].detach().cpu()
        pred_mask = torch.sigmoid(pred_mask_logits[0]).numpy()

        # Build visualization panels
        src_vis = cv2.resize(detect_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        aligned_5pt_vis = cv2.resize(aligned_5pt_vis, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Geo map
        geo_vis = np.transpose(pred_geo, (1, 2, 0))
        geo_vis = np.nan_to_num(geo_vis, nan=0.0, posinf=1.0, neginf=0.0)
        geo_vis = np.clip(geo_vis, 0.0, 1.0)
        mask_resized = cv2.resize(np.clip(pred_mask, 0.0, 1.0), (geo_vis.shape[1], geo_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
        geo_vis = geo_vis * mask_resized[..., None]
        geo_vis = cv2.resize((geo_vis * 255.0).astype(np.uint8), (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Basecolor map
        pred_basecolor = outputs["pred_basecolor"][0].detach().cpu().float().numpy()
        base_vis = np.transpose(pred_basecolor, (1, 2, 0))
        base_vis = np.nan_to_num(base_vis, nan=0.0, posinf=1.0, neginf=0.0)
        base_vis = np.clip(base_vis, 0.0, 1.0) * mask_resized[..., None]
        base_vis = cv2.resize((base_vis * 255.0).astype(np.uint8), (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Normal map
        pred_normal = outputs["pred_normal"][0].detach().cpu().float().numpy()
        normal_vis = np.transpose(pred_normal, (1, 2, 0))
        normal_vis = np.nan_to_num(normal_vis, nan=0.0, posinf=1.0, neginf=0.0)
        normal_vis = np.clip(normal_vis, 0.0, 1.0) * mask_resized[..., None]
        normal_vis = cv2.resize((normal_vis * 255.0).astype(np.uint8), (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Searched points overlay on aligned image
        searched_uv_r = _restore(searched_uv.copy())
        match_mask_r = _restore(match_mask > 0.5)
        searched_uv_r[~match_mask_r] = -1.0

        searched_pts = _prepare_overlay_points(searched_uv_r, image_size, image_size)
        searched_overlay = img_aligned.copy()
        for idx in range(searched_pts.shape[0]):
            px, py = searched_pts[idx]
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            color = (0, 220, 0) if match_mask_r[idx] else (0, 80, 255)
            cv2.circle(searched_overlay, (int(round(px)), int(round(py))), 1, color, -1, cv2.LINE_AA)
        searched_overlay = cv2.resize(searched_overlay, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Predicted mesh overlay on aligned image
        pred_uv = _restore(pred_mesh[:, 3:5])
        pred_pts = _prepare_overlay_points(pred_uv, image_size, image_size)
        if mesh_topology:
            pred_overlay = create_combined_overlay(img_aligned, pred_pts, mesh_topology)
        else:
            pred_overlay = img_aligned.copy()
            for idx in range(pred_pts.shape[0]):
                px, py = pred_pts[idx]
                if np.isfinite(px) and np.isfinite(py):
                    cv2.circle(pred_overlay, (int(round(px)), int(round(py))), 1, (80, 255, 255), -1, cv2.LINE_AA)
        pred_overlay = cv2.resize(pred_overlay, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        # Compose: src | aligned+5pt | geo | basecolor | normal | searched | mesh overlay
        row = np.concatenate([src_vis, aligned_5pt_vis, geo_vis, base_vis, normal_vis, searched_overlay, pred_overlay], axis=1)
        out_path = os.path.join(output_dir, f"{basename}.png")
        cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    print(f"[TestPredict] Saved epoch {epoch + 1} test predictions to {output_dir}")


def log_and_checkpoint(epoch: int, rank: int, built, train_out, val_out, writer, train_cfg: TrainConfig, mesh_topology, mesh_restore_indices):
    if rank != 0:
        return

    optimizer = built["optimizer"]
    metric_msg = ""
    if val_out["avg_val_3d_error"] is not None:
        metric_msg += f" | Val3D: {val_out['avg_val_3d_error']:.6f}"
    if val_out["avg_val_2d_pixel_error"] is not None:
        metric_msg += f" | Val2Dpx: {val_out['avg_val_2d_pixel_error']:.2f}"
    if val_out["avg_search_accept_ratio"] is not None:
        metric_msg += f" | ValMatch: {val_out['avg_search_accept_ratio'] * 100.0:.2f}%"
    if val_out["avg_search_uv_pixel_error"] is not None:
        metric_msg += f" | ValSearchPx: {val_out['avg_search_uv_pixel_error']:.2f}"
    print(f"Train Loss: {train_out['avg_train_loss']:.6f} | Val Loss: {val_out['avg_val_loss']:.6f}{metric_msg}")

    if writer:
        writer.add_scalar("Loss/Train", train_out["avg_train_loss"], epoch)
        writer.add_scalar("Loss/Val", val_out["avg_val_loss"], epoch)
        if train_out["avg_train_3d_error"] is not None:
            writer.add_scalar("Metrics/Train_3D_Error", train_out["avg_train_3d_error"], epoch)
        if train_out["avg_train_2d_pixel_error"] is not None:
            writer.add_scalar("Metrics/Train_2D_Pixel_Error", train_out["avg_train_2d_pixel_error"], epoch)
        if train_out["avg_search_accept_ratio"] is not None:
            writer.add_scalar("Metrics/Train_Search_Accept", train_out["avg_search_accept_ratio"], epoch)
        if train_out["avg_search_uv_pixel_error"] is not None:
            writer.add_scalar("Metrics/Train_Search_Pixel_Error", train_out["avg_search_uv_pixel_error"], epoch)
        if val_out["avg_val_3d_error"] is not None:
            writer.add_scalar("Metrics/Val_3D_Error", val_out["avg_val_3d_error"], epoch)
        if val_out["avg_val_2d_pixel_error"] is not None:
            writer.add_scalar("Metrics/Val_2D_Pixel_Error", val_out["avg_val_2d_pixel_error"], epoch)
        if val_out["avg_search_accept_ratio"] is not None:
            writer.add_scalar("Metrics/Val_Search_Accept", val_out["avg_search_accept_ratio"], epoch)
        if val_out["avg_search_uv_pixel_error"] is not None:
            writer.add_scalar("Metrics/Val_Search_Pixel_Error", val_out["avg_search_uv_pixel_error"], epoch)
        writer.add_scalar("LR/main", optimizer.param_groups[-1]["lr"], epoch)

    save_dict = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(built["model"]).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": built["scheduler"].state_dict(),
        "scaler_state_dict": built["scaler"].state_dict(),
        "val_loss": val_out["avg_val_loss"],
        "best_loss": built["best_loss"],
    }
    if val_out["avg_val_loss"] < built["best_loss"]:
        built["best_loss"] = val_out["avg_val_loss"]
        save_dict["best_loss"] = built["best_loss"]
        print(f"New best model (Val Loss: {val_out['avg_val_loss']:.6f})")
    save_dir = os.path.dirname(train_cfg.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(save_dict, train_cfg.save_path)
    print(f"Saved checkpoint to {train_cfg.save_path} (epoch {epoch + 1}, Val Loss: {val_out['avg_val_loss']:.6f})")

    if train_out.get("visualization_batch") is not None and mesh_topology:
        try:
            save_dense2geometry_visualizations(
                _unwrap_model(built["model"]),
                train_out["visualization_batch"],
                epoch,
                torch.device("cuda:0"),
                "artifacts/training_samples",
                mesh_topology,
                mesh_restore_indices=mesh_restore_indices,
            )
        except Exception as exc:
            print(f"Warning: failed to save dense2geometry visualizations: {exc}")


def train_worker(rank: int, world_size: int, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, backend: str):
    setup_distributed(rank, world_size, train_cfg.master_port, backend)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    writer = None
    train_loader = None
    val_loader = None
    try:
        mesh_topology = None
        mesh_restore_indices = None
        if rank == 0:
            run_name = f"dense2geometry_{int(time.time())}"
            writer = SummaryWriter(f"artifacts/runs/{run_name}")
            mesh_topology = load_mesh_topology()
            mesh_restore_candidates = [
                os.path.join("assets", "topology", "mesh_inverse.npy"),
                os.path.join("model", "mesh_inverse.npy"),
            ]
            for mesh_restore_path in mesh_restore_candidates:
                if os.path.exists(mesh_restore_path):
                    mesh_restore_indices = np.load(mesh_restore_path)
                    print(
                        f"[Vis] Loaded mesh restore indices from {mesh_restore_path} "
                        f"(filtered={int(mesh_restore_indices.max()) + 1} full={len(mesh_restore_indices)})"
                    )
                    break
            if mesh_restore_indices is None:
                print("[Vis] Warning: mesh restore indices not found; mesh overlays may look scrambled.")

        train_loader, val_loader, train_sampler = create_distributed_dataloaders(data_cfg, rank, world_size)
        built = build_model_and_optim(rank, world_size, data_cfg, model_cfg, train_cfg)

        for epoch in range(built["start_epoch"], train_cfg.epochs):
            if rank == 0:
                print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")
            train_out = train_one_epoch(epoch, rank, world_size, data_cfg, built, train_loader, train_sampler)
            val_out = validate_one_epoch(rank, world_size, data_cfg, built, val_loader)
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            log_and_checkpoint(epoch, rank, built, train_out, val_out, writer, train_cfg, mesh_topology, mesh_restore_indices)
            if rank == 0:
                try:
                    predict_test_folder(
                        epoch=epoch,
                        model=built["model"],
                        device=torch.device("cuda:0"),
                        test_folder="test",
                        output_dir="artifacts/test_predictions",
                        mesh_topology=mesh_topology,
                        mesh_restore_indices=mesh_restore_indices,
                        image_size=data_cfg.image_size,
                        amp_dtype=built["amp_dtype"],
                    )
                except Exception as exc:
                    print(f"[TestPredict] Warning: {exc}")
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.barrier()
            built["scheduler"].step()
    finally:
        _shutdown_dataloader(train_loader)
        _shutdown_dataloader(val_loader)
        if writer:
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
    parser = create_arg_parser("Train Dense2Geometry")
    args = parser.parse_args()
    if sys.platform.startswith("win"):
        launch_windows(args)
    else:
        launch_linux(args)


if __name__ == "__main__":
    main()
