import argparse
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from geometry_transformer import GeometryTransformer
from obj_load_helper import load_uv_obj_file
from metahuman_geometry_dataset import FastGeometryDataset, fast_collate_fn
from train_loss_helper import MetricAccumulator, SimDRLoss, compute_weighted_l1
from train_visualize_helper import (
    load_combined_mesh_uv,
    load_landmark_topology,
    load_mesh_topology,
    save_geometry_visualizations,
)


FIXED_OUTPUT_DIM = 5
FIXED_MESH_TEXTURE_L1_LAMBDA = 1.0
FIXED_TEXTURE_PNG_CACHE_MAX_ITEMS = 16
FIXED_COMBINED_TEXTURE_CACHE_MAX_ITEMS = 0
FIXED_SIMDR_MIN_3D = -0.5
FIXED_SIMDR_MAX_3D = 0.5
FIXED_SIMDR_MIN_2D = -0.5
FIXED_SIMDR_MAX_2D = 0.5


@dataclass
class DataConfig:
    data_roots: list[str]
    texture_root: str
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 1
    persistent_workers: bool = True
    image_size: int = 512
    train_ratio: float = 0.95
    texture_png_cache_max_items: int = FIXED_TEXTURE_PNG_CACHE_MAX_ITEMS
    combined_texture_cache_max_items: int = FIXED_COMBINED_TEXTURE_CACHE_MAX_ITEMS
    max_train_samples: int = 0


@dataclass
class ModelConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    output_dim: int = FIXED_OUTPUT_DIM
    backbone_weights: str = "imagenet"
    model_type: str = "regression"
    k_bins: int = 256
    simdr_min_3d: float = FIXED_SIMDR_MIN_3D
    simdr_max_3d: float = FIXED_SIMDR_MAX_3D
    simdr_min_2d: float = FIXED_SIMDR_MIN_2D
    simdr_max_2d: float = FIXED_SIMDR_MAX_2D
    use_deformable_attention: bool = True
    num_deformable_points: int = 16
    mesh_vertex_feature_dim: int = 16
    mesh_feature_map_size: int = 256
    mesh_texture_size: int = 1024
    freeze_backbone: bool = False


@dataclass
class TrainConfig:
    lr: float = 5e-4
    epochs: int = 200
    load_model: str = ""
    master_port: str = "12355"
    simdr_sigma: float = 2.0
    simdr_l1_lambda: float = 0.1
    deep_supervision_weights: str = ""
    simdr_kl_layers: str = "all"
    amp_dtype: str = "fp16"
    deformable_offset_warmup_epochs: int = 0
    mesh_texture_l1_lambda: float = FIXED_MESH_TEXTURE_L1_LAMBDA


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--persistent_workers", type=_parse_bool_arg, nargs="?", const=True, default=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--master_port", type=str, default="12355")

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--mesh_vertex_feature_dim", type=int, default=16)
    parser.add_argument("--mesh_feature_map_size", type=int, default=256)
    parser.add_argument("--mesh_texture_size", type=int, default=1024)
    parser.add_argument("--backbone_weights", type=str, default="imagenet", choices=["imagenet", "dinov3"])
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--model_type", type=str, default="regression", choices=["regression", "simdr"])
    parser.add_argument("--k_bins", type=int, default=256)
    parser.add_argument("--simdr_sigma", type=float, default=2.0)
    parser.add_argument("--simdr_l1_lambda", type=float, default=0.1)
    parser.add_argument("--deep_supervision_weights", type=str, default="")
    parser.add_argument("--simdr_kl_layers", type=str, default="all", choices=["all", "last2", "final_only"])
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--num_deformable_points", type=int, default=16)
    parser.add_argument("--deformable_offset_warmup_epochs", type=int, default=0)
    parser.add_argument("--use_deformable_attention", type=_parse_bool_arg, nargs="?", const=True, default=True)

    parser.add_argument("--data_roots", nargs="*", default=None)
    parser.add_argument("--texture_root", type=str, default="")
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


def _default_texture_root(platform_name: str) -> str:
    return "/hy-tmp/textures" if platform_name == "linux" else "G:/textures"


def build_configs_from_args(args, platform_name: str) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_roots = list(args.data_roots) if args.data_roots else _default_data_roots(platform_name)
    texture_root = args.texture_root.strip() if args.texture_root else _default_texture_root(platform_name)

    data_cfg = DataConfig(
        data_roots=data_roots,
        texture_root=texture_root,
        batch_size=int(args.batch_size),
        max_train_samples=int(args.max_train_samples),
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=bool(args.persistent_workers),
    )
    model_cfg = ModelConfig(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        backbone_weights=str(args.backbone_weights),
        model_type=str(args.model_type),
        k_bins=int(args.k_bins),
        use_deformable_attention=bool(args.use_deformable_attention),
        num_deformable_points=int(args.num_deformable_points),
        mesh_vertex_feature_dim=int(args.mesh_vertex_feature_dim),
        mesh_feature_map_size=int(args.mesh_feature_map_size),
        mesh_texture_size=int(args.mesh_texture_size),
        freeze_backbone=bool(args.freeze_backbone),
    )
    train_cfg = TrainConfig(
        lr=float(args.lr),
        epochs=int(args.epochs),
        load_model=str(args.load_model),
        master_port=str(args.master_port),
        simdr_sigma=float(args.simdr_sigma),
        simdr_l1_lambda=float(args.simdr_l1_lambda),
        deep_supervision_weights=str(args.deep_supervision_weights),
        simdr_kl_layers=str(args.simdr_kl_layers),
        amp_dtype=str(args.amp_dtype),
        deformable_offset_warmup_epochs=int(args.deformable_offset_warmup_epochs),
    )
    return data_cfg, model_cfg, train_cfg


def load_template_mesh_uv(model_dir: str = "model") -> np.ndarray:
    return load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32, copy=False)


def load_combined_mesh_triangle_faces(model_dir: str = "model") -> np.ndarray:
    part_files = [
        "mesh_head.obj",
        "mesh_eye_l.obj",
        "mesh_eye_r.obj",
        "mesh_mouth.obj",
    ]
    tris = []
    offset = 0
    for file_name in part_files:
        obj_path = os.path.join(model_dir, file_name)
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing mesh OBJ for face loading: {obj_path}")
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(obj_path, triangulate=True)
        if verts is None or uvs is None or v_faces is None:
            raise ValueError(f"Failed to load verts/uv/faces from {obj_path}")
        if len(verts) != len(uvs):
            raise ValueError(
                f"Vertex/UV count mismatch in {obj_path}: verts={len(verts)}, uvs={len(uvs)}"
            )
        tris.append(np.asarray(v_faces, dtype=np.int32) + int(offset))
        offset += int(len(verts))

    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    return np.concatenate(tris, axis=0).astype(np.int32, copy=False)


def remap_triangle_faces_after_vertex_filter(
    triangle_faces: np.ndarray,
    kept_vertex_indices: np.ndarray,
    original_vertex_count: int,
) -> np.ndarray:
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)

    tri_new = remap[tri]
    valid = np.all(tri_new >= 0, axis=1)
    tri_new = tri_new[valid]
    return tri_new.astype(np.int32, copy=False)

def build_deep_supervision_weights(num_layers: int, custom_weights: str = "", progressive: bool = False) -> list[float]:
    if custom_weights:
        vals = [float(x.strip()) for x in custom_weights.split(",") if x.strip()]
        if len(vals) != num_layers:
            raise ValueError(f"deep_supervision_weights expects {num_layers} values, got {len(vals)}")
        weights = np.asarray(vals, dtype=np.float32)
    elif progressive:
        weights = np.arange(1, num_layers + 1, dtype=np.float32)
    else:
        weights = np.ones((num_layers,), dtype=np.float32)
    return (weights / float(weights.sum())).tolist()


def build_simdr_kl_layer_mask(num_layers: int, mode: str) -> list[bool]:
    mode = str(mode).strip().lower()
    if mode == "all":
        return [True] * num_layers
    if mode == "final_only":
        return [False] * (num_layers - 1) + [True]
    if mode == "last2":
        if num_layers == 1:
            return [True]
        return [False] * (num_layers - 2) + [True, True]
    raise ValueError(f"Unsupported simdr_kl_layers: {mode}")


def ensure_parameter_contiguity(module: nn.Module) -> int:
    fixed = 0
    for param in module.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
            fixed += 1
    return fixed


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
    try:
        if dist.is_available() and dist.is_initialized():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def worker_init_fn(_worker_id):
    import cv2

    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def create_distributed_dataloaders(data_cfg: DataConfig, rank: int, world_size: int):
    train_dataset = FastGeometryDataset(
        data_roots=data_cfg.data_roots,
        split="train",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=True,
        texture_root=data_cfg.texture_root,
        texture_png_cache_max_items=data_cfg.texture_png_cache_max_items,
        combined_texture_cache_max_items=data_cfg.combined_texture_cache_max_items,
    )
    val_dataset = FastGeometryDataset(
        data_roots=data_cfg.data_roots,
        split="val",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=False,
        texture_root=data_cfg.texture_root,
        texture_png_cache_max_items=data_cfg.texture_png_cache_max_items,
        combined_texture_cache_max_items=data_cfg.combined_texture_cache_max_items,
    )

    if int(data_cfg.max_train_samples) > 0 and len(train_dataset) > int(data_cfg.max_train_samples):
        if rank == 0:
            print(f"[Info] Limiting train dataset for test run: {int(data_cfg.max_train_samples)} samples")
        train_dataset = Subset(train_dataset, list(range(int(data_cfg.max_train_samples))))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=fast_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
        persistent_workers=bool(data_cfg.persistent_workers) if data_cfg.num_workers > 0 else False,
        collate_fn=fast_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, val_loader, train_sampler


def _load_auxiliary_geometry(rank: int):
    landmark_restore_indices = None
    mesh_restore_indices = None

    template_landmark = np.load("model/landmark_template.npy")
    landmark_indices_path = os.path.join("model", "landmark_indices.npy")
    if os.path.exists(landmark_indices_path):
        landmark_indices = np.load(landmark_indices_path)
        landmark_inverse_path = os.path.join("model", "landmark_inverse.npy")
        if rank == 0 and os.path.exists(landmark_inverse_path):
            landmark_restore_indices = np.load(landmark_inverse_path)
        if landmark_indices.max() < template_landmark.shape[0]:
            template_landmark = template_landmark[landmark_indices]

    template_mesh = np.load("model/mesh_template.npy")
    template_mesh_full_count = int(template_mesh.shape[0])
    template_mesh_uv = load_template_mesh_uv(model_dir="model")
    template_mesh_faces = load_combined_mesh_triangle_faces(model_dir="model")

    mesh_indices_path = os.path.join("model", "mesh_indices.npy")
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path)
        mesh_inverse_path = os.path.join("model", "mesh_inverse.npy")
        if rank == 0 and os.path.exists(mesh_inverse_path):
            mesh_restore_indices = np.load(mesh_inverse_path)
        if mesh_indices.max() < template_mesh.shape[0]:
            template_mesh = template_mesh[mesh_indices]
            if mesh_indices.max() < template_mesh_uv.shape[0]:
                template_mesh_uv = template_mesh_uv[mesh_indices]
            template_mesh_faces = remap_triangle_faces_after_vertex_filter(
                template_mesh_faces,
                mesh_indices,
                original_vertex_count=template_mesh_full_count,
            )

    if template_mesh_uv.shape[0] != template_mesh.shape[0]:
        if template_mesh.shape[1] >= 5:
            template_mesh_uv = template_mesh[:, 3:5].astype(np.float32, copy=True)
        else:
            template_mesh_uv = np.zeros((template_mesh.shape[0], 2), dtype=np.float32)

    if template_mesh_faces.shape[0] == 0:
        raise ValueError("template_mesh_faces is empty after filtering.")
    if template_mesh_faces.max() >= template_mesh.shape[0] or template_mesh_faces.min() < 0:
        raise ValueError(
            f"template_mesh_faces index range invalid for current mesh size {template_mesh.shape[0]}"
        )

    landmark2keypoint_idx = np.load("model/landmark2keypoint_knn_indices.npy")
    landmark2keypoint_w = np.load("model/landmark2keypoint_knn_weights.npy")
    n_keypoint = int(landmark2keypoint_idx.max()) + 1

    mesh2landmark_idx = np.load("model/mesh2landmark_knn_indices.npy")
    mesh2landmark_w = np.load("model/mesh2landmark_knn_weights.npy")

    return {
        "num_landmarks": int(template_landmark.shape[0]),
        "num_mesh": int(template_mesh.shape[0]),
        "template_landmark": template_landmark,
        "template_mesh": template_mesh,
        "template_mesh_uv": template_mesh_uv,
        "template_mesh_faces": template_mesh_faces,
        "landmark2keypoint_idx": landmark2keypoint_idx,
        "landmark2keypoint_w": landmark2keypoint_w,
        "mesh2landmark_idx": mesh2landmark_idx,
        "mesh2landmark_w": mesh2landmark_w,
        "n_keypoint": n_keypoint,
        "landmark_restore_indices": landmark_restore_indices,
        "mesh_restore_indices": mesh_restore_indices,
    }

def prepare_data(rank: int, world_size: int, data_cfg: DataConfig):
    train_loader, val_loader, train_sampler = create_distributed_dataloaders(data_cfg, rank, world_size)
    aux = _load_auxiliary_geometry(rank)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        **aux,
    }


def build_model_and_optim(rank: int, world_size: int, model_cfg: ModelConfig, train_cfg: TrainConfig, prepared):
    device = torch.device(f"cuda:{rank}")

    model = GeometryTransformer(
        num_landmarks=prepared["num_landmarks"],
        num_mesh=prepared["num_mesh"],
        template_landmark=prepared["template_landmark"],
        template_mesh=prepared["template_mesh"],
        template_mesh_uv=prepared["template_mesh_uv"],
        template_mesh_faces=prepared["template_mesh_faces"],
        landmark2keypoint_knn_indices=prepared["landmark2keypoint_idx"],
        landmark2keypoint_knn_weights=prepared["landmark2keypoint_w"],
        mesh2landmark_knn_indices=prepared["mesh2landmark_idx"],
        mesh2landmark_knn_weights=prepared["mesh2landmark_w"],
        n_keypoint=prepared["n_keypoint"],
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        output_dim=model_cfg.output_dim,
        backbone_weights=model_cfg.backbone_weights,
        model_type=model_cfg.model_type,
        flatten_regression_outputs=(model_cfg.model_type != "simdr"),
        k_bins=model_cfg.k_bins,
        simdr_range_3d=(model_cfg.simdr_min_3d, model_cfg.simdr_max_3d),
        simdr_range_2d=(model_cfg.simdr_min_2d, model_cfg.simdr_max_2d),
        use_deformable_attention=(model_cfg.use_deformable_attention if model_cfg.model_type == "simdr" else False),
        num_deformable_points=model_cfg.num_deformable_points,
        use_fast_aux_regression_heads=(model_cfg.model_type == "simdr" and train_cfg.simdr_kl_layers != "all"),
        mesh_vertex_feature_dim=model_cfg.mesh_vertex_feature_dim,
        texture_feature_map_size=model_cfg.mesh_feature_map_size,
        texture_output_size=model_cfg.mesh_texture_size,
    ).to(device)

    if train_cfg.load_model and os.path.exists(train_cfg.load_model):
        ckpt = torch.load(train_cfg.load_model, map_location="cpu")
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        model.load_state_dict(ckpt, strict=False)

    if model_cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    ensure_parameter_contiguity(model)

    # Use DDP only when running real multi-process training.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl":
            model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=False,
                gradient_as_bucket_view=False,
            )
        else:
            model = DDP(
                model,
                find_unused_parameters=False,
                gradient_as_bucket_view=False,
            )

    model_ref = _unwrap_model(model)
    backbone_params, knn_params, head_params = [], [], []
    for name, param in model_ref.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "knn_weights" in name:
            knn_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": train_cfg.lr * 0.01},
            {"params": knn_params, "lr": train_cfg.lr * 0.01},
            {"params": head_params, "lr": train_cfg.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.epochs, eta_min=1e-6)

    if model_cfg.model_type == "simdr":
        criterion_train = SimDRLoss(
            k_bins=model_cfg.k_bins,
            sigma=train_cfg.simdr_sigma,
            min_3d=model_cfg.simdr_min_3d,
            max_3d=model_cfg.simdr_max_3d,
            min_2d=model_cfg.simdr_min_2d,
            max_2d=model_cfg.simdr_max_2d,
        ).to(device)
    else:
        # Regression mode uses robust weighted L1 directly in train loop.
        criterion_train = None

    criterion_val = nn.L1Loss(reduction="none").to(device)

    amp_dtype = torch.float16 if train_cfg.amp_dtype == "fp16" else torch.bfloat16
    if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    layer_weights = build_deep_supervision_weights(
        num_layers=model_cfg.num_layers,
        custom_weights=train_cfg.deep_supervision_weights,
        progressive=(model_cfg.model_type == "simdr"),
    )
    simdr_kl_layer_mask = build_simdr_kl_layer_mask(model_cfg.num_layers, train_cfg.simdr_kl_layers) if model_cfg.model_type == "simdr" else None

    landmark_mask_weights_tensor = None
    mesh_mask_weights_tensor = None
    if os.path.exists("model/landmark_mask.txt"):
        landmark_mask_labels = np.loadtxt("model/landmark_mask.txt").astype(np.float32)
        landmark_mask_weights_tensor = torch.from_numpy(np.repeat(landmark_mask_labels, model_cfg.output_dim)).to(device)
    if os.path.exists("model/mesh_mask.txt"):
        mesh_mask_labels = np.loadtxt("model/mesh_mask.txt").astype(np.float32)
        mesh_mask_weights_tensor = torch.from_numpy(np.repeat(mesh_mask_labels, model_cfg.output_dim)).to(device)

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion_train": criterion_train,
        "criterion_val": criterion_val,
        "scaler": scaler,
        "amp_dtype": amp_dtype,
        "layer_weights": layer_weights,
        "simdr_kl_layer_mask": simdr_kl_layer_mask,
        "landmark_mask_weights_tensor": landmark_mask_weights_tensor,
        "mesh_mask_weights_tensor": mesh_mask_weights_tensor,
        "best_loss": float("inf"),
    }


def _build_batch_weights(batch, key: str, output_dim: int, model_type: str, device):
    if key not in batch:
        return None
    w = batch[key].to(device)
    if model_type == "simdr":
        return w.unsqueeze(-1).repeat(1, 1, output_dim)
    return torch.repeat_interleave(w, output_dim, dim=1)

def train_one_epoch(epoch: int, rank: int, world_size: int, prepared, built, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model = built["model"]
    optimizer = built["optimizer"]
    criterion_train = built["criterion_train"]
    scaler = built["scaler"]
    amp_dtype = built["amp_dtype"]
    layer_weights = built["layer_weights"]
    simdr_kl_layer_mask = built["simdr_kl_layer_mask"]
    landmark_mask_weights_tensor = built["landmark_mask_weights_tensor"]
    mesh_mask_weights_tensor = built["mesh_mask_weights_tensor"]

    train_loader = prepared["train_loader"]
    train_sampler = prepared["train_sampler"]
    device = torch.device(f"cuda:{rank}")

    train_sampler.set_epoch(epoch)
    model.train()
    model_ref = _unwrap_model(model)

    if model_cfg.model_type == "simdr" and hasattr(model_ref, "set_deformable_offset_scale"):
        if model_cfg.use_deformable_attention and train_cfg.deformable_offset_warmup_epochs > 0:
            warmup_denom = max(int(train_cfg.deformable_offset_warmup_epochs), 1)
            model_ref.set_deformable_offset_scale(min(1.0, float(epoch + 1) / float(warmup_denom)))
        else:
            model_ref.set_deformable_offset_scale(1.0)

    train_loss = 0.0
    num_batches = 0
    train_metrics = MetricAccumulator()
    train_oob_sum = torch.zeros(5, device=device)
    train_oob_count = torch.tensor(0.0, device=device)

    pbar = tqdm(train_loader, desc="Training") if rank == 0 else train_loader
    visualization_batch = None

    for batch in pbar:
        if "landmarks" not in batch or "mesh" not in batch:
            continue

        if rank == 0 and visualization_batch is None:
            visualization_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        rgb = batch["rgb"].to(device, non_blocking=True)
        lm_gt_full = batch["landmarks"].to(device, non_blocking=True)
        mesh_gt_full = batch["mesh"].to(device, non_blocking=True)
        mesh_texture_gt_batch = batch["mesh_texture"].to(device, non_blocking=True) if "mesh_texture" in batch else None
        mesh_texture_valid = batch["mesh_texture_valid"].to(device, non_blocking=True) if "mesh_texture_valid" in batch else None

        if model_cfg.model_type == "simdr":
            lm_gt = lm_gt_full
            mesh_gt = mesh_gt_full
        else:
            lm_gt = lm_gt_full.reshape(rgb.shape[0], -1)
            mesh_gt = mesh_gt_full.reshape(rgb.shape[0], -1)

        rgb_in = F.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=True)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        rgb_in = (rgb_in - mean) / std

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            if model_cfg.model_type == "simdr":
                outputs_list = model(
                    rgb_in,
                    return_logits_mask=simdr_kl_layer_mask,
                    use_simdr_mask=simdr_kl_layer_mask,
                    decode_texture_mask=([False] * (model_cfg.num_layers - 1) + [True]),
                )
            else:
                outputs_list = model(rgb_in, decode_texture_mask=([False] * (model_cfg.num_layers - 1) + [True]))

        lm_batch_weights = _build_batch_weights(batch, "landmark_weights", model_cfg.output_dim, model_cfg.model_type, device)
        mesh_batch_weights = _build_batch_weights(batch, "mesh_weights", model_cfg.output_dim, model_cfg.model_type, device)

        if landmark_mask_weights_tensor is not None and "image_path" in batch:
            if lm_batch_weights is None:
                lm_batch_weights = torch.ones_like(lm_gt)
            for i, path in enumerate(batch["image_path"]):
                if "_flux" in os.path.basename(path):
                    lm_batch_weights[i] *= landmark_mask_weights_tensor.view(-1, model_cfg.output_dim) if model_cfg.model_type == "simdr" else landmark_mask_weights_tensor

        if mesh_mask_weights_tensor is not None and "image_path" in batch:
            if mesh_batch_weights is None:
                mesh_batch_weights = torch.ones_like(mesh_gt)
            for i, path in enumerate(batch["image_path"]):
                if "_flux" in os.path.basename(path):
                    mesh_batch_weights[i] *= mesh_mask_weights_tensor.view(-1, model_cfg.output_dim) if model_cfg.model_type == "simdr" else mesh_mask_weights_tensor

        if model_cfg.model_type == "simdr":
            lm_gt_offsets = lm_gt - model_ref.template_landmark.unsqueeze(0)
            mesh_gt_offsets = mesh_gt - model_ref.template_mesh.unsqueeze(0)

        mesh_texture_gt = None
        mesh_texture_loss_weights = None
        mesh_texture_valid_count = None
        if mesh_texture_gt_batch is not None:
            mesh_texture_gt = mesh_texture_gt_batch.float()
            expected_size = int(model_ref.texture_output_size)
            if mesh_texture_gt.shape[-2] != expected_size or mesh_texture_gt.shape[-1] != expected_size:
                mesh_texture_gt = F.interpolate(mesh_texture_gt, size=(expected_size, expected_size), mode="bilinear", align_corners=False)
            mesh_texture_loss_weights = torch.ones_like(mesh_texture_gt)
            if mesh_texture_valid is not None:
                mesh_texture_valid_count = int(mesh_texture_valid.sum().detach().item())
                mesh_texture_loss_weights = mesh_texture_loss_weights * mesh_texture_valid.view(-1, 1, 1, 1).to(device=mesh_texture_gt.device, dtype=mesh_texture_gt.dtype)

        loss = 0.0
        batch_oob_sum = torch.zeros(5, device=device)
        batch_oob_count = 0

        for layer_idx, (layer_w, output) in enumerate(zip(layer_weights, outputs_list)):
            lm_pred = output["landmark"].float()
            mesh_pred = output["mesh"].float()
            mesh_texture_pred = output.get("mesh_texture", None)
            mesh_feature_coverage = output.get("mesh_feature_coverage", None)

            # Track 2D and 3D losses for final layer
            if layer_idx == (model_cfg.num_layers - 1):
                with torch.no_grad():
                    if model_cfg.model_type == "simdr":
                        lm_err = (lm_pred - lm_gt).abs()
                        mesh_err = (mesh_pred - mesh_gt).abs()
                    else:
                        lm_err = (lm_pred.view(rgb.shape[0], -1, model_cfg.output_dim) - lm_gt.view(rgb.shape[0], -1, model_cfg.output_dim)).abs()
                        mesh_err = (mesh_pred.view(rgb.shape[0], -1, model_cfg.output_dim) - mesh_gt.view(rgb.shape[0], -1, model_cfg.output_dim)).abs()
                    
                    if model_cfg.output_dim >= 3:
                        train_metrics.update_sum_count("train_3d", lm_err[..., :3].sum() + mesh_err[..., :3].sum(), 
                                                      lm_err[..., :3].numel() + mesh_err[..., :3].numel())
                    if model_cfg.output_dim >= 5:
                        train_metrics.update_sum_count("train_2d", lm_err[..., 3:5].sum() + mesh_err[..., 3:5].sum(), 
                                                      lm_err[..., 3:5].numel() + mesh_err[..., 3:5].numel())

            if model_cfg.model_type == "simdr":
                lm_l1 = compute_weighted_l1(lm_pred, lm_gt, lm_batch_weights)
                mesh_l1 = compute_weighted_l1(mesh_pred, mesh_gt, mesh_batch_weights)
                if bool(simdr_kl_layer_mask[layer_idx]):
                    lm_kl = criterion_train(output["landmark_logits"].float(), lm_gt_offsets, weights=lm_batch_weights)
                    if criterion_train.last_oob_ratio is not None:
                        batch_oob_sum += criterion_train.last_oob_ratio.to(device)
                        batch_oob_count += 1
                    mesh_kl = criterion_train(output["mesh_logits"].float(), mesh_gt_offsets, weights=mesh_batch_weights)
                    if criterion_train.last_oob_ratio is not None:
                        batch_oob_sum += criterion_train.last_oob_ratio.to(device)
                        batch_oob_count += 1
                    layer_total = (lm_kl + train_cfg.simdr_l1_lambda * lm_l1) + (mesh_kl + train_cfg.simdr_l1_lambda * mesh_l1)
                else:
                    layer_total = lm_l1 + mesh_l1
            else:
                lm_l1 = compute_weighted_l1(lm_pred, lm_gt, lm_batch_weights)
                mesh_l1 = compute_weighted_l1(mesh_pred, mesh_gt, mesh_batch_weights)
                layer_total = lm_l1 + mesh_l1

            if mesh_texture_pred is not None:
                if mesh_texture_gt is not None:
                    tex_weights = mesh_texture_loss_weights
                    if mesh_feature_coverage is not None:
                        cov = mesh_feature_coverage.float()
                        if cov.shape[-2] != mesh_texture_gt.shape[-2] or cov.shape[-1] != mesh_texture_gt.shape[-1]:
                            cov = F.interpolate(cov, size=mesh_texture_gt.shape[-2:], mode="nearest")
                        cov = cov.to(device=mesh_texture_gt.device, dtype=mesh_texture_gt.dtype)
                        if tex_weights is None:
                            tex_weights = cov.expand_as(mesh_texture_gt)
                        else:
                            tex_weights = tex_weights * cov.expand_as(mesh_texture_gt)

                    skip_tex = False
                    if tex_weights is not None and tex_weights.sum().detach().item() <= 0:
                        skip_tex = True
                    if mesh_texture_valid_count is not None and mesh_texture_valid_count <= 0:
                        skip_tex = True
                    if not skip_tex:
                        mesh_texture_l1 = compute_weighted_l1(mesh_texture_pred.float(), mesh_texture_gt, tex_weights)
                        layer_total = layer_total + train_cfg.mesh_texture_l1_lambda * mesh_texture_l1
                        if layer_idx == (model_cfg.num_layers - 1):
                            train_metrics.update_sum_count("mesh_texture_l1", mesh_texture_l1, 1.0)
                    else:
                        # Keep texture branch in autograd graph even with no valid texture target.
                        layer_total = layer_total + (mesh_texture_pred.sum() * 0.0)
                else:
                    # No texture GT in batch: still touch branch so DDP sees all params as used.
                    layer_total = layer_total + (mesh_texture_pred.sum() * 0.0)

            loss += float(layer_w) * layer_total

        if model_cfg.model_type == "simdr" and batch_oob_count > 0:
            train_oob_sum += batch_oob_sum
            train_oob_count += float(batch_oob_count)

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

        train_loss += loss.item()
        num_batches += 1

        # Update progress bar with current metrics
        if rank == 0 and isinstance(pbar, tqdm):
            postfix_dict = {"loss": f"{train_loss / num_batches:.4f}"}
            if train_metrics.has("train_3d"):
                postfix_dict["3D"] = f"{train_metrics.mean('train_3d'):.5f}"
            if train_metrics.has("train_2d"):
                train_2d_px = train_metrics.mean("train_2d") * 1024.0
                postfix_dict["2D_px"] = f"{train_2d_px:.2f}"
            if train_metrics.has("mesh_texture_l1"):
                postfix_dict["tex"] = f"{train_metrics.mean('mesh_texture_l1'):.4f}"
            pbar.set_postfix(postfix_dict)

    if world_size > 1:
        loss_tensor = torch.tensor([train_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor[0].item()
        num_batches = loss_tensor[1].item()

    # Aggregate training metrics across GPUs
    train_metric_tensor = torch.tensor([
        train_metrics.get_sum("train_3d"),
        train_metrics.get_count("train_3d"),
        train_metrics.get_sum("train_2d"),
        train_metrics.get_count("train_2d"),
        train_metrics.get_sum("mesh_texture_l1"),
        train_metrics.get_count("mesh_texture_l1"),
    ], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(train_metric_tensor, op=dist.ReduceOp.SUM)
    
    train_3d_sum, train_3d_count = train_metric_tensor[0].item(), train_metric_tensor[1].item()
    train_2d_sum, train_2d_count = train_metric_tensor[2].item(), train_metric_tensor[3].item()
    train_tex_sum, train_tex_count = train_metric_tensor[4].item(), train_metric_tensor[5].item()

    avg_train_3d_error = train_3d_sum / max(train_3d_count, 1e-6) if train_3d_count > 0 else None
    avg_train_2d_error = train_2d_sum / max(train_2d_count, 1e-6) if train_2d_count > 0 else None
    avg_train_2d_pixel_error = avg_train_2d_error * 1024.0 if avg_train_2d_error is not None else None
    avg_train_mesh_texture_l1 = train_tex_sum / max(train_tex_count, 1e-6) if train_tex_count > 0 else None

    simdr_oob_avg = None
    if model_cfg.model_type == "simdr":
        oob_tensor = torch.cat([train_oob_sum, train_oob_count.unsqueeze(0)])
        if world_size > 1:
            dist.all_reduce(oob_tensor, op=dist.ReduceOp.SUM)
        if oob_tensor[-1].item() > 0:
            simdr_oob_avg = (oob_tensor[:5] / oob_tensor[-1]).detach().cpu()

    return {
        "avg_train_loss": train_loss / max(num_batches, 1),
        "avg_train_3d_error": avg_train_3d_error,
        "avg_train_2d_pixel_error": avg_train_2d_pixel_error,
        "avg_train_mesh_texture_l1": avg_train_mesh_texture_l1,
        "simdr_oob_avg": simdr_oob_avg,
        "visualization_batch": visualization_batch,
    }

def validate_one_epoch(rank: int, world_size: int, prepared, built, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model = built["model"]
    criterion_val = built["criterion_val"]
    landmark_mask_weights_tensor = built["landmark_mask_weights_tensor"]
    mesh_mask_weights_tensor = built["mesh_mask_weights_tensor"]

    val_loader = prepared["val_loader"]
    device = torch.device(f"cuda:{rank}")

    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_metrics = MetricAccumulator()

    with torch.no_grad():
        for batch in val_loader:
            if "landmarks" not in batch or "mesh" not in batch:
                continue

            rgb = batch["rgb"].to(device, non_blocking=True)
            lm_gt_full = batch["landmarks"].to(device, non_blocking=True)
            mesh_gt_full = batch["mesh"].to(device, non_blocking=True)
            mesh_texture_gt_batch = batch["mesh_texture"].to(device, non_blocking=True) if "mesh_texture" in batch else None
            mesh_texture_valid = batch["mesh_texture_valid"].to(device, non_blocking=True) if "mesh_texture_valid" in batch else None

            if model_cfg.model_type == "simdr":
                lm_gt = lm_gt_full
                mesh_gt = mesh_gt_full
            else:
                lm_gt = lm_gt_full.reshape(rgb.shape[0], -1)
                mesh_gt = mesh_gt_full.reshape(rgb.shape[0], -1)

            rgb_in = F.interpolate(rgb, size=(512, 512), mode="bilinear", align_corners=True)
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            rgb_in = (rgb_in - mean) / std

            if model_cfg.model_type == "simdr":
                outputs_list = model(
                    rgb_in,
                    return_logits_mask=[False] * model_cfg.num_layers,
                    predict_layer_mask=([False] * (model_cfg.num_layers - 1) + [True]),
                    decode_texture_mask=([False] * (model_cfg.num_layers - 1) + [True]),
                )
            else:
                outputs_list = model(rgb_in, decode_texture_mask=([False] * (model_cfg.num_layers - 1) + [True]))

            final_output = outputs_list[-1]
            lm_pred = final_output["landmark"]
            mesh_pred = final_output["mesh"]
            mesh_texture_pred = final_output.get("mesh_texture", None)
            mesh_feature_coverage = final_output.get("mesh_feature_coverage", None)

            lm_batch_weights = _build_batch_weights(batch, "landmark_weights", model_cfg.output_dim, model_cfg.model_type, device)
            mesh_batch_weights = _build_batch_weights(batch, "mesh_weights", model_cfg.output_dim, model_cfg.model_type, device)

            if landmark_mask_weights_tensor is not None and "image_path" in batch:
                if lm_batch_weights is None:
                    lm_batch_weights = torch.ones_like(lm_gt)
                for i, path in enumerate(batch["image_path"]):
                    if "_flux" in os.path.basename(path):
                        lm_batch_weights[i] *= landmark_mask_weights_tensor.view(-1, model_cfg.output_dim) if model_cfg.model_type == "simdr" else landmark_mask_weights_tensor

            if mesh_mask_weights_tensor is not None and "image_path" in batch:
                if mesh_batch_weights is None:
                    mesh_batch_weights = torch.ones_like(mesh_gt)
                for i, path in enumerate(batch["image_path"]):
                    if "_flux" in os.path.basename(path):
                        mesh_batch_weights[i] *= mesh_mask_weights_tensor.view(-1, model_cfg.output_dim) if model_cfg.model_type == "simdr" else mesh_mask_weights_tensor

            lm_l1_raw = criterion_val(lm_pred, lm_gt)
            mesh_l1_raw = criterion_val(mesh_pred, mesh_gt)
            lm_loss = (lm_l1_raw * lm_batch_weights).sum() / (lm_batch_weights.sum() + 1e-6) if lm_batch_weights is not None else lm_l1_raw.mean()
            mesh_loss = (mesh_l1_raw * mesh_batch_weights).sum() / (mesh_batch_weights.sum() + 1e-6) if mesh_batch_weights is not None else mesh_l1_raw.mean()
            loss = lm_loss + mesh_loss

            if mesh_texture_gt_batch is not None and mesh_texture_pred is not None:
                mesh_texture_gt = mesh_texture_gt_batch.float()
                expected_size = int(_unwrap_model(model).texture_output_size)
                if mesh_texture_gt.shape[-2] != expected_size or mesh_texture_gt.shape[-1] != expected_size:
                    mesh_texture_gt = F.interpolate(mesh_texture_gt, size=(expected_size, expected_size), mode="bilinear", align_corners=False)

                mesh_texture_l1_raw = criterion_val(mesh_texture_pred, mesh_texture_gt)
                if mesh_texture_valid is not None:
                    w = mesh_texture_valid.view(-1, 1, 1, 1).to(device=mesh_texture_l1_raw.device, dtype=mesh_texture_l1_raw.dtype)
                    if mesh_feature_coverage is not None:
                        cov = mesh_feature_coverage.float()
                        if cov.shape[-2] != mesh_texture_l1_raw.shape[-2] or cov.shape[-1] != mesh_texture_l1_raw.shape[-1]:
                            cov = F.interpolate(cov, size=mesh_texture_l1_raw.shape[-2:], mode="nearest")
                        cov = cov.to(device=mesh_texture_l1_raw.device, dtype=mesh_texture_l1_raw.dtype)
                        w = w * cov
                    w = w.expand_as(mesh_texture_l1_raw)
                    denom = w.sum()
                    if denom.detach().item() > 0:
                        masked = mesh_texture_l1_raw * w
                        mesh_texture_loss = masked.sum() / (denom + 1e-6)
                        val_metrics.update_sum_count("val_mesh_texture_l1", masked.sum(), denom)
                        loss = loss + train_cfg.mesh_texture_l1_lambda * mesh_texture_loss
                else:
                    mesh_texture_loss = mesh_texture_l1_raw.mean()
                    val_metrics.update_sum_count("val_mesh_texture_l1", mesh_texture_l1_raw.sum(), mesh_texture_l1_raw.new_tensor(float(mesh_texture_l1_raw.numel())))
                    loss = loss + train_cfg.mesh_texture_l1_lambda * mesh_texture_loss

            def _acc(pred_m, gt_m, w_m):
                l1 = (pred_m - gt_m).abs()
                if w_m is None:
                    w_m = torch.ones_like(l1)
                if model_cfg.output_dim >= 3:
                    val_metrics.update_sum_count("val_3d", (l1[..., :3] * w_m[..., :3]).sum(), w_m[..., :3].sum())
                if model_cfg.output_dim >= 5:
                    val_metrics.update_sum_count("val_2d", (l1[..., 3:5] * w_m[..., 3:5]).sum(), w_m[..., 3:5].sum())

            if model_cfg.model_type == "simdr":
                _acc(lm_pred, lm_gt, lm_batch_weights)
                _acc(mesh_pred, mesh_gt, mesh_batch_weights)
            else:
                _acc(lm_pred.view(rgb.shape[0], -1, model_cfg.output_dim), lm_gt.view(rgb.shape[0], -1, model_cfg.output_dim), lm_batch_weights.view(rgb.shape[0], -1, model_cfg.output_dim) if lm_batch_weights is not None else None)
                _acc(mesh_pred.view(rgb.shape[0], -1, model_cfg.output_dim), mesh_gt.view(rgb.shape[0], -1, model_cfg.output_dim), mesh_batch_weights.view(rgb.shape[0], -1, model_cfg.output_dim) if mesh_batch_weights is not None else None)

            val_loss += loss.item()
            val_batches += 1

    if world_size > 1:
        loss_tensor = torch.tensor([val_loss, val_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = loss_tensor[0].item()
        val_batches = loss_tensor[1].item()

    metric_tensor = torch.tensor([
        val_metrics.get_sum("val_3d"),
        val_metrics.get_count("val_3d"),
        val_metrics.get_sum("val_2d"),
        val_metrics.get_count("val_2d"),
        val_metrics.get_sum("val_mesh_texture_l1"),
        val_metrics.get_count("val_mesh_texture_l1"),
    ], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    val_3d_sum, val_3d_count = metric_tensor[0].item(), metric_tensor[1].item()
    val_2d_sum, val_2d_count = metric_tensor[2].item(), metric_tensor[3].item()
    tex_sum, tex_count = metric_tensor[4].item(), metric_tensor[5].item()

    avg_val_loss = val_loss / max(val_batches, 1)
    avg_val_3d_error = val_3d_sum / max(val_3d_count, 1e-6) if val_3d_count > 0 else None
    avg_val_2d_error = val_2d_sum / max(val_2d_count, 1e-6) if val_2d_count > 0 else None
    avg_val_2d_pixel_error = avg_val_2d_error * 1024.0 if avg_val_2d_error is not None else None
    avg_val_mesh_texture_l1 = tex_sum / max(tex_count, 1e-6) if tex_count > 0 else None

    return {
        "avg_val_loss": avg_val_loss,
        "avg_val_3d_error": avg_val_3d_error,
        "avg_val_2d_pixel_error": avg_val_2d_pixel_error,
        "avg_val_mesh_texture_l1": avg_val_mesh_texture_l1,
    }


def log_and_checkpoint(epoch: int, rank: int, built, prepared, model_cfg: ModelConfig, train_out, val_out, writer, landmark_topology, mesh_topology):
    if rank != 0:
        return

    model = built["model"]
    optimizer = built["optimizer"]

    metric_msg = ""
    if val_out["avg_val_3d_error"] is not None:
        metric_msg += f" | Val3D: {val_out['avg_val_3d_error']:.6f}"
    if val_out["avg_val_2d_pixel_error"] is not None:
        metric_msg += f" | Val2Dpx: {val_out['avg_val_2d_pixel_error']:.2f}"
    if val_out["avg_val_mesh_texture_l1"] is not None:
        metric_msg += f" | ValTex: {val_out['avg_val_mesh_texture_l1']:.4f}"
    print(f"Train Loss: {train_out['avg_train_loss']:.6f} | Val Loss: {val_out['avg_val_loss']:.6f}{metric_msg}")

    if writer:
        writer.add_scalar("Loss/Train", train_out["avg_train_loss"], epoch)
        writer.add_scalar("Loss/Val", val_out["avg_val_loss"], epoch)
        if train_out["avg_train_mesh_texture_l1"] is not None:
            writer.add_scalar("Loss/MeshTexture_Train_L1", train_out["avg_train_mesh_texture_l1"], epoch)
        if val_out["avg_val_mesh_texture_l1"] is not None:
            writer.add_scalar("Loss/MeshTexture_Val_L1", val_out["avg_val_mesh_texture_l1"], epoch)
        if val_out["avg_val_3d_error"] is not None:
            writer.add_scalar("Metrics/Val_3D_Error", val_out["avg_val_3d_error"], epoch)
        if val_out["avg_val_2d_pixel_error"] is not None:
            writer.add_scalar("Metrics/Val_2D_Pixel_Error", val_out["avg_val_2d_pixel_error"], epoch)
            writer.add_scalar("Metrics/Val_Scaled_Error", val_out["avg_val_2d_pixel_error"], epoch)
        writer.add_scalar("LR/backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/head", optimizer.param_groups[2]["lr"], epoch)

    if val_out["avg_val_loss"] < built["best_loss"]:
        built["best_loss"] = val_out["avg_val_loss"]
        filename = f"best_geometry_transformer_dim{model_cfg.output_dim}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": _unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_out["avg_val_loss"],
                "output_dim": model_cfg.output_dim,
            },
            filename,
        )
        print(f"Saved best model to {filename} (Val Loss: {val_out['avg_val_loss']:.6f})")

    if train_out["visualization_batch"] is not None and landmark_topology and mesh_topology:
        save_geometry_visualizations(
            _unwrap_model(model),
            train_out["visualization_batch"],
            epoch,
            torch.device("cuda:0"),
            "training_samples",
            landmark_topology,
            mesh_topology,
            output_dim=model_cfg.output_dim,
            landmark_restore_indices=prepared["landmark_restore_indices"],
            mesh_restore_indices=prepared["mesh_restore_indices"],
        )


def train_worker(rank: int, world_size: int, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig, backend: str):
    setup_distributed(rank, world_size, train_cfg.master_port, backend)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    writer = None
    landmark_topology = None
    mesh_topology = None
    if rank == 0:
        run_name = f"{model_cfg.backbone_weights}_mt-{model_cfg.model_type}_{int(time.time())}"
        writer = SummaryWriter(f"runs/{run_name}")
        landmark_topology = load_landmark_topology()
        mesh_topology = load_mesh_topology()

    prepared = prepare_data(rank, world_size, data_cfg)
    built = build_model_and_optim(rank, world_size, model_cfg, train_cfg, prepared)

    for epoch in range(train_cfg.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")
        train_out = train_one_epoch(epoch, rank, world_size, prepared, built, model_cfg, train_cfg)
        val_out = validate_one_epoch(rank, world_size, prepared, built, model_cfg, train_cfg)
        log_and_checkpoint(epoch, rank, built, prepared, model_cfg, train_out, val_out, writer, landmark_topology, mesh_topology)
        built["scheduler"].step()

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
