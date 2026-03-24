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

try:
    import nvdiffrast.torch as dr
    _NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    _NVDIFFRAST_AVAILABLE = False

from geometry_transformer import GeometryTransformer
from data_utils.obj_io import load_uv_obj_file
from geometry_dataset import GeometryDataset, fast_collate_fn
from train_loss_helper import MetricAccumulator, MeshSmoothnessLoss, SimDRLoss, compute_weighted_l1
from train_visualize_helper import (
    derive_depth_from_3d_to_2d_torch,
    load_combined_mesh_uv,
    load_landmark_topology,
    load_mesh_topology,
    save_geometry_visualizations,
)


FIXED_OUTPUT_DIM = 6
FIXED_MESH_TEXTURE_L1_LAMBDA = 1.0
FIXED_TEXTURE_PNG_CACHE_MAX_ITEMS = 16
FIXED_COMBINED_TEXTURE_CACHE_MAX_ITEMS = 0
FIXED_SIMDR_MIN_3D = -0.5
FIXED_SIMDR_MAX_3D = 0.5
FIXED_SIMDR_MIN_2D = -0.5
FIXED_SIMDR_MAX_2D = 0.5
FIXED_BASECOLOR_RENDER_L1_LAMBDA = 1.0
FIXED_MESH_SMOOTHNESS_LAMBDA = 0.2


@dataclass
class DataConfig:
    data_roots: list[str]
    texture_root: str
    synthetic_data_roots: list[str] = None  # Folders with full GT (2D/3D/texture)
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
    basecolor_render_l1_lambda: float = FIXED_BASECOLOR_RENDER_L1_LAMBDA
    normal_render_l1_lambda: float = 0.0
    geo_render_l1_lambda: float = 1.0
    real_data_geo_start_epoch: int = 10
    real_data_basecolor_start_epoch: int = 10
    mesh_smoothness_lambda: float = FIXED_MESH_SMOOTHNESS_LAMBDA
    real_data_warmup_epochs: int = 5
    test_mode: bool = False
    test_preview_every: int = 100
    ddp_find_unused_parameters: bool = False


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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=6)
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
    parser.add_argument("--real_data_geo_start_epoch", type=int, default=10,
                        help="Epochs after resume before enabling real data geo render loss")
    parser.add_argument("--real_data_basecolor_start_epoch", type=int, default=10,
                        help="Epochs after resume before enabling real data basecolor render loss")
    parser.add_argument("--mesh_smoothness_lambda", type=float, default=FIXED_MESH_SMOOTHNESS_LAMBDA)
    parser.add_argument("--real_data_warmup_epochs", type=int, default=5)
    parser.add_argument("--use_deformable_attention", type=_parse_bool_arg, nargs="?", const=True, default=True)

    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--test_preview_every", type=int, default=100)
    parser.add_argument("--ddp_find_unused_parameters", type=_parse_bool_arg, nargs="?", const=True, default=False)

    parser.add_argument("--data_roots", nargs="*", default=None)
    parser.add_argument("--texture_root", type=str, default="")
    return parser


def build_configs_from_args(args) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    data_roots = list(args.data_roots) if args.data_roots else []
    texture_root = args.texture_root.strip() if args.texture_root else ""
    
    # Get synthetic_data_roots if specified
    synthetic_data_roots = getattr(args, 'synthetic_data_roots', None)

    data_cfg = DataConfig(
        data_roots=data_roots,
        texture_root=texture_root,
        synthetic_data_roots=synthetic_data_roots,
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
        real_data_geo_start_epoch=int(args.real_data_geo_start_epoch),
        real_data_basecolor_start_epoch=int(args.real_data_basecolor_start_epoch),
        mesh_smoothness_lambda=float(args.mesh_smoothness_lambda),
        real_data_warmup_epochs=int(args.real_data_warmup_epochs),
        test_mode=bool(getattr(args, "test_mode", False)),
        test_preview_every=int(getattr(args, "test_preview_every", 100)),
        ddp_find_unused_parameters=bool(getattr(args, "ddp_find_unused_parameters", False)),
    )
    return data_cfg, model_cfg, train_cfg


def load_template_mesh_uv(model_dir: str = "assets/topology") -> np.ndarray:
    return load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32, copy=False)


def load_combined_mesh_triangle_faces(model_dir: str = "assets/topology") -> np.ndarray:
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
    vertex_positions: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remap face indices from original vertex space to filtered vertex space.
    Filtered-out vertices are substituted with their nearest kept neighbor
    (using vertex_positions for distance, or index proximity as fallback),
    so no faces are dropped and back-of-head triangles are preserved.
    """
    tri = np.asarray(triangle_faces, dtype=np.int64)
    if tri.size == 0:
        return np.zeros((0, 3), dtype=np.int32)

    kept = np.asarray(kept_vertex_indices, dtype=np.int64)
    remap = np.full((int(original_vertex_count),), -1, dtype=np.int64)
    remap[kept] = np.arange(kept.shape[0], dtype=np.int64)

    # Find filtered-out vertices and substitute with nearest kept neighbor
    all_indices = np.arange(int(original_vertex_count), dtype=np.int64)
    filtered = all_indices[remap < 0]
    if filtered.size > 0:
        if vertex_positions is not None and vertex_positions.shape[0] == int(original_vertex_count):
            pos = np.asarray(vertex_positions, dtype=np.float32)
            kept_pos = pos[kept]                             # [K, D]
            filt_pos = pos[filtered]                         # [F, D]
            # Batched nearest-neighbor via squared distances
            diff = filt_pos[:, None, :] - kept_pos[None, :, :]   # [F, K, D]
            nn_idx = (diff * diff).sum(axis=-1).argmin(axis=-1)   # [F]
        else:
            # Fallback: nearest by index value
            nn_idx = np.searchsorted(kept, filtered).clip(0, kept.shape[0] - 1)
        remap[filtered] = nn_idx

    tri_new = remap[tri]
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


def _load_matching_state_dict(model, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    """Load state dict, skipping keys with incompatible shapes."""
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


def create_distributed_dataloaders(data_cfg: DataConfig, rank: int, world_size: int):
    train_dataset = GeometryDataset(
        data_roots=data_cfg.data_roots,
        split="train",
        image_size=data_cfg.image_size,
        train_ratio=data_cfg.train_ratio,
        augment=True,
        texture_root=data_cfg.texture_root,
        texture_png_cache_max_items=data_cfg.texture_png_cache_max_items,
        combined_texture_cache_max_items=data_cfg.combined_texture_cache_max_items,
    )
    val_dataset = GeometryDataset(
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

    template_landmark = np.load("assets/topology/landmark_template.npy")
    landmark_indices_path = os.path.join("model", "landmark_indices.npy")
    if os.path.exists(landmark_indices_path):
        landmark_indices = np.load(landmark_indices_path)
        landmark_inverse_path = os.path.join("model", "landmark_inverse.npy")
        if rank == 0 and os.path.exists(landmark_inverse_path):
            landmark_restore_indices = np.load(landmark_inverse_path)
        if landmark_indices.max() < template_landmark.shape[0]:
            template_landmark = template_landmark[landmark_indices]

    template_mesh = np.load("assets/topology/mesh_template.npy")
    template_mesh_full_count = int(template_mesh.shape[0])
    template_mesh_uv = load_template_mesh_uv(model_dir="assets/topology")
    template_mesh_uv_full = template_mesh_uv.copy()
    # Load full faces first (N_full indexed) — preserved as template_mesh_faces_full for rendering
    template_mesh_faces_full = load_combined_mesh_triangle_faces(model_dir="assets/topology")
    template_mesh_faces = template_mesh_faces_full.copy()   # will be remapped to N_unique for normals

    mesh_indices_path = os.path.join("model", "mesh_indices.npy")
    if os.path.exists(mesh_indices_path):
        mesh_indices = np.load(mesh_indices_path)
        mesh_inverse_path = os.path.join("model", "mesh_inverse.npy")
        if os.path.exists(mesh_inverse_path):
            mesh_restore_indices = np.load(mesh_inverse_path)
        if mesh_indices.max() < template_mesh.shape[0]:
            template_mesh_full_positions = template_mesh.copy()
            template_mesh = template_mesh[mesh_indices]
            if mesh_indices.max() < template_mesh_uv.shape[0]:
                template_mesh_uv = template_mesh_uv[mesh_indices]
            # Remap faces to N_unique index space (with NN substitution) for compute_vertex_normals
            template_mesh_faces = remap_triangle_faces_after_vertex_filter(
                template_mesh_faces_full,
                mesh_indices,
                original_vertex_count=template_mesh_full_count,
                vertex_positions=template_mesh_full_positions[:, :3],
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
    if template_mesh_faces_full.size > 0 and template_mesh_faces_full.max() >= template_mesh_full_count:
        raise ValueError(
            f"template_mesh_faces_full index out of range [0, {template_mesh_full_count - 1}]."
        )

    landmark2keypoint_idx = np.load("assets/topology/landmark2keypoint_knn_indices.npy")
    landmark2keypoint_w = np.load("assets/topology/landmark2keypoint_knn_weights.npy")
    n_keypoint = int(landmark2keypoint_idx.max()) + 1

    mesh2landmark_idx = np.load("assets/topology/mesh2landmark_knn_indices.npy")
    mesh2landmark_w = np.load("assets/topology/mesh2landmark_knn_weights.npy")

    return {
        "num_landmarks": int(template_landmark.shape[0]),
        "num_mesh": int(template_mesh.shape[0]),
        "template_landmark": template_landmark,
        "template_mesh": template_mesh,
        "template_mesh_uv": template_mesh_uv,
        "template_mesh_uv_full": template_mesh_uv_full,
        "template_mesh_faces": template_mesh_faces,
        "template_mesh_faces_full": template_mesh_faces_full,
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

    # Create a synthetic-only loader used before real_data_{geo,basecolor}_start_epoch
    synthetic_train_loader = None
    synthetic_train_sampler = None
    if (
        data_cfg.synthetic_data_roots
        and sorted(data_cfg.synthetic_data_roots) != sorted(data_cfg.data_roots)
    ):
        synthetic_cfg = DataConfig(
            data_roots=data_cfg.synthetic_data_roots,
            texture_root=data_cfg.texture_root,
            batch_size=data_cfg.batch_size,
            num_workers=data_cfg.num_workers,
            prefetch_factor=data_cfg.prefetch_factor,
            persistent_workers=data_cfg.persistent_workers,
            max_train_samples=data_cfg.max_train_samples,
        )
        syn_loader, _, syn_sampler = create_distributed_dataloaders(synthetic_cfg, rank, world_size)
        synthetic_train_loader = syn_loader
        synthetic_train_sampler = syn_sampler

    aux = _load_auxiliary_geometry(rank)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_sampler": train_sampler,
        "synthetic_train_loader": synthetic_train_loader,
        "synthetic_train_sampler": synthetic_train_sampler,
        "synthetic_data_roots": list(data_cfg.synthetic_data_roots) if data_cfg.synthetic_data_roots else None,
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
        template_mesh_uv_full=prepared.get("template_mesh_uv_full"),
        template_mesh_faces=prepared["template_mesh_faces"],
        template_mesh_faces_full=prepared.get("template_mesh_faces_full"),
        mesh_restore_indices=prepared.get("mesh_restore_indices"),
        landmark2keypoint_knn_indices=prepared["landmark2keypoint_idx"],
        landmark2keypoint_knn_weights=prepared["landmark2keypoint_w"],
        mesh2landmark_knn_indices=prepared["mesh2landmark_idx"],
        mesh2landmark_knn_weights=prepared["mesh2landmark_w"],
        n_keypoint=prepared["n_keypoint"],
        d_model=model_cfg.d_model,
        nhead=model_cfg.nhead,
        num_layers=model_cfg.num_layers,
        backbone_weights=model_cfg.backbone_weights,
        model_type=model_cfg.model_type,
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

    if model_cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    ensure_parameter_contiguity(model)

    # Use DDP only when running real multi-process training.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        ddp_find_unused = bool(getattr(train_cfg, "ddp_find_unused_parameters", False))
        if dist.get_backend() == "nccl":
            model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=ddp_find_unused,
                gradient_as_bucket_view=False,
            )
        else:
            model = DDP(
                model,
                find_unused_parameters=ddp_find_unused,
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

    # Restore training state from checkpoint if available
    start_epoch = 0
    best_loss = float("inf")
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
            # If scheduler state wasn't restored, fast-forward it to match resume epoch
            if "scheduler_state_dict" not in resumed_ckpt and start_epoch > 0:
                for _ in range(start_epoch):
                    scheduler.step()
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"[Resume] Fast-forwarded scheduler to epoch {start_epoch}, LR={current_lr:.6e}")
        if "best_loss" in resumed_ckpt:
            best_loss = float(resumed_ckpt["best_loss"])
        elif "val_loss" in resumed_ckpt:
            best_loss = float(resumed_ckpt["val_loss"])

    layer_weights = build_deep_supervision_weights(
        num_layers=model_cfg.num_layers,
        custom_weights=train_cfg.deep_supervision_weights,
        progressive=(model_cfg.model_type == "simdr"),
    )
    simdr_kl_layer_mask = build_simdr_kl_layer_mask(model_cfg.num_layers, train_cfg.simdr_kl_layers) if model_cfg.model_type == "simdr" else None

    landmark_mask_weights_tensor = None
    mesh_mask_weights_tensor = None
    if os.path.exists("assets/topology/landmark_mask.txt"):
        landmark_mask_labels = np.loadtxt("assets/topology/landmark_mask.txt").astype(np.float32)
        landmark_mask_weights_tensor = torch.from_numpy(np.repeat(landmark_mask_labels, 6, axis=0)).to(device)
    if os.path.exists("assets/topology/mesh_mask.txt"):
        mesh_mask_labels = np.loadtxt("assets/topology/mesh_mask.txt").astype(np.float32)
        mesh_mask_weights_tensor = torch.from_numpy(np.repeat(mesh_mask_labels, 6, axis=0)).to(device)

    mesh_smooth_loss_fn = None
    if train_cfg.mesh_smoothness_lambda > 0:
        mesh_smooth_loss_fn = MeshSmoothnessLoss(
            faces=prepared["template_mesh_faces"],
            template=prepared["template_mesh"],
        ).to(device)

    # Load face mask texture for masking render losses (neck/back of head)
    real_face_mask_texture = None
    face_mask_path = os.path.join("model", "real_dataset_face_mask.png")
    if os.path.exists(face_mask_path):
        import cv2 as _cv2
        mask_img = _cv2.imread(face_mask_path, _cv2.IMREAD_COLOR)  # BGR
        if mask_img is not None:
            # Mask is RGB with: red=back-of-head, green=neck, white=face
            # Store as [1, 3, H, W] so rendered mask preserves per-channel info
            mask_f = mask_img[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
            mask_t = torch.from_numpy(mask_f.copy()).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)  # [1, 3, H, W]
            real_face_mask_texture = mask_t
            if rank == 0:
                print(f"[Info] Loaded face mask texture (color): {face_mask_path} shape={mask_img.shape}")

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
        "mesh_smooth_loss_fn": mesh_smooth_loss_fn,
        "real_face_mask_texture": real_face_mask_texture,
        "best_loss": best_loss,
        "start_epoch": start_epoch,
    }


def _build_batch_weights(batch, key: str, model_type: str, device):
    if key not in batch:
        return None
    w = batch[key].to(device)
    if model_type == "simdr":
        return w.unsqueeze(-1).repeat(1, 1, 6)
    return torch.repeat_interleave(w, 6, dim=1)


def _merge_batch_weight_masks(primary: torch.Tensor | None, extra: torch.Tensor | None) -> torch.Tensor | None:
    if primary is None:
        return extra
    if extra is None:
        return primary
    return primary * extra


def _build_domain_masks(batch, synthetic_data_roots, device):
    """Split a batch into synthetic and real samples from image_path prefixes."""
    if "image_path" not in batch or not synthetic_data_roots:
        return None, None

    synthetic_roots = tuple(os.path.normpath(r) for r in synthetic_data_roots)
    batch_size = len(batch["image_path"])
    synthetic_sample_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    real_sample_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    for i, path in enumerate(batch["image_path"]):
        path_normalized = os.path.normpath(path)
        is_synthetic = any(path_normalized.startswith(root) for root in synthetic_roots)
        synthetic_sample_mask[i] = bool(is_synthetic)
        real_sample_mask[i] = not bool(is_synthetic)

    return synthetic_sample_mask, real_sample_mask


def render_vertex_attrs_to_image(model_ref, mesh_pred: torch.Tensor, vertex_attrs: torch.Tensor, out_h: int, out_w: int, use_pred_depth: bool = True):
    """
    Rasterize per-vertex attributes directly to screen space via nvdiffrast interpolation.
    Skips UV atlas entirely — no flood fill needed.
    Args:
        mesh_pred:    [B, N, 6] predicted mesh (xy in cols 3:5, depth in col 5)
        vertex_attrs: [B, N, C] per-vertex values to interpolate (e.g. normals in [0,1])
    Returns:
        color: [B, C, H, W], cov: [B, 1, H, W]
    """
    if (not _NVDIFFRAST_AVAILABLE) or vertex_attrs is None or mesh_pred is None:
        return None, None

    if mesh_pred.ndim == 2:
        mesh_pred = mesh_pred.view(mesh_pred.shape[0], -1, int(model_ref.output_dim))
    if mesh_pred.shape[-1] < 5:
        return None, None

    device = mesh_pred.device
    if device.type != "cuda":
        return None, None

    try:
        ctx = model_ref._get_raster_context(device)
    except Exception:
        return None, None

    bsz, n_verts, _ = mesh_pred.shape
    if int(n_verts) != int(model_ref.num_mesh):
        return None, None

    # Expand N_unique → N_full using restore mapping so all original faces are covered
    restore_idx = model_ref.mesh_restore_indices.to(device=device)   # [N_full]
    mesh_pred_full = mesh_pred[:, restore_idx]                        # [B, N_full, D]
    attrs_full = vertex_attrs[:, restore_idx]                         # [B, N_full, C]

    xy = mesh_pred_full[..., 3:5].float()
    x_clip = torch.nan_to_num(xy[..., 0] * 2.0 - 1.0,       nan=0.0, posinf=2.0, neginf=-2.0)
    y_clip = torch.nan_to_num(1.0 - xy[..., 1] * 2.0,        nan=0.0, posinf=2.0, neginf=-2.0)

    # Derive depth from 3D→2D alignment
    depth_scale, depth_bias = 0.8, 0.1
    if use_pred_depth and mesh_pred_full.shape[-1] >= 5:
        derived_depth = derive_depth_from_3d_to_2d_torch(
            mesh_pred_full[..., :3], mesh_pred_full[..., 3:5],
        )
        z_clip = -(derived_depth.float().clamp(0.0, 1.0) * depth_scale + depth_bias)
    else:
        z_clip = torch.zeros_like(x_clip)
    z_clip = torch.nan_to_num(z_clip, nan=0.0, posinf=0.0, neginf=0.0)

    clip_pos = torch.stack([x_clip, y_clip, z_clip, torch.ones_like(x_clip)], dim=-1).contiguous()

    tri = model_ref.template_mesh_faces_full.to(device=device, dtype=torch.int32)
    render_h, render_w = int(out_h) * 2, int(out_w) * 2

    attrs = attrs_full.float().contiguous()

    with torch.amp.autocast(device_type=device.type, enabled=False):
        rast, _ = dr.rasterize(ctx, clip_pos, tri, resolution=[render_h, render_w])
        color, _ = dr.interpolate(attrs, rast, tri)          # [B, H, W, C]
        color = dr.antialias(color, rast, clip_pos, tri)
        cov = (rast[..., 3:4] > 0).to(dtype=color.dtype)
        color = color * cov
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        cov   = torch.nan_to_num(cov,   nan=0.0, posinf=0.0, neginf=0.0)

    color = color.permute(0, 3, 1, 2).contiguous().to(dtype=vertex_attrs.dtype)
    cov   = cov.permute(0, 3, 1, 2).contiguous().to(dtype=vertex_attrs.dtype)
    color = F.interpolate(color, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
    cov   = F.interpolate(cov,   size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
    color = torch.flip(color, dims=[2])
    cov   = torch.flip(cov,   dims=[2])
    return color, cov


def render_mesh_texture_to_image(model_ref, mesh_pred: torch.Tensor, mesh_texture: torch.Tensor, out_h: int, out_w: int, use_pred_depth: bool = True, antialias: bool = True):
    if (not _NVDIFFRAST_AVAILABLE) or mesh_texture is None:
        return None, None
    if mesh_pred is None:
        return None, None

    if mesh_pred.ndim == 2:
        mesh_pred = mesh_pred.view(mesh_pred.shape[0], -1, int(model_ref.output_dim))
    elif mesh_pred.ndim != 3:
        return None, None
    if mesh_pred.shape[-1] < 5:
        return None, None

    device = mesh_pred.device
    if device.type != "cuda":
        return None, None

    try:
        ctx = model_ref._get_raster_context(device)
    except Exception:
        return None, None

    bsz, n_verts, _ = mesh_pred.shape
    if int(n_verts) != int(model_ref.num_mesh):
        return None, None

    # Expand N_unique → N_full using restore mapping so all original faces are covered
    restore_idx = model_ref.mesh_restore_indices.to(device=device)   # [N_full]
    mesh_pred_full = mesh_pred[:, restore_idx]                        # [B, N_full, D]

    xy = mesh_pred_full[..., 3:5].float()
    x_clip = torch.nan_to_num(xy[..., 0] * 2.0 - 1.0,  nan=0.0, posinf=2.0, neginf=-2.0)
    y_clip = torch.nan_to_num(1.0 - xy[..., 1] * 2.0,   nan=0.0, posinf=2.0, neginf=-2.0)

    # Derive depth from 3D→2D alignment
    depth_scale = 0.8
    depth_bias = 0.1
    if use_pred_depth and mesh_pred_full.shape[-1] >= 5:
        derived_depth = derive_depth_from_3d_to_2d_torch(
            mesh_pred_full[..., :3], mesh_pred_full[..., 3:5],
        )
        z_clip = -(derived_depth.float().clamp(0.0, 1.0) * depth_scale + depth_bias)
    else:
        z_clip = torch.zeros_like(x_clip)
    z_clip = torch.nan_to_num(z_clip, nan=0.0, posinf=0.0, neginf=0.0)

    clip_pos = torch.stack([x_clip, y_clip, z_clip, torch.ones_like(x_clip)], dim=-1).contiguous()

    tri = model_ref.template_mesh_faces_full.to(device=device, dtype=torch.int32)
    uv_full = model_ref.template_mesh_uv_full.to(device=device, dtype=torch.float32)
    uv = uv_full.unsqueeze(0).expand(bsz, -1, -1).contiguous()
    tex = mesh_texture.float().permute(0, 2, 3, 1).contiguous()

    # Render at 2x resolution for anti-aliasing, then downscale (skip for masks)
    render_h = int(out_h) * 2 if antialias else int(out_h)
    render_w = int(out_w) * 2 if antialias else int(out_w)

    with torch.amp.autocast(device_type=device.type, enabled=False):
        rast, _ = dr.rasterize(ctx, clip_pos, tri, resolution=[render_h, render_w])
        uv_pix, _ = dr.interpolate(uv, rast, tri)
        if bool(getattr(model_ref, "flip_uv_v", True)):
            uv_pix = torch.stack([uv_pix[..., 0], 1.0 - uv_pix[..., 1]], dim=-1)
        uv_pix = uv_pix.clamp(0.0, 1.0)

        color = dr.texture(tex, uv_pix, filter_mode="linear", boundary_mode="clamp")
        if antialias:
            color = dr.antialias(color, rast, clip_pos, tri)
        cov = (rast[..., 3:4] > 0).to(dtype=color.dtype)
        color = color * cov
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    color = color.permute(0, 3, 1, 2).contiguous().to(dtype=mesh_texture.dtype)
    cov = cov.permute(0, 3, 1, 2).contiguous().to(dtype=mesh_texture.dtype)

    if antialias:
        # Downsample from 2x to target resolution
        color = F.interpolate(color, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
        cov = F.interpolate(cov, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
    
    # Flip vertically after rendering
    color = torch.flip(color, dims=[2])
    cov = torch.flip(cov, dims=[2])
    
    return color, cov

def _save_test_preview(
    epoch: int,
    step: int,
    batch,
    model_ref,
    mesh_pred: torch.Tensor,
    mesh_texture_pred: torch.Tensor | None,
    real_face_mask_texture: torch.Tensor | None = None,
    output_dir: str = "test_previews",
):
    """Save a visualization grid for real samples in test mode.

    Grid columns per sample:
      input_rgb | gt_basecolor | rendered_basecolor | basecolor_diff
                | gt_geo       | rendered_geo       | geo_diff
    """
    import cv2 as _cv2

    os.makedirs(output_dir, exist_ok=True)
    bsz = mesh_pred.shape[0]

    rgb_np = batch["rgb"][:bsz].detach().cpu().numpy()  # [B, 3, H, W]
    out_h, out_w = rgb_np.shape[2], rgb_np.shape[3]

    # GT basecolor and geo from dataset
    gt_basecolor_np = batch["basecolor"][:bsz].detach().cpu().numpy() if "basecolor" in batch else None
    gt_geo_np = batch["geo_gt"][:bsz].detach().cpu().numpy() if "geo_gt" in batch else None

    # Render basecolor from predicted mesh + texture
    render_bc_np = None
    if mesh_texture_pred is not None and _NVDIFFRAST_AVAILABLE:
        try:
            render_bc, render_bc_cov = render_mesh_texture_to_image(
                model_ref, mesh_pred, mesh_texture_pred,
                out_h=out_h, out_w=out_w, use_pred_depth=True,
            )
            if render_bc is not None:
                render_bc_np = (render_bc * render_bc_cov).detach().cpu().numpy()
        except Exception:
            pass

    # Render geo from predicted mesh + static geo atlas
    render_geo_np = None
    if hasattr(model_ref, "geo_feature_atlas") and _NVDIFFRAST_AVAILABLE:
        try:
            geo_texture = model_ref.geo_feature_atlas.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()
            geo_texture_rs = F.interpolate(geo_texture, size=(out_h, out_w), mode="bilinear", align_corners=False)
            render_geo, render_geo_cov = render_mesh_texture_to_image(
                model_ref, mesh_pred, geo_texture_rs,
                out_h=out_h, out_w=out_w, use_pred_depth=True,
            )
            if render_geo is not None:
                render_geo_np = (render_geo * render_geo_cov).detach().cpu().numpy()
        except Exception:
            pass

    # Render face mask to screen space
    render_mask_np = None
    if real_face_mask_texture is not None and _NVDIFFRAST_AVAILABLE:
        try:
            mask_tex = real_face_mask_texture.expand(bsz, -1, -1, -1).contiguous()
            render_mask, render_mask_cov = render_mesh_texture_to_image(
                model_ref, mesh_pred, mask_tex,
                out_h=out_h, out_w=out_w, use_pred_depth=True, antialias=False,
            )
            if render_mask is not None:
                r_ch = render_mask[:, 0:1].detach()
                g_ch = render_mask[:, 1:2].detach()
                cov1 = render_mask_cov[:, :1]
                is_red = (r_ch > 0.3) & (g_ch < 0.3) & (cov1 > 0)
                is_green = (g_ch > 0.3) & (r_ch < 0.3) & (cov1 > 0)
                red_expanded = F.max_pool2d(is_red.float(), kernel_size=21, stride=1, padding=10)
                green_expanded = F.max_pool2d(is_green.float(), kernel_size=5, stride=1, padding=2)
                fm = 1.0 - (red_expanded + green_expanded).clamp(0.0, 1.0)
                render_mask_np = fm.cpu().numpy()
        except Exception:
            pass

    def _to_hwc(arr):
        """Convert [C, H, W] float to [H, W, 3] uint8."""
        img = np.clip(np.transpose(arr, (1, 2, 0))[:, :, :3], 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    def _diff_map(a, b):
        """Absolute diff, amplified x3 for visibility."""
        d = np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0) * 3.0
        return (np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)

    def _placeholder(h, w):
        return np.zeros((h, w, 3), dtype=np.uint8)

    rows = []
    for i in range(bsz):
        h, w = out_h, out_w
        input_img = _to_hwc(rgb_np[i])

        # Face mask for this sample (HxW float, 0..1)
        if render_mask_np is not None:
            mask_gray = np.clip(render_mask_np[i, 0], 0.0, 1.0)
            mask_rgb = np.stack([mask_gray] * 3, axis=-1)
            mask_img = (mask_rgb * 255.0).astype(np.uint8)
        else:
            mask_gray = None
            mask_img = _placeholder(h, w)

        # Basecolor row: gt | rendered | diff
        gt_bc = _to_hwc(gt_basecolor_np[i]) if gt_basecolor_np is not None else _placeholder(h, w)
        rd_bc = _to_hwc(render_bc_np[i]) if render_bc_np is not None else _placeholder(h, w)
        bc_diff = _diff_map(gt_bc, rd_bc) if (gt_basecolor_np is not None and render_bc_np is not None) else _placeholder(h, w)
        if mask_gray is not None:
            bc_diff = (bc_diff.astype(np.float32) * mask_rgb).astype(np.uint8)

        # Geo row: gt | rendered | diff
        gt_geo = _to_hwc(gt_geo_np[i]) if gt_geo_np is not None else _placeholder(h, w)
        rd_geo = _to_hwc(render_geo_np[i]) if render_geo_np is not None else _placeholder(h, w)
        geo_diff = _diff_map(gt_geo, rd_geo) if (gt_geo_np is not None and render_geo_np is not None) else _placeholder(h, w)
        if mask_gray is not None:
            geo_diff = (geo_diff.astype(np.float32) * mask_rgb).astype(np.uint8)

        # Build grid: 2 rows x 5 cols
        # Row 1: input | gt_basecolor | rendered_basecolor | basecolor_diff | face_mask
        # Row 2: input | gt_geo       | rendered_geo       | geo_diff       | face_mask
        row_bc = np.concatenate([input_img, gt_bc, rd_bc, bc_diff, mask_img], axis=1)
        row_geo = np.concatenate([input_img, gt_geo, rd_geo, geo_diff, mask_img], axis=1)
        sample_grid = np.concatenate([row_bc, row_geo], axis=0)
        rows.append(sample_grid)

    if rows:
        canvas = np.concatenate(rows, axis=0)
        out_path = os.path.join(output_dir, f"test_e{epoch + 1:03d}_s{step:05d}.png")
        _cv2.imwrite(out_path, _cv2.cvtColor(canvas, _cv2.COLOR_RGB2BGR))
        print(f"[TestMode] Saved preview: {out_path}")


def train_one_epoch(epoch: int, rank: int, world_size: int, prepared, built, data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model = built["model"]
    optimizer = built["optimizer"]
    criterion_train = built["criterion_train"]
    scaler = built["scaler"]
    amp_dtype = built["amp_dtype"]
    layer_weights = built["layer_weights"]
    simdr_kl_layer_mask = built["simdr_kl_layer_mask"]
    landmark_mask_weights_tensor = built["landmark_mask_weights_tensor"]
    mesh_mask_weights_tensor = built["mesh_mask_weights_tensor"]
    mesh_smooth_loss_fn = built.get("mesh_smooth_loss_fn", None)
    real_face_mask_texture = built.get("real_face_mask_texture", None)

    train_loader = prepared["train_loader"]
    train_sampler = prepared["train_sampler"]
    device = torch.device(f"cuda:{rank}")

    train_sampler.set_epoch(epoch)
    model.train()
    model_ref = _unwrap_model(model)

    # real_data_{geo,basecolor}_start_epoch count from the loaded/resume epoch
    start_epoch = built["start_epoch"]
    epochs_since_load = epoch - start_epoch  # 0-based count since resume

    if rank == 0:
        geo_start = int(train_cfg.real_data_geo_start_epoch)
        bc_start = int(train_cfg.real_data_basecolor_start_epoch)
        print(f"[Epoch {epoch + 1}] epochs_since_load={epochs_since_load}, "
              f"real_data_geo_start={geo_start}, real_data_basecolor_start={bc_start}")

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
        if rank == 0 and visualization_batch is None:
            visualization_batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        rgb = batch["rgb"].to(device, non_blocking=True)
        basecolor_gt = batch["basecolor"].to(device, non_blocking=True) if "basecolor" in batch else None
        basecolor_valid = batch["basecolor_valid"].to(device, non_blocking=True) if "basecolor_valid" in batch else None
        geo_gt = batch["geo_gt"].to(device, non_blocking=True) if "geo_gt" in batch else None
        geo_valid = batch["geo_valid"].to(device, non_blocking=True) if "geo_valid" in batch else None

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

        # Check GT data for NaN
        if torch.isnan(lm_gt).any() or torch.isinf(lm_gt).any():
            print(f"[NaN DETECTED] lm_gt contains NaN/inf at batch {num_batches}")
            print(f"  lm_gt shape: {lm_gt.shape}")
            print(f"  NaN count: {torch.isnan(lm_gt).sum().item()}")
            if "image_path" in batch:
                print(f"  Sample paths: {batch['image_path'][:2]}")
        if torch.isnan(mesh_gt).any() or torch.isinf(mesh_gt).any():
            print(f"[NaN DETECTED] mesh_gt contains NaN/inf at batch {num_batches}")
            print(f"  mesh_gt shape: {mesh_gt.shape}")
            print(f"  NaN count: {torch.isnan(mesh_gt).sum().item()}")
            if "image_path" in batch:
                print(f"  Sample paths: {batch['image_path'][:2]}")

        lm_batch_weights = _build_batch_weights(batch, "landmark_weights", model_cfg.model_type, device)
        mesh_batch_weights = _build_batch_weights(batch, "mesh_weights", model_cfg.model_type, device)
        lm_found_mask = _build_batch_weights(batch, "landmark_found_mask", model_cfg.model_type, device)
        mesh_found_mask = _build_batch_weights(batch, "mesh_found_mask", model_cfg.model_type, device)
        lm_batch_weights = _merge_batch_weight_masks(lm_batch_weights, lm_found_mask)
        mesh_batch_weights = _merge_batch_weight_masks(mesh_batch_weights, mesh_found_mask)
        
        # Check if all weights are zero
        if lm_batch_weights is not None:
            lm_weight_sum = lm_batch_weights.sum().item()
            if lm_weight_sum == 0:
                print(f"[WARNING] All landmark weights are zero at batch {num_batches}")
        if mesh_batch_weights is not None:
            mesh_weight_sum = mesh_batch_weights.sum().item()
            if mesh_weight_sum == 0:
                print(f"[WARNING] All mesh weights are zero at batch {num_batches}")

        # Split mixed batches into synthetic-vs-real supervision paths.
        # Synthetic: GT mesh/landmark loss plus UV texture loss on the de-duplicated topology.
        # Real: render/image supervision only (no GT geometry or UV texture-space loss).
        synthetic_sample_mask, real_sample_mask = _build_domain_masks(
            batch,
            getattr(data_cfg, "synthetic_data_roots", None),
            device,
        )
        if real_sample_mask is not None and bool(real_sample_mask.any().item()):
            if lm_batch_weights is not None:
                lm_batch_weights = lm_batch_weights.clone()
                lm_batch_weights[real_sample_mask] = 0.0
            if mesh_batch_weights is not None:
                mesh_batch_weights = mesh_batch_weights.clone()
                mesh_batch_weights[real_sample_mask] = 0.0
            if mesh_texture_valid is not None:
                mesh_texture_valid = mesh_texture_valid.clone()
                mesh_texture_valid[real_sample_mask] = 0.0

        # Synthetic data still keeps UV texture supervision; skip only screen-space basecolor/geo render losses.
        if synthetic_sample_mask is not None and bool(synthetic_sample_mask.any().item()):
            if basecolor_valid is not None:
                basecolor_valid = basecolor_valid.clone()
                basecolor_valid[synthetic_sample_mask] = 0.0
            if geo_valid is not None:
                geo_valid = geo_valid.clone()
                geo_valid[synthetic_sample_mask] = 0.0

        # Delay real-data training (render/texture supervision) until configured epoch.
        # start_epoch counts are relative to resume epoch (epochs_since_load).
        _REAL_WARMUP_SCHEDULE = [0.01, 0.03, 0.1, 0.2, 0.4, 0.6]

        def _real_data_warmup_weight(start_ep):
            """Compute warmup weight for a real-data channel given its start epoch offset."""
            if start_ep <= 0:
                return 1.0
            if epochs_since_load < start_ep:
                return 0.0
            es = epochs_since_load - start_ep + 1
            if es <= len(_REAL_WARMUP_SCHEDULE):
                return _REAL_WARMUP_SCHEDULE[es - 1]
            return _REAL_WARMUP_SCHEDULE[-1]

        geo_start = int(train_cfg.real_data_geo_start_epoch)
        bc_start = int(train_cfg.real_data_basecolor_start_epoch)

        if real_sample_mask is not None and bool(real_sample_mask.any().item()) and (geo_start > 0 or bc_start > 0):
            geo_warmup = _real_data_warmup_weight(geo_start)
            bc_warmup = _real_data_warmup_weight(bc_start)
            if num_batches == 0 and rank == 0:
                print(f"[Info] Real data warmup scale: geo={geo_warmup:.3f}, basecolor={bc_warmup:.3f} "
                      f"(epochs_since_load={epochs_since_load}, geo_start={geo_start}, bc_start={bc_start})")
            if basecolor_valid is not None:
                basecolor_valid = basecolor_valid.clone()
                if bc_warmup <= 0:
                    basecolor_valid[real_sample_mask] = 0.0
                else:
                    basecolor_valid[real_sample_mask] *= float(bc_warmup)
            if geo_valid is not None:
                geo_valid = geo_valid.clone()
                if geo_warmup <= 0:
                    geo_valid[real_sample_mask] = 0.0
                else:
                    geo_valid[real_sample_mask] *= float(geo_warmup)

        if landmark_mask_weights_tensor is not None and "image_path" in batch:
            if lm_batch_weights is None:
                lm_batch_weights = torch.ones_like(lm_gt)
            for i, path in enumerate(batch["image_path"]):
                if "_flux" in os.path.basename(path):
                    lm_batch_weights[i] *= landmark_mask_weights_tensor.view(-1, 6) if model_cfg.model_type == "simdr" else landmark_mask_weights_tensor

        if mesh_mask_weights_tensor is not None and "image_path" in batch:
            if mesh_batch_weights is None:
                mesh_batch_weights = torch.ones_like(mesh_gt)
            for i, path in enumerate(batch["image_path"]):
                if "_flux" in os.path.basename(path):
                    mesh_batch_weights[i] *= mesh_mask_weights_tensor.view(-1, 6) if model_cfg.model_type == "simdr" else mesh_mask_weights_tensor

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

            if layer_idx == (model_cfg.num_layers - 1):
                with torch.no_grad():
                    if model_cfg.model_type == "simdr":
                        lm_err = (lm_pred - lm_gt).abs()
                        mesh_err = (mesh_pred - mesh_gt).abs()
                        lm_metric_weights = lm_batch_weights if lm_batch_weights is not None else torch.ones_like(lm_err)
                        mesh_metric_weights = mesh_batch_weights if mesh_batch_weights is not None else torch.ones_like(mesh_err)
                    else:
                        lm_err = (lm_pred.view(rgb.shape[0], -1, 6) - lm_gt.view(rgb.shape[0], -1, 6)).abs()
                        mesh_err = (mesh_pred.view(rgb.shape[0], -1, 6) - mesh_gt.view(rgb.shape[0], -1, 6)).abs()
                        lm_metric_weights = lm_batch_weights.view(rgb.shape[0], -1, 6) if lm_batch_weights is not None else torch.ones_like(lm_err)
                        mesh_metric_weights = mesh_batch_weights.view(rgb.shape[0], -1, 6) if mesh_batch_weights is not None else torch.ones_like(mesh_err)

                    train_metrics.update_sum_count(
                        "train_3d",
                        (lm_err[..., :3] * lm_metric_weights[..., :3]).sum() + (mesh_err[..., :3] * mesh_metric_weights[..., :3]).sum(),
                        lm_metric_weights[..., :3].sum() + mesh_metric_weights[..., :3].sum(),
                    )
                    train_metrics.update_sum_count(
                        "train_2d",
                        (lm_err[..., 3:5] * lm_metric_weights[..., 3:5]).sum() + (mesh_err[..., 3:5] * mesh_metric_weights[..., 3:5]).sum(),
                        lm_metric_weights[..., 3:5].sum() + mesh_metric_weights[..., 3:5].sum(),
                    )

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
                lm_l1 = compute_weighted_l1(lm_pred.reshape(rgb.shape[0], -1), lm_gt, lm_batch_weights)
                mesh_l1 = compute_weighted_l1(mesh_pred.reshape(rgb.shape[0], -1), mesh_gt, mesh_batch_weights)
                
                # Check for NaN in geometry losses
                if torch.isnan(lm_l1) or torch.isinf(lm_l1):
                    print(f"[NaN DETECTED] lm_l1 is NaN/inf at batch {num_batches}, layer {layer_idx}")
                    print(f"  lm_pred shape: {lm_pred.shape}, contains NaN: {torch.isnan(lm_pred).any()}")
                    print(f"  lm_gt shape: {lm_gt.shape}, contains NaN: {torch.isnan(lm_gt).any()}")
                    if lm_batch_weights is not None:
                        print(f"  lm_batch_weights sum: {lm_batch_weights.sum().item()}")
                if torch.isnan(mesh_l1) or torch.isinf(mesh_l1):
                    print(f"[NaN DETECTED] mesh_l1 is NaN/inf at batch {num_batches}, layer {layer_idx}")
                    print(f"  mesh_pred shape: {mesh_pred.shape}, contains NaN: {torch.isnan(mesh_pred).any()}")
                    print(f"  mesh_gt shape: {mesh_gt.shape}, contains NaN: {torch.isnan(mesh_gt).any()}")
                    if mesh_batch_weights is not None:
                        print(f"  mesh_batch_weights sum: {mesh_batch_weights.sum().item()}")
                
                layer_total = lm_l1 + mesh_l1

            if mesh_smooth_loss_fn is not None and layer_idx == (model_cfg.num_layers - 1):
                mesh_pred_6d = mesh_pred if mesh_pred.ndim == 3 else mesh_pred.view(rgb.shape[0], -1, 6)
                s3d, s2d, sdepth = mesh_smooth_loss_fn(mesh_pred_6d.float())
                smooth_lambda = float(train_cfg.mesh_smoothness_lambda)
                if smooth_lambda > 0 and torch.isfinite(s3d):
                    smooth_3d_total = smooth_lambda * s3d
                    layer_total = layer_total + smooth_3d_total
                    train_metrics.update_sum_count("smooth_3d", smooth_3d_total.detach(), 1.0)
                if smooth_lambda > 0 and torch.isfinite(s2d):
                    smooth_2d_total = smooth_lambda * s2d
                    layer_total = layer_total + smooth_2d_total
                    train_metrics.update_sum_count("smooth_2d", smooth_2d_total.detach(), 1.0)
                if smooth_lambda > 0 and torch.isfinite(sdepth):
                    smooth_depth_total = smooth_lambda * sdepth
                    layer_total = layer_total + smooth_depth_total
                    train_metrics.update_sum_count("smooth_depth", smooth_depth_total.detach(), 1.0)

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
                        if torch.isfinite(mesh_texture_l1):
                            layer_total = layer_total + train_cfg.mesh_texture_l1_lambda * mesh_texture_l1
                            if layer_idx == (model_cfg.num_layers - 1):
                                train_metrics.update_sum_count("mesh_texture_l1", mesh_texture_l1, 1.0)
                    else:
                        layer_total = layer_total + torch.zeros((), device=layer_total.device, dtype=layer_total.dtype)
                else:
                    layer_total = layer_total + torch.zeros((), device=layer_total.device, dtype=layer_total.dtype)

                # Render face mask to screen space for masking neck/back of head
                rendered_face_mask = None
                if real_face_mask_texture is not None and mesh_pred is not None:
                    mask_tex = real_face_mask_texture.expand(mesh_pred.shape[0], -1, -1, -1).contiguous()
                    rendered_face_mask_raw, face_mask_cov = render_mesh_texture_to_image(
                        model_ref, mesh_pred, mask_tex,
                        out_h=256, out_w=256, use_pred_depth=True, antialias=False,
                    )
                    if rendered_face_mask_raw is not None:
                        # Rendered mask has RGB channels: R=back-of-head, G=neck, B=~0
                        # Detect red (back-of-head) and green (neck) regions
                        r_ch = rendered_face_mask_raw[:, 0:1].detach()  # red channel
                        g_ch = rendered_face_mask_raw[:, 1:2].detach()  # green channel
                        cov1 = face_mask_cov[:, :1]
                        is_red = (r_ch > 0.3) & (g_ch < 0.3) & (cov1 > 0)    # back of head
                        is_green = (g_ch > 0.3) & (r_ch < 0.3) & (cov1 > 0)   # neck
                        # Expand red area by 10 pixels, green area by 1 pixel
                        red_expanded = F.max_pool2d(is_red.float(), kernel_size=21, stride=1, padding=10)
                        green_expanded = F.max_pool2d(is_green.float(), kernel_size=5, stride=1, padding=2)
                        # Mask = 1 (keep) everywhere except expanded red/green areas
                        rendered_face_mask = 1.0 - (red_expanded + green_expanded).clamp(0.0, 1.0)

                if basecolor_gt is not None and mesh_texture_pred is not None and train_cfg.basecolor_render_l1_lambda > 0:
                    # Render at 256 (internally 512 with 2x AA), compare at 256
                    bc_loss_h = basecolor_gt.shape[-2] // 2
                    bc_loss_w = basecolor_gt.shape[-1] // 2
                    render_pred, render_cov = render_mesh_texture_to_image(
                        model_ref,
                        mesh_pred,
                        mesh_texture_pred,
                        out_h=bc_loss_h,
                        out_w=bc_loss_w,
                        use_pred_depth=True,
                    )
                    if render_pred is not None:
                        basecolor_gt_ds = F.interpolate(basecolor_gt, size=(bc_loss_h, bc_loss_w), mode="bilinear", align_corners=False)
                        bc_weights = render_cov
                        if rendered_face_mask is not None:
                            fm = F.interpolate(rendered_face_mask, size=(bc_loss_h, bc_loss_w), mode="bilinear", align_corners=False)
                            bc_weights = bc_weights * fm
                        if basecolor_valid is not None:
                            bc_weights = bc_weights * basecolor_valid.view(-1, 1, 1, 1).to(device=render_pred.device, dtype=render_pred.dtype)
                        if bc_weights.sum().detach().item() > 0:
                            bc_l1 = compute_weighted_l1(render_pred.float(), basecolor_gt_ds.float(), bc_weights.expand_as(render_pred))
                            if torch.isfinite(bc_l1):
                                layer_total = layer_total + train_cfg.basecolor_render_l1_lambda * bc_l1
                                if layer_idx == (model_cfg.num_layers - 1):
                                    train_metrics.update_sum_count("basecolor_render_l1", bc_l1, 1.0)

                # Normal render loss — direct screen-space interpolation of vertex normals
                # Geo render loss (static UV-map texture rendered via predicted mesh UVs)
                if geo_gt is not None and mesh_pred is not None and train_cfg.geo_render_l1_lambda > 0:
                    # Render at 256 (internally 512 with 2x AA), compare at 256
                    geo_loss_h = geo_gt.shape[-2] // 2
                    geo_loss_w = geo_gt.shape[-1] // 2
                    geo_texture = model_ref.geo_feature_atlas.unsqueeze(0).expand(mesh_pred.shape[0], -1, -1, -1).contiguous()
                    geo_texture_rs = F.interpolate(
                        geo_texture,
                        size=(geo_loss_h, geo_loss_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    render_geo, render_geo_cov = render_mesh_texture_to_image(
                        model_ref, mesh_pred, geo_texture_rs,
                        out_h=geo_loss_h, out_w=geo_loss_w,
                        use_pred_depth=True,
                    )
                    if render_geo is not None:
                        geo_gt_ds = F.interpolate(geo_gt, size=(geo_loss_h, geo_loss_w), mode="bilinear", align_corners=False)
                        g_w = render_geo_cov
                        if rendered_face_mask is not None:
                            fm = F.interpolate(rendered_face_mask, size=(geo_loss_h, geo_loss_w), mode="bilinear", align_corners=False)
                            g_w = g_w * fm
                        if geo_valid is not None:
                            g_w = g_w * geo_valid.view(-1, 1, 1, 1).to(device=render_geo.device, dtype=render_geo.dtype)
                        if g_w.sum().detach().item() > 0:
                            g_l1 = compute_weighted_l1(render_geo.float(), geo_gt_ds.float(), g_w.expand_as(render_geo))
                            if torch.isfinite(g_l1):
                                layer_total = layer_total + train_cfg.geo_render_l1_lambda * g_l1
                                if layer_idx == (model_cfg.num_layers - 1):
                                    train_metrics.update_sum_count("geo_render_l1", g_l1, 1.0)

            loss += float(layer_w) * layer_total

        if model_cfg.model_type == "simdr" and batch_oob_count > 0:
            train_oob_sum += batch_oob_sum
            train_oob_count += float(batch_oob_count)

        # Test mode: save preview grids for real samples every N steps
        if (
            train_cfg.test_mode
            and rank == 0
            and real_sample_mask is not None
            and bool(real_sample_mask.any().item())
            and num_batches % train_cfg.test_preview_every == 0
        ):
            final_output = outputs_list[-1]
            test_mesh_pred = final_output["mesh"].float()
            if test_mesh_pred.ndim == 2:
                test_mesh_pred = test_mesh_pred.view(rgb.shape[0], -1, 6)
            test_tex_pred = final_output.get("mesh_texture", None)
            with torch.no_grad():
                _save_test_preview(
                    epoch=epoch,
                    step=num_batches,
                    batch=batch,
                    model_ref=model_ref,
                    mesh_pred=test_mesh_pred.detach(),
                    mesh_texture_pred=test_tex_pred.detach() if test_tex_pred is not None else None,
                    real_face_mask_texture=real_face_mask_texture,
                )

        # Check for NaN/inf before backward.
        # Important for DDP: if any rank sees non-finite loss, all ranks must follow
        # the same control flow and still execute a backward pass to keep reducer state valid.
        loss_finite_local = torch.isfinite(loss).to(device=device, dtype=torch.int32)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_finite_local, op=dist.ReduceOp.MIN)
        loss_finite_global = bool(loss_finite_local.item() > 0)

        if not loss_finite_global:
            if rank == 0:
                print(f"[NaN DETECTED] Loss is NaN/inf at batch {num_batches}")
                print(f"  Loss value: {loss.item()}")
                if "image_path" in batch:
                    print(f"  Sample paths: {batch['image_path'][:2]}")
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

        train_loss += loss.item()
        num_batches += 1

        if rank == 0 and isinstance(pbar, tqdm):
            postfix_dict = {"loss": f"{train_loss / num_batches:.4f}"}
            if train_metrics.has("train_3d"):
                postfix_dict["3D"] = f"{train_metrics.mean('train_3d'):.5f}"
            if train_metrics.has("train_2d"):
                train_2d_px = train_metrics.mean("train_2d") * 1024.0
                postfix_dict["px"] = f"{train_2d_px:.2f}"
            if train_metrics.has("mesh_texture_l1"):
                postfix_dict["tex"] = f"{train_metrics.mean('mesh_texture_l1'):.4f}"
            if train_metrics.has("basecolor_render_l1"):
                postfix_dict["bc"] = f"{train_metrics.mean('basecolor_render_l1'):.4f}"
            if train_metrics.has("geo_render_l1"):
                postfix_dict["geo"] = f"{train_metrics.mean('geo_render_l1'):.4f}"
            pbar.set_postfix(postfix_dict)

    if world_size > 1:
        loss_tensor = torch.tensor([train_loss, num_batches], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        train_loss = loss_tensor[0].item()
        num_batches = loss_tensor[1].item()

    train_metric_tensor = torch.tensor([
        train_metrics.get_sum("train_3d"),
        train_metrics.get_count("train_3d"),
        train_metrics.get_sum("train_2d"),
        train_metrics.get_count("train_2d"),
        train_metrics.get_sum("mesh_texture_l1"),
        train_metrics.get_count("mesh_texture_l1"),
        train_metrics.get_sum("basecolor_render_l1"),
        train_metrics.get_count("basecolor_render_l1"),
        train_metrics.get_sum("smooth_3d"),
        train_metrics.get_count("smooth_3d"),
        train_metrics.get_sum("smooth_2d"),
        train_metrics.get_count("smooth_2d"),
    ], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(train_metric_tensor, op=dist.ReduceOp.SUM)

    train_3d_sum, train_3d_count = train_metric_tensor[0].item(), train_metric_tensor[1].item()
    train_2d_sum, train_2d_count = train_metric_tensor[2].item(), train_metric_tensor[3].item()
    train_tex_sum, train_tex_count = train_metric_tensor[4].item(), train_metric_tensor[5].item()
    train_bc_sum, train_bc_count = train_metric_tensor[6].item(), train_metric_tensor[7].item()
    train_s3d_sum, train_s3d_count = train_metric_tensor[8].item(), train_metric_tensor[9].item()
    train_s2d_sum, train_s2d_count = train_metric_tensor[10].item(), train_metric_tensor[11].item()

    avg_train_3d_error = train_3d_sum / max(train_3d_count, 1e-6) if train_3d_count > 0 else None
    avg_train_2d_error = train_2d_sum / max(train_2d_count, 1e-6) if train_2d_count > 0 else None
    avg_train_2d_pixel_error = avg_train_2d_error * 1024.0 if avg_train_2d_error is not None else None
    avg_train_mesh_texture_l1 = train_tex_sum / max(train_tex_count, 1e-6) if train_tex_count > 0 else None
    avg_train_basecolor_render_l1 = train_bc_sum / max(train_bc_count, 1e-6) if train_bc_count > 0 else None
    avg_train_smooth_3d = train_s3d_sum / max(train_s3d_count, 1e-6) if train_s3d_count > 0 else None
    avg_train_smooth_2d = train_s2d_sum / max(train_s2d_count, 1e-6) if train_s2d_count > 0 else None

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
        "avg_train_basecolor_render_l1": avg_train_basecolor_render_l1,
        "avg_train_smooth_3d": avg_train_smooth_3d,
        "avg_train_smooth_2d": avg_train_smooth_2d,
        "simdr_oob_avg": simdr_oob_avg,
        "visualization_batch": visualization_batch,
    }


def validate_one_epoch(rank: int, world_size: int, prepared, built, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model = built["model"]
    criterion_val = built["criterion_val"]
    landmark_mask_weights_tensor = built["landmark_mask_weights_tensor"]
    mesh_mask_weights_tensor = built["mesh_mask_weights_tensor"]
    real_face_mask_texture = built.get("real_face_mask_texture", None)

    val_loader = prepared["val_loader"]
    device = torch.device(f"cuda:{rank}")

    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_metrics = MetricAccumulator()

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            basecolor_gt = batch["basecolor"].to(device, non_blocking=True) if "basecolor" in batch else None
            basecolor_valid = batch["basecolor_valid"].to(device, non_blocking=True) if "basecolor_valid" in batch else None

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

            lm_batch_weights = _build_batch_weights(batch, "landmark_weights", model_cfg.model_type, device)
            mesh_batch_weights = _build_batch_weights(batch, "mesh_weights", model_cfg.model_type, device)
            lm_found_mask = _build_batch_weights(batch, "landmark_found_mask", model_cfg.model_type, device)
            mesh_found_mask = _build_batch_weights(batch, "mesh_found_mask", model_cfg.model_type, device)
            lm_batch_weights = _merge_batch_weight_masks(lm_batch_weights, lm_found_mask)
            mesh_batch_weights = _merge_batch_weight_masks(mesh_batch_weights, mesh_found_mask)
            synthetic_sample_mask, real_sample_mask = _build_domain_masks(
                batch,
                prepared.get("synthetic_data_roots"),
                device,
            )

            if real_sample_mask is not None and bool(real_sample_mask.any().item()) and mesh_texture_valid is not None:
                mesh_texture_valid = mesh_texture_valid.clone()
                mesh_texture_valid[real_sample_mask] = 0.0

            if landmark_mask_weights_tensor is not None and "image_path" in batch:
                if lm_batch_weights is None:
                    lm_batch_weights = torch.ones_like(lm_gt)
                for i, path in enumerate(batch["image_path"]):
                    if "_flux" in os.path.basename(path):
                        lm_batch_weights[i] *= landmark_mask_weights_tensor.view(-1, 6) if model_cfg.model_type == "simdr" else landmark_mask_weights_tensor

            if mesh_mask_weights_tensor is not None and "image_path" in batch:
                if mesh_batch_weights is None:
                    mesh_batch_weights = torch.ones_like(mesh_gt)
                for i, path in enumerate(batch["image_path"]):
                    if "_flux" in os.path.basename(path):
                        mesh_batch_weights[i] *= mesh_mask_weights_tensor.view(-1, 6) if model_cfg.model_type == "simdr" else mesh_mask_weights_tensor

            if model_cfg.model_type != "simdr":
                lm_pred = lm_pred.reshape(rgb.shape[0], -1)
                mesh_pred = mesh_pred.reshape(rgb.shape[0], -1)
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
                        if torch.isfinite(mesh_texture_loss):
                            val_metrics.update_sum_count("val_mesh_texture_l1", masked.sum(), denom)
                            loss = loss + train_cfg.mesh_texture_l1_lambda * mesh_texture_loss
                else:
                    mesh_texture_loss = mesh_texture_l1_raw.mean()
                    val_metrics.update_sum_count("val_mesh_texture_l1", mesh_texture_l1_raw.sum(), mesh_texture_l1_raw.new_tensor(float(mesh_texture_l1_raw.numel())))
                    loss = loss + train_cfg.mesh_texture_l1_lambda * mesh_texture_loss

            if mesh_texture_pred is not None and basecolor_gt is not None:
                model_ref = _unwrap_model(model)
                # Render at 256 (internally 512 with 2x AA), compare at 256
                val_bc_h = basecolor_gt.shape[-2] // 2
                val_bc_w = basecolor_gt.shape[-1] // 2
                render_pred, render_cov = render_mesh_texture_to_image(
                    model_ref,
                    mesh_pred,
                    mesh_texture_pred,
                    out_h=val_bc_h,
                    out_w=val_bc_w,
                    use_pred_depth=True,
                )
                if render_pred is not None:
                    basecolor_gt_ds = F.interpolate(basecolor_gt, size=(val_bc_h, val_bc_w), mode="bilinear", align_corners=False)
                    bc_weights = render_cov
                    if real_face_mask_texture is not None:
                        mask_tex = real_face_mask_texture.expand(mesh_pred.shape[0], -1, -1, -1).contiguous()
                        rendered_mask_raw, _ = render_mesh_texture_to_image(
                            model_ref, mesh_pred, mask_tex,
                            out_h=val_bc_h, out_w=val_bc_w, use_pred_depth=True,
                        )
                        if rendered_mask_raw is not None:
                            fm = rendered_mask_raw[:, :1].detach()
                            if fm.shape[-2] != val_bc_h or fm.shape[-1] != val_bc_w:
                                fm = F.interpolate(fm, size=(val_bc_h, val_bc_w), mode="bilinear", align_corners=False)
                            bc_weights = bc_weights * fm
                    if basecolor_valid is not None:
                        bc_weights = bc_weights * basecolor_valid.view(-1, 1, 1, 1).to(device=render_pred.device, dtype=render_pred.dtype)
                    denom = bc_weights.sum()
                    if denom.detach().item() > 0:
                        bc_weights_3 = bc_weights.expand_as(render_pred)
                        bc_l1 = compute_weighted_l1(render_pred.float(), basecolor_gt_ds.float(), bc_weights_3)
                        if torch.isfinite(bc_l1):
                            val_metrics.update_sum_count("val_basecolor_render_l1", bc_l1, 1.0)
                            loss = loss + train_cfg.basecolor_render_l1_lambda * bc_l1

            def _acc(pred_m, gt_m, w_m):
                l1 = (pred_m - gt_m).abs()
                if w_m is None:
                    w_m = torch.ones_like(l1)
                val_metrics.update_sum_count("val_3d", (l1[..., :3] * w_m[..., :3]).sum(), w_m[..., :3].sum())
                val_metrics.update_sum_count("val_2d", (l1[..., 3:5] * w_m[..., 3:5]).sum(), w_m[..., 3:5].sum())

            if model_cfg.model_type == "simdr":
                _acc(lm_pred, lm_gt, lm_batch_weights)
                _acc(mesh_pred, mesh_gt, mesh_batch_weights)
            else:
                _acc(lm_pred.view(rgb.shape[0], -1, 6), lm_gt.view(rgb.shape[0], -1, 6), lm_batch_weights.view(rgb.shape[0], -1, 6) if lm_batch_weights is not None else None)
                _acc(mesh_pred.view(rgb.shape[0], -1, 6), mesh_gt.view(rgb.shape[0], -1, 6), mesh_batch_weights.view(rgb.shape[0], -1, 6) if mesh_batch_weights is not None else None)

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
        val_metrics.get_sum("val_basecolor_render_l1"),
        val_metrics.get_count("val_basecolor_render_l1"),
    ], device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    val_3d_sum, val_3d_count = metric_tensor[0].item(), metric_tensor[1].item()
    val_2d_sum, val_2d_count = metric_tensor[2].item(), metric_tensor[3].item()
    tex_sum, tex_count = metric_tensor[4].item(), metric_tensor[5].item()
    bc_sum, bc_count = metric_tensor[6].item(), metric_tensor[7].item()

    avg_val_loss = val_loss / max(val_batches, 1)
    avg_val_3d_error = val_3d_sum / max(val_3d_count, 1e-6) if val_3d_count > 0 else None
    avg_val_2d_error = val_2d_sum / max(val_2d_count, 1e-6) if val_2d_count > 0 else None
    avg_val_2d_pixel_error = avg_val_2d_error * 1024.0 if avg_val_2d_error is not None else None
    avg_val_mesh_texture_l1 = tex_sum / max(tex_count, 1e-6) if tex_count > 0 else None
    avg_val_basecolor_render_l1 = bc_sum / max(bc_count, 1e-6) if bc_count > 0 else None

    return {
        "avg_val_loss": avg_val_loss,
        "avg_val_3d_error": avg_val_3d_error,
        "avg_val_2d_pixel_error": avg_val_2d_pixel_error,
        "avg_val_mesh_texture_l1": avg_val_mesh_texture_l1,
        "avg_val_basecolor_render_l1": avg_val_basecolor_render_l1,
    }


def log_and_checkpoint(epoch: int, rank: int, built, prepared, train_out, val_out, writer, landmark_topology, mesh_topology):
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
    if val_out["avg_val_basecolor_render_l1"] is not None:
        metric_msg += f" | ValBase: {val_out['avg_val_basecolor_render_l1']:.4f}"
    print(f"Train Loss: {train_out['avg_train_loss']:.6f} | Val Loss: {val_out['avg_val_loss']:.6f}{metric_msg}")

    if writer:
        writer.add_scalar("Loss/Train", train_out["avg_train_loss"], epoch)
        writer.add_scalar("Loss/Val", val_out["avg_val_loss"], epoch)
        if train_out["avg_train_mesh_texture_l1"] is not None:
            writer.add_scalar("Loss/MeshTexture_Train_L1", train_out["avg_train_mesh_texture_l1"], epoch)
        if val_out["avg_val_mesh_texture_l1"] is not None:
            writer.add_scalar("Loss/MeshTexture_Val_L1", val_out["avg_val_mesh_texture_l1"], epoch)
        if train_out.get("avg_train_basecolor_render_l1") is not None:
            writer.add_scalar("Loss/BasecolorRender_Train_L1", train_out["avg_train_basecolor_render_l1"], epoch)
        if val_out.get("avg_val_basecolor_render_l1") is not None:
            writer.add_scalar("Loss/BasecolorRender_Val_L1", val_out["avg_val_basecolor_render_l1"], epoch)
        if train_out.get("avg_train_smooth_3d") is not None:
            writer.add_scalar("Loss/Smooth3D_Train", train_out["avg_train_smooth_3d"], epoch)
        if train_out.get("avg_train_smooth_2d") is not None:
            writer.add_scalar("Loss/Smooth2D_Train", train_out["avg_train_smooth_2d"], epoch)
        if val_out["avg_val_3d_error"] is not None:
            writer.add_scalar("Metrics/Val_3D_Error", val_out["avg_val_3d_error"], epoch)
        if val_out["avg_val_2d_pixel_error"] is not None:
            writer.add_scalar("Metrics/Val_2D_Pixel_Error", val_out["avg_val_2d_pixel_error"], epoch)
            writer.add_scalar("Metrics/Val_Scaled_Error", val_out["avg_val_2d_pixel_error"], epoch)
        writer.add_scalar("LR/backbone", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("LR/head", optimizer.param_groups[2]["lr"], epoch)

    # Always save latest checkpoint (with full training state)
    filename = "artifacts/checkpoints/best_geometry_transformer_dim6.pth"
    save_dict = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": built["scheduler"].state_dict(),
        "scaler_state_dict": built["scaler"].state_dict(),
        "val_loss": val_out["avg_val_loss"],
        "best_loss": built["best_loss"],
        "output_dim": 6,
    }
    if val_out["avg_val_loss"] < built["best_loss"]:
        built["best_loss"] = val_out["avg_val_loss"]
        save_dict["best_loss"] = built["best_loss"]
        print(f"New best model (Val Loss: {val_out['avg_val_loss']:.6f})")
    save_dir = os.path.dirname(filename)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(save_dict, filename)
    print(f"Saved checkpoint to {filename} (epoch {epoch + 1}, Val Loss: {val_out['avg_val_loss']:.6f})")

    if train_out["visualization_batch"] is not None and landmark_topology and mesh_topology:
        save_geometry_visualizations(
            _unwrap_model(model),
            train_out["visualization_batch"],
            epoch,
            torch.device("cuda:0"),
            "artifacts/training_samples",
            landmark_topology,
            mesh_topology,
            output_dim=6,
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
        writer = SummaryWriter(f"artifacts/runs/{run_name}")
        landmark_topology = load_landmark_topology()
        mesh_topology = load_mesh_topology()

    prepared = prepare_data(rank, world_size, data_cfg)
    built = build_model_and_optim(rank, world_size, model_cfg, train_cfg, prepared)

    # Cache the full (real+synthetic) loaders so we can switch back after epoch N
    prepared["_all_train_loader"] = prepared["train_loader"]
    prepared["_all_train_sampler"] = prepared["train_sampler"]

    for epoch in range(built["start_epoch"], train_cfg.epochs):
        # Swap loader: use synthetic-only before real data starts (use min of geo/basecolor start)
        syn_loader = prepared.get("synthetic_train_loader")
        real_data_start = min(int(train_cfg.real_data_geo_start_epoch), int(train_cfg.real_data_basecolor_start_epoch))
        epochs_since_load = epoch - built["start_epoch"]
        if syn_loader is not None and real_data_start > 0:
            if epochs_since_load < real_data_start:
                if epochs_since_load == 0 and rank == 0:
                    print(f"[Info] Using synthetic-only data for first {real_data_start} epochs after resume")
                prepared["train_loader"] = syn_loader
                prepared["train_sampler"] = prepared["synthetic_train_sampler"]
            else:
                if epochs_since_load == real_data_start and rank == 0:
                    print(f"[Info] Epoch {epoch + 1}: switching to full dataset (real + synthetic)")
                prepared["train_loader"] = prepared["_all_train_loader"]
                prepared["train_sampler"] = prepared["_all_train_sampler"]

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{train_cfg.epochs}")
        train_out = train_one_epoch(epoch, rank, world_size, prepared, built, data_cfg, model_cfg, train_cfg)
        val_out = validate_one_epoch(rank, world_size, prepared, built, model_cfg, train_cfg)
        log_and_checkpoint(epoch, rank, built, prepared, train_out, val_out, writer, landmark_topology, mesh_topology)
        built["scheduler"].step()

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
