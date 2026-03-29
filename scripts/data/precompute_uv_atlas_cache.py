from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
for _candidate in (_THIS_FILE.parent, *_THIS_FILE.parents):
    if (_candidate / "data_utils").exists():
        _PROJECT_ROOT = _candidate
        break
else:
    _PROJECT_ROOT = _THIS_FILE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import argparse
import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import freeze_support
from typing import Any

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from align_5pt_helper import Align5PtHelper
from data_utils.alignment_io import compute_alignment_transform, warp_rgb_image
from data_utils.image_io import load_rgb_image
from data_utils.sample_index import GeometrySampleRecord, collect_geometry_sample_records
from data_utils.texture_pack import TexturePackHelper
from data_utils.uv_atlas_io import (
    build_partial_uv_atlas_inputs,
    load_filtered_mesh_uv_and_faces,
    save_float01_png,
    save_mask_png,
)
from dense2geometry import Dense2Geometry, DenseStageConfig
from train_launcher_common import data_roots_from_folders, resolve_platform_key, resolve_platform_value


DATA_FOLDERS_BY_PLATFORM = {
    "linux": [
        "/hy-tmp/CapturedFrames_final1_processed",
        "/hy-tmp/CapturedFrames_final2_processed",
        "/hy-tmp/CapturedFrames_final3_processed",
        "/hy-tmp/CapturedFrames_final4_processed",
        "/hy-tmp/CapturedFrames_final5_processed",
        "/hy-tmp/CapturedFrames_final6_processed",
        "/hy-tmp/CapturedFrames_final7_processed",
        "/hy-tmp/CapturedFrames_final8_processed",
        "/hy-tmp/CapturedFrames_final9_processed",
    ],
    "windows": [
        "G:/CapturedFrames_final1_processed",
        "G:/CapturedFrames_final2_processed",
        "G:/CapturedFrames_final3_processed",
        "G:/CapturedFrames_final4_processed",
        "G:/CapturedFrames_final5_processed",
        "G:/CapturedFrames_final6_processed",
        "G:/CapturedFrames_final7_processed",
        "G:/CapturedFrames_final8_processed",
        "G:/CapturedFrames_final9_processed",
    ],
}

TEXTURE_ROOT_BY_PLATFORM = {
    "linux": "/hy-tmp/textures",
    "windows": "G:/textures",
}

REQUIRED_CACHE_FILES = (
    "src_color_uv.png",
    "pred_basecolor_uv.png",
    "pred_geo_uv.png",
    "pred_geometry_normal_uv.png",
    "pred_detail_normal_uv.png",
    "uv_valid_mask.png",
    "gt_basecolor_atlas.png",
    "gt_detail_normal_atlas.png",
    "metadata.json",
)


@dataclass(frozen=True)
class UVAtlasPrecomputeRecord:
    split: str
    data_root: str
    sample_id: str
    color_path: str
    landmark_path: str
    sample_dir: str


def _parse_bool_arg(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _root_tag(data_root: str) -> str:
    base = os.path.basename(os.path.normpath(data_root)) or "root"
    digest = hashlib.sha1(os.path.abspath(data_root).encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def _sample_dir(output_root: str, split: str, data_root: str, sample_id: str) -> str:
    return os.path.join(output_root, split, _root_tag(data_root), str(sample_id))


def _cache_complete(sample_dir: str) -> bool:
    return all(os.path.exists(os.path.join(sample_dir, name)) for name in REQUIRED_CACHE_FILES)


def _collect_precompute_records(
    data_roots: list[str],
    output_root: str,
    split: str,
    train_ratio: float,
    overwrite: bool,
) -> list[UVAtlasPrecomputeRecord]:
    splits = [str(split)] if str(split) in {"train", "val"} else ["train", "val"]
    out: list[UVAtlasPrecomputeRecord] = []
    for split_name in splits:
        records = collect_geometry_sample_records(data_roots=data_roots, split=split_name, train_ratio=train_ratio)
        for record in records:
            sample_dir = _sample_dir(output_root, split_name, record.data_root, record.sample_id)
            if not overwrite and _cache_complete(sample_dir):
                continue
            out.append(
                UVAtlasPrecomputeRecord(
                    split=split_name,
                    data_root=record.data_root,
                    sample_id=record.sample_id,
                    color_path=record.color_path,
                    landmark_path=record.landmark_path,
                    sample_dir=sample_dir,
                )
            )
    return out


def _set_cv_cuda_device(device_id: int) -> None:
    try:
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "setDevice"):
            cv2.cuda.setDevice(int(device_id))
    except Exception:
        pass


def _load_matching_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
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


_WRITER_TEXTURE_PACKERS: dict[str, TexturePackHelper] = {}


def worker_init_fn(_worker_id):
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _get_writer_texture_packer(texture_root: str) -> TexturePackHelper:
    key = str(texture_root or "")
    packer = _WRITER_TEXTURE_PACKERS.get(key)
    if packer is None:
        packer = TexturePackHelper(
            texture_root=key,
            texture_png_cache_max_items=256,
            combined_texture_cache_max_items=64,
        )
        _WRITER_TEXTURE_PACKERS[key] = packer
    return packer


class UVAtlasPrecomputeDataset(Dataset):
    def __init__(
        self,
        records: list[UVAtlasPrecomputeRecord],
        image_size: int,
        texture_root: str,
    ):
        self.records = list(records)
        self.image_size = int(image_size)
        self.texture_root = str(texture_root or "")
        self.align_helper = Align5PtHelper(
            image_size=self.image_size,
            scale_jitter=0.0,
            translate_jitter=0.0,
            lm_jitter=0.0,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        record = self.records[idx]
        try:
            image_np = load_rgb_image(record.color_path)
            if image_np is None:
                raise FileNotFoundError(record.color_path)

            lm6, detection_source, transform = compute_alignment_transform(
                image_np=image_np,
                landmark_path=record.landmark_path,
                align_helper=self.align_helper,
                split="val",
            )
            del lm6
            aligned_rgb = warp_rgb_image(image_np, transform, self.image_size).astype(np.float32) / 255.0

            return {
                "rgb": torch.from_numpy(aligned_rgb).permute(2, 0, 1).float(),
                "sample_dir": str(record.sample_dir),
                "sample_id": str(record.sample_id),
                "data_root": str(record.data_root),
                "image_path": str(record.color_path),
                "split": str(record.split),
                "detection_source": str(detection_source),
                "texture_root": self.texture_root,
            }
        except Exception as exc:
            print(f"[UVAtlasPrecomputeDataset] Skip {record.sample_id}: {exc}")
            return None


def _collate_precompute(batch: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return {
        "rgb": torch.stack([item["rgb"] for item in batch], dim=0),
        "sample_dir": [item["sample_dir"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "data_root": [item["data_root"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "split": [item["split"] for item in batch],
        "detection_source": [item["detection_source"] for item in batch],
        "texture_root": [item["texture_root"] for item in batch],
    }


def create_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_roots", nargs="*", default=None)
    parser.add_argument("--texture_root", type=str, default="")
    parser.add_argument("--output_root", type=str, default="artifacts/uv_atlas_cache")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val"])
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--atlas_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--writer_threads", type=int, default=8)
    parser.add_argument("--max_pending_writes", type=int, default=32)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--overwrite", type=_parse_bool_arg, nargs="?", const=True, default=False)
    parser.add_argument("--master_amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument("--load_model", type=str, default="artifacts/checkpoints/best_dense2geometry.pth")
    parser.add_argument("--load_dense_model", type=str, default="")

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

    parser.add_argument("--dense_d_model", type=int, default=256)
    parser.add_argument("--dense_nhead", type=int, default=8)
    parser.add_argument("--dense_num_layers", type=int, default=4)
    parser.add_argument("--dense_transformer_map_size", type=int, default=32)
    parser.add_argument("--dense_backbone_weights", type=str, default="imagenet", choices=["imagenet", "dinov3", "none"])
    parser.add_argument("--dense_decoder_type", type=str, default="multitask", choices=["multitask", "shared"])
    return parser


def _build_dense2geometry_model(args, device: torch.device) -> Dense2Geometry:
    dense_cfg = DenseStageConfig(
        d_model=int(args.dense_d_model),
        nhead=int(args.dense_nhead),
        num_layers=int(args.dense_num_layers),
        output_size=int(args.image_size),
        transformer_map_size=int(args.dense_transformer_map_size),
        backbone_weights=str(args.dense_backbone_weights),
        decoder_type=str(args.dense_decoder_type),
    )
    dense_override_after_load = bool(args.load_model and args.load_dense_model)
    model = Dense2Geometry(
        d_model=int(args.d_model),
        nhead=int(args.nhead),
        num_layers=int(args.num_layers),
        dense_stage_cfg=dense_cfg,
        dense_checkpoint="" if dense_override_after_load else str(args.load_dense_model or ""),
        freeze_dense_stage=True,
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
        model_dir="assets/topology",
    ).to(device)

    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        skipped_keys, load_notes = _load_matching_state_dict(model, state_dict)
        print(f"Loaded Dense2Geometry checkpoint: {args.load_model}")
        if skipped_keys:
            print(f"[Warn] Skipped incompatible checkpoint keys: {skipped_keys[:10]}")
        if load_notes:
            print(f"[Warn] Missing/unexpected checkpoint keys: {load_notes[:10]}")

    if dense_override_after_load:
        model.load_dense_stage(str(args.load_dense_model))
        print(f"Reloaded dense stage from {args.load_dense_model}")

    model.eval()
    return model


def _resolve_num_workers(requested_workers: int, world_size: int) -> int:
    requested_workers = int(requested_workers)
    if requested_workers > 0:
        return requested_workers
    cpu_count = int(os.cpu_count() or 8)
    auto_workers = cpu_count // max(int(world_size) * 2, 1)
    return int(max(2, min(12, auto_workers)))


def _to_cpu_cache_payloads(batch, partials: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    src_color_uv = partials["src_color_uv"].detach().cpu().numpy()
    pred_basecolor_uv = partials["pred_basecolor_uv"].detach().cpu().numpy()
    pred_geo_uv = partials["pred_geo_uv"].detach().cpu().numpy()
    pred_geometry_normal_uv = partials["pred_geometry_normal_uv"].detach().cpu().numpy()
    pred_detail_normal_uv = partials["pred_detail_normal_uv"].detach().cpu().numpy()
    uv_valid_mask = partials["uv_valid_mask"].detach().cpu().numpy()

    payloads: list[dict[str, Any]] = []
    batch_size = int(src_color_uv.shape[0])
    for index in range(batch_size):
        sample_dir = batch["sample_dir"][index]
        payloads.append({
            "sample_dir": sample_dir,
            "texture_root": batch["texture_root"][index],
            "sample_id": batch["sample_id"][index],
            "data_root": batch["data_root"][index],
            "src_color_uv": np.transpose(src_color_uv[index], (1, 2, 0)),
            "pred_basecolor_uv": np.transpose(pred_basecolor_uv[index], (1, 2, 0)),
            "pred_geo_uv": np.transpose(pred_geo_uv[index], (1, 2, 0)),
            "pred_geometry_normal_uv": np.transpose(pred_geometry_normal_uv[index], (1, 2, 0)),
            "pred_detail_normal_uv": np.transpose(pred_detail_normal_uv[index], (1, 2, 0)),
            "uv_valid_mask": uv_valid_mask[index, 0],
            "metadata": {
                "sample_id": batch["sample_id"][index],
                "data_root": batch["data_root"][index],
                "image_path": batch["image_path"][index],
                "split": batch["split"][index],
                "sample_key": os.path.join(os.path.basename(os.path.dirname(sample_dir)), os.path.basename(sample_dir)).replace("\\", "/"),
                "detection_source": batch["detection_source"][index],
            },
        })
    return payloads


def _wait_for_pending_writes(pending_writes: list, max_pending: int) -> list:
    if len(pending_writes) < int(max_pending):
        return pending_writes
    done, not_done = wait(pending_writes, return_when=FIRST_COMPLETED)
    for future in done:
        future.result()
    return list(not_done)


def _write_sample_cache(payload: dict[str, Any]) -> None:
    sample_dir = str(payload["sample_dir"])
    os.makedirs(sample_dir, exist_ok=True)

    texture_packer = _get_writer_texture_packer(str(payload.get("texture_root", "")))
    gt_basecolor_atlas = texture_packer.load_mesh_texture_map(str(payload["data_root"]), str(payload["sample_id"]))
    gt_detail_normal_atlas = texture_packer.load_mesh_detail_normal_map(str(payload["data_root"]), str(payload["sample_id"]))
    if gt_basecolor_atlas is None:
        raise RuntimeError(f"Missing basecolor atlas GT for {payload['sample_id']}")
    if gt_detail_normal_atlas is None:
        raise RuntimeError(f"Missing detail normal atlas GT for {payload['sample_id']}")

    save_float01_png(os.path.join(sample_dir, "src_color_uv.png"), payload["src_color_uv"], bit_depth=8)
    save_float01_png(os.path.join(sample_dir, "pred_basecolor_uv.png"), payload["pred_basecolor_uv"], bit_depth=8)
    save_float01_png(os.path.join(sample_dir, "pred_geo_uv.png"), payload["pred_geo_uv"], bit_depth=16)
    save_float01_png(os.path.join(sample_dir, "pred_geometry_normal_uv.png"), payload["pred_geometry_normal_uv"], bit_depth=8)
    save_float01_png(os.path.join(sample_dir, "pred_detail_normal_uv.png"), payload["pred_detail_normal_uv"], bit_depth=8)
    save_mask_png(os.path.join(sample_dir, "uv_valid_mask.png"), payload["uv_valid_mask"])
    save_float01_png(os.path.join(sample_dir, "gt_basecolor_atlas.png"), gt_basecolor_atlas, bit_depth=8)
    save_float01_png(os.path.join(sample_dir, "gt_detail_normal_atlas.png"), gt_detail_normal_atlas, bit_depth=8)

    metadata_path = os.path.join(sample_dir, "metadata.json")
    metadata_tmp = os.path.join(sample_dir, f".metadata.json.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}")
    try:
        with open(metadata_tmp, "w", encoding="utf-8") as f:
            json.dump(payload["metadata"], f, indent=2)
        os.replace(metadata_tmp, metadata_path)
    finally:
        if os.path.exists(metadata_tmp):
            try:
                os.remove(metadata_tmp)
            except OSError:
                pass


def _worker(rank: int, world_size: int, args, records: list[UVAtlasPrecomputeRecord]) -> None:
    if torch.cuda.is_available():
        device_id = rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(device_id)
        _set_cv_cuda_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        raise RuntimeError("CUDA is required for UV atlas cache precompute")

    local_records = records[rank::world_size]
    if not local_records:
        print(f"[UVAtlasCache][rank={rank}] No samples assigned")
        return

    dataset = UVAtlasPrecomputeDataset(
        records=local_records,
        image_size=int(args.image_size),
        texture_root=str(args.texture_root or ""),
    )
    resolved_num_workers = _resolve_num_workers(int(args.num_workers), world_size)
    loader_kwargs = {
        "batch_size": int(max(1, args.batch_size)),
        "shuffle": False,
        "num_workers": int(max(0, resolved_num_workers)),
        "pin_memory": True,
        "collate_fn": _collate_precompute,
        "worker_init_fn": worker_init_fn,
        "drop_last": False,
    }
    if int(resolved_num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(max(2, args.prefetch_factor))
    loader = DataLoader(dataset, **loader_kwargs)

    if rank == 0:
        print(
            f"[UVAtlasCache] world_size={world_size} batch_size={int(args.batch_size)} "
            f"num_workers/gpu={resolved_num_workers} prefetch_factor={loader_kwargs.get('prefetch_factor', 0)} "
            f"writer_threads/gpu={int(max(1, args.writer_threads))}"
        )

    torch.backends.cudnn.benchmark = True
    model = _build_dense2geometry_model(args, device=device)
    template_uv_np, template_faces_np = load_filtered_mesh_uv_and_faces(model_dir="assets/topology")
    template_uv_t = torch.from_numpy(template_uv_np).to(device=device, dtype=torch.float32)
    template_faces_t = torch.from_numpy(template_faces_np).to(device=device, dtype=torch.int32)
    amp_dtype = torch.float16 if str(args.master_amp_dtype).lower() == "fp16" else torch.bfloat16

    progress = tqdm(loader, desc=f"UV cache rank {rank}") if rank == 0 else loader
    processed = 0
    pending_writes = []
    writer_threads = int(max(1, args.writer_threads))
    max_pending_writes = int(max(writer_threads * 2, args.max_pending_writes))
    with ThreadPoolExecutor(max_workers=writer_threads) as write_pool:
        with torch.inference_mode():
            for batch in progress:
                if batch is None:
                    continue
                rgb = batch["rgb"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    outputs = model(rgb)
                    partials = build_partial_uv_atlas_inputs(
                        rgb=rgb,
                        pred_basecolor=outputs["pred_basecolor"],
                        pred_geo=outputs["pred_geo"],
                        pred_geometry_normal=outputs["pred_geometry_normal"],
                        pred_detail_normal=outputs["pred_detail_normal"],
                        mesh_uv=outputs["mesh"][..., 3:5].clamp(0.0, 1.0),
                        match_mask=outputs["match_mask"].float(),
                        template_uv=template_uv_t,
                        template_faces=template_faces_t,
                        atlas_size=int(args.atlas_size),
                        device=device,
                    )

                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()
                payloads = _to_cpu_cache_payloads(batch, partials)
                del outputs, partials, rgb

                for payload in payloads:
                    pending_writes.append(write_pool.submit(_write_sample_cache, payload))
                    pending_writes = _wait_for_pending_writes(pending_writes, max_pending_writes)

                processed += len(payloads)
                if rank == 0 and hasattr(progress, "set_postfix"):
                    progress.set_postfix({"saved": processed, "pending": len(pending_writes)})

        for future in pending_writes:
            future.result()

    print(f"[UVAtlasCache][rank={rank}] Saved {processed} samples")


def main(platform_key: str | None = None) -> None:
    freeze_support()
    resolved_platform = resolve_platform_key(platform_key)
    parser = create_arg_parser(f"Precompute UV Atlas Cache ({resolved_platform.capitalize()})")
    args = parser.parse_args()

    if not args.data_roots:
        args.data_roots = data_roots_from_folders(resolve_platform_value(DATA_FOLDERS_BY_PLATFORM, resolved_platform))
    if not args.texture_root:
        args.texture_root = str(resolve_platform_value(TEXTURE_ROOT_BY_PLATFORM, resolved_platform))

    records = _collect_precompute_records(
        data_roots=list(args.data_roots),
        output_root=str(args.output_root),
        split=str(args.split),
        train_ratio=float(args.train_ratio),
        overwrite=bool(args.overwrite),
    )
    print(f"UV atlas cache output: {args.output_root}")
    print(f"Texture root: {args.texture_root}")
    print(f"Dense2Geometry checkpoint: {args.load_model}")
    print(f"Queued samples: {len(records)}")
    if not records:
        print("No samples to process")
        return

    available_gpus = int(torch.cuda.device_count())
    if available_gpus <= 0:
        raise RuntimeError("No CUDA devices found")
    world_size = min(int(max(1, args.num_gpus)), available_gpus, len(records))
    print(f"Launching UV atlas cache precompute on {world_size} GPU(s)")

    if world_size <= 1:
        _worker(0, 1, args, records)
    else:
        mp.spawn(_worker, args=(world_size, args, records), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
