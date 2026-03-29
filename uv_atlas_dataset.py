from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from data_utils.uv_atlas_io import is_png_intact, load_float01_png, load_mask_png


UV_ATLAS_RGB_INPUT_KEYS = (
    "src_color_uv",
    "pred_basecolor_uv",
    "pred_geo_uv",
    "pred_geometry_normal_uv",
    "pred_detail_normal_uv",
)
UV_ATLAS_MASK_INPUT_KEYS = ("uv_valid_mask",)
UV_ATLAS_INPUT_KEYS = UV_ATLAS_RGB_INPUT_KEYS + UV_ATLAS_MASK_INPUT_KEYS
UV_ATLAS_REQUIRED_PNGS = (
    "src_color_uv.png",
    "pred_basecolor_uv.png",
    "pred_geo_uv.png",
    "pred_geometry_normal_uv.png",
    "pred_detail_normal_uv.png",
    "uv_valid_mask.png",
    "gt_basecolor_atlas.png",
    "gt_detail_normal_atlas.png",
)


@dataclass(frozen=True)
class UVAtlasSampleRecord:
    split: str
    sample_dir: str
    sample_key: str
    metadata_path: str


def _should_log_dataset_init() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank()) == 0
    except Exception:
        pass
    return True


def _validate_cache_record(record: UVAtlasSampleRecord) -> bool:
    if not os.path.exists(record.metadata_path):
        return False
    try:
        with open(record.metadata_path, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False

    sample_dir = record.sample_dir
    for file_name in UV_ATLAS_REQUIRED_PNGS:
        path = os.path.join(sample_dir, file_name)
        if not is_png_intact(path):
            return False
    return True


def collect_uv_atlas_sample_records(cache_root: str, split: str = "train") -> list[UVAtlasSampleRecord]:
    split_dir = os.path.join(cache_root, str(split))
    if not os.path.isdir(split_dir):
        return []

    pattern = os.path.join(split_dir, "**", "metadata.json")
    records: list[UVAtlasSampleRecord] = []
    for metadata_path in sorted(glob.glob(pattern, recursive=True)):
        sample_dir = os.path.dirname(metadata_path)
        sample_key = os.path.relpath(sample_dir, split_dir).replace("\\", "/")
        records.append(
            UVAtlasSampleRecord(
                split=str(split),
                sample_dir=str(sample_dir),
                sample_key=str(sample_key),
                metadata_path=str(metadata_path),
            )
        )
    return records


class UVAtlasDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str = "train",
        max_samples: int = 0,
        validate_samples: bool = True,
    ):
        self.cache_root = str(cache_root)
        self.split = str(split)
        self.validate_samples = bool(validate_samples)
        self.records = collect_uv_atlas_sample_records(self.cache_root, split=self.split)

        if self.validate_samples and self.records:
            valid_records: list[UVAtlasSampleRecord] = []
            invalid_count = 0
            for record in self.records:
                if _validate_cache_record(record):
                    valid_records.append(record)
                else:
                    invalid_count += 1
            self.records = valid_records
            if _should_log_dataset_init() and invalid_count > 0:
                print(f"[{self.__class__.__name__}] dropped {invalid_count} invalid cache samples from split={self.split}")

        if int(max_samples) > 0:
            self.records = self.records[: int(max_samples)]

        if _should_log_dataset_init():
            print(f"[{self.__class__.__name__}] split={self.split} total={len(self.records)} cache_root={self.cache_root}")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_rgb_map(path: str) -> torch.Tensor:
        image = load_float01_png(path)
        if image is None:
            raise FileNotFoundError(path)
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = image.repeat(3, axis=2)
        return torch.from_numpy(image.astype("float32", copy=False)).permute(2, 0, 1).float()

    @staticmethod
    def _load_mask_map(path: str) -> torch.Tensor:
        mask = load_mask_png(path)
        if mask is None:
            raise FileNotFoundError(path)
        return torch.from_numpy(mask.astype("float32", copy=False)).unsqueeze(0).float()

    def _load_record(self, record: UVAtlasSampleRecord) -> dict[str, Any]:
        with open(record.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        sample_dir = record.sample_dir
        src_color_uv = self._load_rgb_map(os.path.join(sample_dir, "src_color_uv.png"))
        pred_basecolor_uv = self._load_rgb_map(os.path.join(sample_dir, "pred_basecolor_uv.png"))
        pred_geo_uv = self._load_rgb_map(os.path.join(sample_dir, "pred_geo_uv.png"))
        pred_geometry_normal_uv = self._load_rgb_map(os.path.join(sample_dir, "pred_geometry_normal_uv.png"))
        pred_detail_normal_uv = self._load_rgb_map(os.path.join(sample_dir, "pred_detail_normal_uv.png"))
        uv_valid_mask = self._load_mask_map(os.path.join(sample_dir, "uv_valid_mask.png"))
        basecolor_atlas = self._load_rgb_map(os.path.join(sample_dir, "gt_basecolor_atlas.png"))
        detail_normal_atlas = self._load_rgb_map(os.path.join(sample_dir, "gt_detail_normal_atlas.png"))

        uv_input = torch.cat(
            [
                src_color_uv,
                pred_basecolor_uv,
                pred_geo_uv,
                pred_geometry_normal_uv,
                pred_detail_normal_uv,
                uv_valid_mask,
            ],
            dim=0,
        )

        return {
            "uv_input": uv_input,
            "src_color_uv": src_color_uv,
            "pred_basecolor_uv": pred_basecolor_uv,
            "pred_geo_uv": pred_geo_uv,
            "pred_geometry_normal_uv": pred_geometry_normal_uv,
            "pred_detail_normal_uv": pred_detail_normal_uv,
            "uv_valid_mask": uv_valid_mask,
            "basecolor_atlas": basecolor_atlas,
            "detail_normal_atlas": detail_normal_atlas,
            "sample_id": str(metadata.get("sample_id", "")),
            "data_root": str(metadata.get("data_root", "")),
            "image_path": str(metadata.get("image_path", "")),
            "sample_key": str(metadata.get("sample_key", record.sample_key)),
            "cache_dir": str(sample_dir),
            "split": str(metadata.get("split", record.split)),
            "detection_source": str(metadata.get("detection_source", "")),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.records:
            raise IndexError("UVAtlasDataset is empty")

        attempts = min(8, len(self.records))
        last_error: Exception | None = None
        for attempt in range(attempts):
            record = self.records[(idx + attempt) % len(self.records)]
            try:
                return self._load_record(record)
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"Failed to load a valid UV atlas sample after {attempts} attempts: {last_error}")


def uv_atlas_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("uv_atlas_collate_fn received empty batch")

    tensor_keys = (
        "uv_input",
        "src_color_uv",
        "pred_basecolor_uv",
        "pred_geo_uv",
        "pred_geometry_normal_uv",
        "pred_detail_normal_uv",
        "uv_valid_mask",
        "basecolor_atlas",
        "detail_normal_atlas",
    )
    collated: dict[str, Any] = {
        key: torch.stack([item[key] for item in batch], dim=0)
        for key in tensor_keys
    }
    list_keys = (
        "sample_id",
        "data_root",
        "image_path",
        "sample_key",
        "cache_dir",
        "split",
        "detection_source",
    )
    for key in list_keys:
        collated[key] = [item[key] for item in batch]
    return collated
