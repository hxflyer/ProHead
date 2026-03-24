from __future__ import annotations

import glob
import os
from dataclasses import dataclass


VARIANT_SUFFIXES = ("_gemini", "_flux", "_seedream")


@dataclass(frozen=True)
class GeometrySampleRecord:
    data_root: str
    sample_id: str
    color_path: str
    landmark_path: str
    mesh_path: str
    basecolor_path: str | None
    mask_path: str | None
    facemask_path: str | None
    geo_path: str | None
    mat_path: str | None


def normalize_data_roots(data_roots) -> list[str]:
    if isinstance(data_roots, str):
        return [data_roots]
    return [str(root) for root in data_roots]


def strip_sample_variant_suffix(sample_name: str) -> str:
    out = str(sample_name)
    for suffix in VARIANT_SUFFIXES:
        if out.endswith(suffix):
            return out[: -len(suffix)]
    return out


def parse_color_sample_id(color_path: str) -> str:
    filename = os.path.basename(color_path)
    if filename.startswith("Color_"):
        filename = filename[len("Color_") :]
    sample_name = os.path.splitext(filename)[0]
    return strip_sample_variant_suffix(sample_name)


def _existing_or_none(path: str) -> str | None:
    if path and os.path.exists(path):
        return path
    return None


def _candidate_mask_path(color_path: str) -> str | None:
    dir_name = os.path.dirname(color_path)
    filename = os.path.basename(color_path)
    name_no_ext, ext = os.path.splitext(filename)
    if name_no_ext.startswith("Color_"):
        name_no_ext = name_no_ext[len("Color_") :]
    sample_name = strip_sample_variant_suffix(name_no_ext)

    candidates = [
        os.path.join(dir_name, f"{sample_name}_mask{ext}"),
        os.path.join(dir_name, f"{sample_name}_mask.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_geometry_sample_record(data_root: str, color_path: str) -> GeometrySampleRecord | None:
    if not os.path.exists(color_path):
        return None

    filename = os.path.basename(color_path)
    if not filename.startswith("Color_") or filename.endswith("_mask.png"):
        return None

    sample_id = parse_color_sample_id(color_path)
    landmark_path = os.path.join(data_root, "landmark", f"landmark_{sample_id}.txt")
    mesh_path = os.path.join(data_root, "mesh", f"mesh_{sample_id}.txt")
    if not (os.path.exists(landmark_path) and os.path.exists(mesh_path)):
        return None

    return GeometrySampleRecord(
        data_root=str(data_root),
        sample_id=str(sample_id),
        color_path=str(color_path),
        landmark_path=str(landmark_path),
        mesh_path=str(mesh_path),
        basecolor_path=_existing_or_none(os.path.join(data_root, "basecolor", f"BaseColor_{sample_id}.png")),
        mask_path=_candidate_mask_path(color_path),
        facemask_path=_existing_or_none(os.path.join(data_root, "facemask", f"Face_Mask_{sample_id}.png")),
        geo_path=_existing_or_none(os.path.join(data_root, "geo", f"Geo_{sample_id}.exr")),
        mat_path=_existing_or_none(os.path.join(data_root, "mat", f"Mats_{sample_id}.txt")),
    )


def collect_geometry_sample_records(
    data_roots,
    split: str = "train",
    train_ratio: float = 0.8,
) -> list[GeometrySampleRecord]:
    records: list[GeometrySampleRecord] = []
    for data_root in normalize_data_roots(data_roots):
        if not os.path.exists(data_root):
            continue
        color_files = glob.glob(os.path.join(data_root, "Color_*"))
        for color_path in color_files:
            record = build_geometry_sample_record(data_root, color_path)
            if record is not None:
                records.append(record)

    records = sorted(set(records), key=lambda record: (record.data_root, record.color_path))

    val_ratio = float(min(max(1.0 - float(train_ratio), 0.0), 1.0))
    records_by_root: dict[str, list[GeometrySampleRecord]] = {}
    for record in records:
        records_by_root.setdefault(record.data_root, []).append(record)

    split_records: list[GeometrySampleRecord] = []
    for data_root in sorted(records_by_root.keys()):
        root_records = sorted(records_by_root[data_root], key=lambda record: record.color_path)
        val_count = int(len(root_records) * val_ratio)
        if str(split).lower() == "train":
            split_records.extend(root_records[val_count:])
        else:
            split_records.extend(root_records[:val_count])

    return split_records
