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


@dataclass(frozen=True)
class DenseImageSampleRecord:
    data_root: str
    sample_id: str
    color_path: str
    basecolor_path: str
    geo_path: str
    screen_normal_path: str
    face_mask_path: str
    error_mask_path: str | None
    landmark_path: str | None
    geo_normal_path: str


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


def _find_first_existing(paths: list[str]) -> str | None:
    for path in paths:
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


def _split_records_by_root(records, split: str, train_ratio: float, sort_key):
    records = sorted(set(records), key=sort_key)

    val_ratio = float(min(max(1.0 - float(train_ratio), 0.0), 1.0))
    records_by_root: dict[str, list] = {}
    for record in records:
        records_by_root.setdefault(record.data_root, []).append(record)

    split_records: list = []
    for data_root in sorted(records_by_root.keys()):
        root_records = sorted(records_by_root[data_root], key=sort_key)
        val_count = int(len(root_records) * val_ratio)
        if str(split).lower() == "train":
            split_records.extend(root_records[val_count:])
        else:
            split_records.extend(root_records[:val_count])

    return split_records


def _geometry_record_sort_key(record: GeometrySampleRecord):
    return (record.data_root, record.color_path)


def _dense_image_record_sort_key(record: DenseImageSampleRecord):
    return (
        record.data_root,
        record.sample_id,
        record.color_path,
        record.basecolor_path,
        record.geo_path,
        record.screen_normal_path,
        record.face_mask_path,
        record.error_mask_path or "",
        record.landmark_path or "",
        record.geo_normal_path,
    )


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

    return _split_records_by_root(
        records,
        split=split,
        train_ratio=train_ratio,
        sort_key=_geometry_record_sort_key,
    )


def build_dense_image_sample_record(data_root: str, color_path: str) -> DenseImageSampleRecord | None:
    if not os.path.exists(color_path):
        return None

    filename = os.path.basename(color_path)
    if not filename.startswith("Color_") or filename.endswith("_mask.png"):
        return None

    sample_id = parse_color_sample_id(color_path)
    basecolor_path = os.path.join(data_root, "basecolor", f"BaseColor_{sample_id}.png")
    geo_path = os.path.join(data_root, "geo", f"Geo_{sample_id}.exr")
    screen_normal_path = os.path.join(data_root, "normal", f"ScreenNormal_{sample_id}.png")
    face_mask_path = os.path.join(data_root, "facemask", f"Face_Mask_{sample_id}.png")
    geo_normal_path = os.path.join(data_root, "geo_normal", f"GeoNormal_{sample_id}.png")
    landmark_path = _find_first_existing(
        [
            os.path.join(data_root, "landmark", f"landmark_{sample_id}.txt"),
            os.path.join(data_root, "landmark", f"{sample_id}.txt"),
        ]
    )
    error_mask_path = os.path.join(data_root, f"Color_{sample_id}_mask.png")

    required_paths = [
        basecolor_path,
        geo_path,
        screen_normal_path,
        face_mask_path,
        geo_normal_path,
    ]
    if not all(os.path.exists(path) for path in required_paths):
        return None

    return DenseImageSampleRecord(
        data_root=str(data_root),
        sample_id=str(sample_id),
        color_path=str(color_path),
        basecolor_path=str(basecolor_path),
        geo_path=str(geo_path),
        screen_normal_path=str(screen_normal_path),
        face_mask_path=str(face_mask_path),
        error_mask_path=str(error_mask_path) if os.path.exists(error_mask_path) else None,
        landmark_path=str(landmark_path) if landmark_path is not None else None,
        geo_normal_path=str(geo_normal_path),
    )


def collect_dense_image_sample_records(
    data_roots,
    split: str = "train",
    train_ratio: float = 0.95,
) -> list[DenseImageSampleRecord]:
    records: list[DenseImageSampleRecord] = []
    for data_root in normalize_data_roots(data_roots):
        if not os.path.exists(data_root):
            continue

        color_files = glob.glob(os.path.join(data_root, "Color_*"))
        color_files = [
            path
            for path in color_files
            if os.path.basename(path).startswith("Color_")
            and not os.path.basename(path).endswith("_mask.png")
        ]
        for color_path in color_files:
            record = build_dense_image_sample_record(data_root, color_path)
            if record is not None:
                records.append(record)

    return _split_records_by_root(
        records,
        split=split,
        train_ratio=train_ratio,
        sort_key=_dense_image_record_sort_key,
    )
