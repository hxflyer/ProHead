"""
Search raw real-dataset geo features and write matched 2D points back to txt files.

For each sample in a processed dataset root, this script:
1. Loads the canonical geo atlas from assets/topology/geo_feature_atlas.npy.
2. Samples a canonical geo code for each full landmark and mesh vertex.
3. Loads the sample's raw geo EXR and optional face mask.
4. Searches the nearest geo feature in image space for every landmark/mesh point.
5. Rejects weak matches with a fixed or adaptive threshold.
6. Writes updated 2D positions back to landmark/mesh txt files.

Rejected points are written as `-1 -1`.
"""

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
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

from scripts.data.build_combined_knn import load_mesh_data_full
from geometry_dataset import _load_exr_as_float32
from train_visualize_helper import load_combined_mesh_uv


@dataclass
class SamplePaths:
    sample_id: str
    landmark_path: Path
    mesh_path: Path
    geo_path: Path
    facemask_path: Path | None


@dataclass
class SearchStats:
    name: str
    query_count: int
    accepted_count: int
    rejected_count: int
    threshold: float
    median_distance: float
    mad_distance: float
    min_distance: float
    max_distance: float
    valid_geo_pixels: int


def _parse_sample_indices(text: str) -> list[int]:
    values: list[int] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write geo-feature searched 2D points back to a real dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Processed real dataset root")
    parser.add_argument("--model_dir", type=str, default="assets/topology", help="Directory with template assets")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")
    parser.add_argument("--sample_indices", type=str, default="", help="Comma-separated sample indices from the discovered sample list")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample count cap after filtering")
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="Upper cap for accepted geo distance")
    parser.add_argument("--distance_floor", type=float, default=0.02, help="Lower bound for adaptive threshold")
    parser.add_argument("--mad_scale", type=float, default=3.0, help="Adaptive threshold = median + mad_scale * MAD")
    parser.add_argument("--threshold_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--min_geo_magnitude", type=float, default=0.02, help="Ignore near-zero geo pixels")
    parser.add_argument("--tree_leafsize", type=int, default=64, help="cKDTree leaf size")
    parser.add_argument("--summary_path", type=str, default="", help="Optional JSON summary output path")
    return parser


def _choose_device(name: str) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_geo_atlas(model_dir: str, device: torch.device) -> torch.Tensor:
    atlas_path = os.path.join(model_dir, "geo_feature_atlas.npy")
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Missing geo atlas: {atlas_path}")
    atlas = np.load(atlas_path).astype(np.float32)
    if atlas.ndim != 3 or int(atlas.shape[0]) != 3:
        raise ValueError(f"Expected geo atlas [3, H, W], got {atlas.shape}")
    return torch.from_numpy(atlas).to(device=device, dtype=torch.float32)


def _sample_texture_at_uv(texture_chw: torch.Tensor, uv: np.ndarray, device: torch.device) -> torch.Tensor:
    uv_t = torch.from_numpy(np.asarray(uv, dtype=np.float32)).to(device=device)
    uv_t = uv_t.clamp(0.0, 1.0)
    grid_x = uv_t[:, 0] * 2.0 - 1.0
    grid_y = (1.0 - uv_t[:, 1]) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).unsqueeze(0)
    tex = texture_chw.unsqueeze(0)
    sampled = F.grid_sample(
        tex,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0, :].transpose(0, 1).contiguous()


def _load_template_mesh_full(model_dir: str) -> tuple[np.ndarray, np.ndarray]:
    mesh_template = np.load(os.path.join(model_dir, "mesh_template.npy")).astype(np.float32)
    mesh_uv = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32)
    if mesh_template.shape[0] != mesh_uv.shape[0]:
        raise ValueError(
            f"Template mesh UV count mismatch: mesh={mesh_template.shape[0]} uv={mesh_uv.shape[0]}"
        )
    return mesh_template, mesh_uv


def _load_template_landmark_uv_full(model_dir: str) -> tuple[np.ndarray, np.ndarray]:
    landmark_template = np.load(os.path.join(model_dir, "landmark_template.npy")).astype(np.float32)
    part_files = [
        "landmark_head.obj",
        "landmark_eye_l.obj",
        "landmark_eye_r.obj",
        "landmark_mouth.obj",
    ]

    uvs_list: list[np.ndarray] = []
    for file_name in part_files:
        obj_path = os.path.join(model_dir, file_name)
        verts, uvs, faces = load_mesh_data_full(obj_path)
        if uvs is None:
            uvs = np.full((len(verts), 2), -1.0, dtype=np.float32)
        if len(uvs) != len(verts):
            if len(uvs) < len(verts):
                pad = np.full((len(verts) - len(uvs), 2), -1.0, dtype=np.float32)
                uvs = np.vstack([uvs, pad])
            else:
                uvs = uvs[: len(verts)]
        uvs_list.append(np.asarray(uvs, dtype=np.float32))

    combined_uvs = np.concatenate(uvs_list, axis=0)
    if combined_uvs.shape[0] != landmark_template.shape[0]:
        raise ValueError(
            f"Template landmark UV count mismatch: landmark={landmark_template.shape[0]} uv={combined_uvs.shape[0]}"
        )
    return landmark_template, combined_uvs.astype(np.float32, copy=False)


def _build_landmark_codes_from_mesh(
    landmark_template: np.ndarray,
    mesh_template: np.ndarray,
    mesh_codes: torch.Tensor,
    device: torch.device,
    chunk_size: int = 256,
) -> torch.Tensor:
    landmark_xyz = torch.from_numpy(landmark_template[:, :3].astype(np.float32)).to(device)
    mesh_xyz = torch.from_numpy(mesh_template[:, :3].astype(np.float32)).to(device)
    mesh_xyz_t = mesh_xyz.transpose(0, 1).contiguous()
    mesh_norm_sq = (mesh_xyz * mesh_xyz).sum(dim=1)

    nearest_indices: list[torch.Tensor] = []
    for start in range(0, int(landmark_xyz.shape[0]), int(chunk_size)):
        end = min(start + int(chunk_size), int(landmark_xyz.shape[0]))
        query = landmark_xyz[start:end]
        query_norm_sq = (query * query).sum(dim=1, keepdim=True)
        dist_sq = query_norm_sq + mesh_norm_sq.unsqueeze(0) - 2.0 * (query @ mesh_xyz_t)
        nearest_indices.append(dist_sq.argmin(dim=1))
    nearest = torch.cat(nearest_indices, dim=0)
    return mesh_codes[nearest]


def _load_point_geo_codes(model_dir: str, atlas_chw: torch.Tensor, device: torch.device) -> dict[str, np.ndarray]:
    mesh_template, mesh_uv = _load_template_mesh_full(model_dir)
    landmark_template, landmark_uv = _load_template_landmark_uv_full(model_dir)

    mesh_codes = _sample_texture_at_uv(atlas_chw, mesh_uv, device=device)
    landmark_uv_valid = np.isfinite(landmark_uv).all(axis=1) & np.all(landmark_uv >= 0.0, axis=1) & np.all(landmark_uv <= 1.0, axis=1)
    if bool(np.any(landmark_uv_valid)):
        landmark_codes = torch.zeros((landmark_uv.shape[0], 3), device=device, dtype=torch.float32)
        landmark_codes[:] = float("nan")
        valid_mask_t = torch.from_numpy(landmark_uv_valid).to(device=device)
        landmark_codes[valid_mask_t] = _sample_texture_at_uv(atlas_chw, landmark_uv[landmark_uv_valid], device=device)
        if not bool(np.all(landmark_uv_valid)):
            fallback = _build_landmark_codes_from_mesh(
                landmark_template=landmark_template,
                mesh_template=mesh_template,
                mesh_codes=mesh_codes,
                device=device,
            )
            landmark_codes[~valid_mask_t] = fallback[~valid_mask_t]
    else:
        landmark_codes = _build_landmark_codes_from_mesh(
            landmark_template=landmark_template,
            mesh_template=mesh_template,
            mesh_codes=mesh_codes,
            device=device,
        )

    return {
        "landmark_codes": landmark_codes.detach().cpu().numpy().astype(np.float32),
        "mesh_codes": mesh_codes.detach().cpu().numpy().astype(np.float32),
    }


def _load_face_mask(path: Path | None) -> np.ndarray | None:
    if path is None or (not path.exists()):
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3 and mask.shape[2] == 4:
        mask = mask[:, :, 3]
    elif mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask > 127


def _collect_candidates(
    geo_np: np.ndarray,
    facemask: np.ndarray | None,
    min_geo_magnitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    geo = np.asarray(geo_np, dtype=np.float32)
    valid_mask = np.linalg.norm(geo, axis=-1) > float(min_geo_magnitude)
    if facemask is not None:
        if facemask.shape[:2] != geo.shape[:2]:
            facemask = cv2.resize(facemask.astype(np.uint8), (geo.shape[1], geo.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        valid_mask &= facemask
    ys, xs = np.nonzero(valid_mask)
    if xs.size <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    feats = geo[ys, xs].astype(np.float32, copy=False)
    pos = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    return feats, pos


def _nearest_feature_search(
    query_codes: np.ndarray,
    candidate_codes: np.ndarray,
    candidate_positions: np.ndarray,
    leafsize: int,
) -> tuple[np.ndarray, np.ndarray]:
    if int(query_codes.shape[0]) <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if int(candidate_codes.shape[0]) <= 0:
        return np.full((query_codes.shape[0], 2), -1.0, dtype=np.float32), np.full((query_codes.shape[0],), np.inf, dtype=np.float32)
    tree = cKDTree(candidate_codes, leafsize=max(int(leafsize), 1))
    distances, indices = tree.query(query_codes, k=1, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    return candidate_positions[indices].astype(np.float32, copy=False), distances


def _compute_threshold(distances: np.ndarray, args) -> tuple[float, float, float]:
    if distances.size <= 0:
        return float(args.distance_threshold), 0.0, 0.0
    finite = np.isfinite(distances)
    if not np.any(finite):
        return float(args.distance_threshold), 0.0, 0.0
    values = distances[finite].astype(np.float32, copy=False)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if str(args.threshold_mode) == "fixed":
        threshold = float(args.distance_threshold)
    else:
        adaptive = median + float(args.mad_scale) * mad
        threshold = min(float(args.distance_threshold), adaptive)
        threshold = max(float(args.distance_floor), threshold)
    return float(threshold), median, mad


def _load_geometry_txt(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 5:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            elif len(parts) >= 3:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), -1.0, -1.0])
    if not rows:
        raise ValueError(f"No geometry rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def _write_geometry_txt(path: Path, geom: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        for row in np.asarray(geom, dtype=np.float32):
            f.write(
                f"{float(row[0]):.6f} {float(row[1]):.6f} {float(row[2]):.6f} "
                f"{float(row[3]):.6f} {float(row[4]):.6f}\n"
            )
    os.replace(tmp_path, path)


def _collect_sample_paths(data_root: Path) -> list[SamplePaths]:
    landmark_dir = data_root / "landmark"
    mesh_dir = data_root / "mesh"
    geo_dir = data_root / "geo"
    facemask_dir = data_root / "facemask"
    if not landmark_dir.exists() or not mesh_dir.exists() or not geo_dir.exists():
        raise FileNotFoundError(f"Dataset root is missing landmark/mesh/geo folders: {data_root}")

    samples: list[SamplePaths] = []
    for landmark_path in sorted(landmark_dir.glob("landmark_*.txt")):
        sample_id = landmark_path.stem[len("landmark_") :]
        mesh_path = mesh_dir / f"mesh_{sample_id}.txt"
        geo_path = geo_dir / f"Geo_{sample_id}.exr"
        facemask_path = facemask_dir / f"Face_Mask_{sample_id}.png"
        if not mesh_path.exists() or not geo_path.exists():
            continue
        samples.append(
            SamplePaths(
                sample_id=sample_id,
                landmark_path=landmark_path,
                mesh_path=mesh_path,
                geo_path=geo_path,
                facemask_path=facemask_path if facemask_path.exists() else None,
            )
        )
    return samples


def _select_samples(samples: list[SamplePaths], args) -> list[SamplePaths]:
    explicit = _parse_sample_indices(args.sample_indices)
    if explicit:
        selected = [samples[idx] for idx in explicit if 0 <= int(idx) < len(samples)]
    elif int(args.limit) > 0:
        selected = samples[: int(args.limit)]
    else:
        selected = samples
    return selected


def _build_stats(
    name: str,
    distances: np.ndarray,
    accepted_mask: np.ndarray,
    threshold: float,
    median: float,
    mad: float,
    valid_geo_pixels: int,
) -> SearchStats:
    finite = distances[np.isfinite(distances)]
    return SearchStats(
        name=name,
        query_count=int(distances.size),
        accepted_count=int(np.count_nonzero(accepted_mask)),
        rejected_count=int(distances.size - np.count_nonzero(accepted_mask)),
        threshold=float(threshold),
        median_distance=float(median),
        mad_distance=float(mad),
        min_distance=float(finite.min()) if finite.size > 0 else 0.0,
        max_distance=float(finite.max()) if finite.size > 0 else 0.0,
        valid_geo_pixels=int(valid_geo_pixels),
    )


def _process_one_sample(sample: SamplePaths, point_codes: dict[str, np.ndarray], args) -> dict:
    geo_np = _load_exr_as_float32(str(sample.geo_path))
    if geo_np is None:
        raise RuntimeError(f"Failed to load geo EXR: {sample.geo_path}")
    facemask = _load_face_mask(sample.facemask_path)
    candidate_codes, candidate_positions = _collect_candidates(
        geo_np=geo_np,
        facemask=facemask,
        min_geo_magnitude=float(args.min_geo_magnitude),
    )

    landmark_geom = _load_geometry_txt(sample.landmark_path)
    mesh_geom = _load_geometry_txt(sample.mesh_path)
    landmark_codes = point_codes["landmark_codes"]
    mesh_codes = point_codes["mesh_codes"]

    if int(landmark_geom.shape[0]) != int(landmark_codes.shape[0]):
        raise ValueError(
            f"Landmark count mismatch for {sample.sample_id}: file={landmark_geom.shape[0]} codes={landmark_codes.shape[0]}"
        )
    if int(mesh_geom.shape[0]) != int(mesh_codes.shape[0]):
        raise ValueError(
            f"Mesh count mismatch for {sample.sample_id}: file={mesh_geom.shape[0]} codes={mesh_codes.shape[0]}"
        )

    landmark_positions, landmark_distances = _nearest_feature_search(
        query_codes=landmark_codes,
        candidate_codes=candidate_codes,
        candidate_positions=candidate_positions,
        leafsize=int(args.tree_leafsize),
    )
    landmark_threshold, landmark_median, landmark_mad = _compute_threshold(landmark_distances, args)
    landmark_accept = np.isfinite(landmark_distances) & (landmark_distances <= float(landmark_threshold))

    mesh_positions, mesh_distances = _nearest_feature_search(
        query_codes=mesh_codes,
        candidate_codes=candidate_codes,
        candidate_positions=candidate_positions,
        leafsize=int(args.tree_leafsize),
    )
    mesh_threshold, mesh_median, mesh_mad = _compute_threshold(mesh_distances, args)
    mesh_accept = np.isfinite(mesh_distances) & (mesh_distances <= float(mesh_threshold))

    landmark_updated = landmark_geom.copy()
    landmark_updated[:, 3:5] = -1.0
    landmark_updated[landmark_accept, 3:5] = landmark_positions[landmark_accept]

    mesh_updated = mesh_geom.copy()
    mesh_updated[:, 3:5] = -1.0
    mesh_updated[mesh_accept, 3:5] = mesh_positions[mesh_accept]

    _write_geometry_txt(sample.landmark_path, landmark_updated)
    _write_geometry_txt(sample.mesh_path, mesh_updated)

    return {
        "sample_id": sample.sample_id,
        "landmark_stats": asdict(
            _build_stats(
                name="landmarks",
                distances=landmark_distances,
                accepted_mask=landmark_accept,
                threshold=landmark_threshold,
                median=landmark_median,
                mad=landmark_mad,
                valid_geo_pixels=int(candidate_codes.shape[0]),
            )
        ),
        "mesh_stats": asdict(
            _build_stats(
                name="mesh",
                distances=mesh_distances,
                accepted_mask=mesh_accept,
                threshold=mesh_threshold,
                median=mesh_median,
                mad=mesh_mad,
                valid_geo_pixels=int(candidate_codes.shape[0]),
            )
        ),
    }


def main() -> None:
    parser = create_arg_parser()
    args = parser.parse_args()
    device = _choose_device(args.device)
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    atlas_chw = _load_geo_atlas(args.model_dir, device=device)
    point_codes = _load_point_geo_codes(args.model_dir, atlas_chw, device=device)
    print(f"[Info] landmark codes: {point_codes['landmark_codes'].shape}")
    print(f"[Info] mesh codes: {point_codes['mesh_codes'].shape}")

    samples = _collect_sample_paths(data_root)
    samples = _select_samples(samples, args)
    if not samples:
        raise RuntimeError("No samples selected")
    print(f"[Info] selected samples: {len(samples)}")

    summaries: list[dict] = []
    progress = tqdm(samples, desc="Updating point search", unit="sample")
    for idx, sample in enumerate(progress, start=1):
        sample_summary = _process_one_sample(sample, point_codes, args)
        summaries.append(sample_summary)
        landmark_stats = sample_summary["landmark_stats"]
        mesh_stats = sample_summary["mesh_stats"]
        progress.set_postfix(
            lm=f"{landmark_stats['accepted_count']}/{landmark_stats['query_count']}",
            mesh=f"{mesh_stats['accepted_count']}/{mesh_stats['query_count']}",
        )
        if idx <= 3 or idx % 100 == 0:
            print(
                f"[Info] {sample.sample_id}: "
                f"landmarks {landmark_stats['accepted_count']}/{landmark_stats['query_count']} "
                f"thr={landmark_stats['threshold']:.4f}; "
                f"mesh {mesh_stats['accepted_count']}/{mesh_stats['query_count']} "
                f"thr={mesh_stats['threshold']:.4f}"
            )

    landmark_acc = sum(item["landmark_stats"]["accepted_count"] for item in summaries)
    landmark_total = sum(item["landmark_stats"]["query_count"] for item in summaries)
    mesh_acc = sum(item["mesh_stats"]["accepted_count"] for item in summaries)
    mesh_total = sum(item["mesh_stats"]["query_count"] for item in summaries)
    print(
        f"[Done] landmarks accepted {landmark_acc}/{landmark_total} "
        f"({(100.0 * landmark_acc / max(landmark_total, 1)):.2f}%)"
    )
    print(
        f"[Done] mesh accepted {mesh_acc}/{mesh_total} "
        f"({(100.0 * mesh_acc / max(mesh_total, 1)):.2f}%)"
    )

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "device": str(device),
                    "sample_count": len(samples),
                    "landmark_accepted": landmark_acc,
                    "landmark_total": landmark_total,
                    "mesh_accepted": mesh_acc,
                    "mesh_total": mesh_total,
                    "samples": summaries,
                },
                f,
                indent=2,
            )
        print(f"[Info] saved summary json: {summary_path}")


if __name__ == "__main__":
    main()
