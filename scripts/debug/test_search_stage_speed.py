"""
Benchmark 2-stage vs 3-stage mesh search: speed and match error.
Usage:
    python scripts/debug/test_search_stage_speed.py --dense_checkpoint artifacts/checkpoints/best_dense_image_transformer_ch10.pth
"""

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

import os
import sys
import time
import argparse

import torch
import numpy as np


from dense2geometry import Dense2Geometry, DenseStageConfig


def make_synthetic_dense_outputs(batch_size: int, model: Dense2Geometry, device: torch.device, noise_std: float = 0.005):
    """Create synthetic pred_geo from the geo atlas so that the search can find real matches.

    Paints the geo atlas onto a 1024x1024 map using template mesh UVs, then adds
    small noise to simulate prediction error. This ensures non-trivial matching.
    """
    import torch.nn.functional as F

    geo_atlas = np.load(os.path.join("model", "geo_feature_atlas.npy")).astype(np.float32)
    atlas_t = torch.from_numpy(geo_atlas).to(device)  # [3, H_atlas, W_atlas]

    # Build a 1024x1024 geo map by sampling the atlas at a regular grid
    size = 1024
    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, size, device=device),
        torch.linspace(0, 1, size, device=device),
        indexing="ij",
    )
    # grid_sample expects grid in [-1, 1] with (x, y) order
    # No y-flip: spatial position (x/W, y/H) directly equals UV, so ground truth is template_mesh_uv
    grid = torch.stack([xs * 2 - 1, ys * 2 - 1], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    base_geo = F.grid_sample(
        atlas_t.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(0)  # [3, 1024, 1024]

    # Create a face mask (non-zero geo magnitude)
    geo_mag = torch.linalg.norm(base_geo, dim=0)
    mask_logits = torch.where(geo_mag > 0.02, torch.tensor(5.0, device=device), torch.tensor(-5.0, device=device))
    mask_logits = mask_logits.unsqueeze(0)  # [1, 1024, 1024]

    all_geo = []
    all_mask = []
    for _ in range(batch_size):
        noisy_geo = base_geo + torch.randn_like(base_geo) * noise_std
        all_geo.append(noisy_geo)
        all_mask.append(mask_logits)

    return torch.stack(all_geo, dim=0), torch.stack(all_mask, dim=0)


def run_2stage(model, pred_geo_i, pred_mask_logits_i):
    """Run search with subpixel stage disabled (skip stage 3)."""
    original_method = model._subpixel_refine_mesh_search

    def noop_subpixel(pred_geo, searched_uv, accept, distances):
        return searched_uv, accept, distances
    model._subpixel_refine_mesh_search = noop_subpixel

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    uv, accept, dist = model._search_single_sample(pred_geo_i, pred_mask_logits_i)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    model._subpixel_refine_mesh_search = original_method
    return uv, accept, dist, t1 - t0


def run_3stage(model, pred_geo_i, pred_mask_logits_i):
    """Run search with all 3 stages."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    uv, accept, dist = model._search_single_sample(pred_geo_i, pred_mask_logits_i)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return uv, accept, dist, t1 - t0


def evaluate_geo_error_at_pixel(model, pred_geo_i, uv, accept):
    """Evaluate geo code L2 error by direct pixel lookup (no bilinear interpolation).

    Snaps each UV to the nearest pixel in the 1024 geo map and compares the
    pixel's geo code to the query. This avoids bilinear interpolation giving
    an unfair advantage to UVs that fall between pixels.
    """
    mask = accept > 0.5
    if mask.sum() == 0:
        return torch.zeros(0, device=uv.device)

    matched_uv = uv[mask]  # [K, 2]
    query_codes = model.mesh_geo_codes[mask].float()  # [K, 3]

    H = pred_geo_i.shape[1]
    W = pred_geo_i.shape[2]
    geo_hw = pred_geo_i.permute(1, 2, 0).contiguous()  # [H, W, 3]

    # Snap UV to nearest pixel
    px_x = torch.round(matched_uv[:, 0] * float(W) - 0.5).long().clamp(0, W - 1)
    px_y = torch.round(matched_uv[:, 1] * float(H) - 0.5).long().clamp(0, H - 1)

    sampled = geo_hw[px_y, px_x]  # [K, 3]
    dist = torch.linalg.norm(sampled.float() - query_codes, dim=1)
    return dist


def compute_error_stats(errors):
    """Compute error stats from a tensor."""
    if errors.numel() == 0:
        return 0, float('nan'), float('nan'), float('nan'), float('nan')
    return (
        int(errors.numel()),
        float(errors.mean()),
        float(errors.median()),
        float(errors.std()),
        float(errors.max()),
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark 2-stage vs 3-stage search: speed and error")
    parser.add_argument("--dense_checkpoint", type=str, default="",
                        help="Path to dense image transformer checkpoint (optional)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}, Warmup: {args.warmup}")

    model = Dense2Geometry(
        dense_stage_cfg=DenseStageConfig(),
        dense_checkpoint=args.dense_checkpoint,
        freeze_dense_stage=True,
    ).to(device).eval()

    num_total = args.warmup + args.num_samples
    pred_geo, pred_mask_logits = make_synthetic_dense_outputs(num_total, model, device)

    # Warmup
    print(f"\nWarming up ({args.warmup} iterations)...")
    with torch.no_grad():
        for i in range(args.warmup):
            model._search_single_sample(pred_geo[i], pred_mask_logits[i])
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    pred_geo = pred_geo[args.warmup:]
    pred_mask_logits = pred_mask_logits[args.warmup:]

    # Collect results
    results_2 = []
    results_3 = []
    with torch.no_grad():
        for i in range(args.num_samples):
            r2 = run_2stage(model, pred_geo[i], pred_mask_logits[i])
            r3 = run_3stage(model, pred_geo[i], pred_mask_logits[i])
            results_2.append(r2)
            results_3.append(r3)

    # === Speed Report ===
    print("\n" + "=" * 70)
    print("SPEED COMPARISON")
    print("=" * 70)
    print(f"{'Sample':<10} {'2-stage (ms)':<16} {'3-stage (ms)':<16} {'Overhead (ms)':<16}")
    print("-" * 70)
    times_2 = [r[3] for r in results_2]
    times_3 = [r[3] for r in results_3]
    for i in range(args.num_samples):
        t2 = times_2[i] * 1000
        t3 = times_3[i] * 1000
        print(f"{i:<10} {t2:<16.2f} {t3:<16.2f} {(t3 - t2):<16.2f}")
    avg_2 = np.mean(times_2) * 1000
    avg_3 = np.mean(times_3) * 1000
    print("-" * 70)
    print(f"{'Average':<10} {avg_2:<16.2f} {avg_3:<16.2f} {(avg_3 - avg_2):<16.2f}")
    pct = (avg_3 - avg_2) / avg_2 * 100 if avg_2 > 0 else 0
    print(f"Stage-3 overhead: {avg_3 - avg_2:.2f} ms ({pct:.1f}%)")

    # === Error Report ===
    # === Fair Error Comparison: re-evaluate both at final UV ===
    print("\n" + "=" * 70)
    print("MATCH ERROR COMPARISON (geo code L2 at nearest pixel, no bilinear bias)")
    print("=" * 70)
    print(f"{'Sample':<8} {'Stage':<10} {'Matched':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Max':<10}")
    print("-" * 70)

    all_mean_2, all_mean_3 = [], []
    all_median_2, all_median_3 = [], []
    all_matched_2, all_matched_3 = [], []

    for i in range(args.num_samples):
        uv2, acc2, _, _ = results_2[i]
        uv3, acc3, _, _ = results_3[i]

        err2 = evaluate_geo_error_at_pixel(model, pred_geo[i], uv2, acc2)
        err3 = evaluate_geo_error_at_pixel(model, pred_geo[i], uv3, acc3)

        n2, mean2, med2, std2, max2 = compute_error_stats(err2)
        n3, mean3, med3, std3, max3 = compute_error_stats(err3)

        print(f"{i:<8} {'2-stage':<10} {n2:<10} {mean2:<10.6f} {med2:<10.6f} {std2:<10.6f} {max2:<10.6f}")
        print(f"{'':<8} {'3-stage':<10} {n3:<10} {mean3:<10.6f} {med3:<10.6f} {std3:<10.6f} {max3:<10.6f}")

        all_matched_2.append(n2)
        all_matched_3.append(n3)
        if not np.isnan(mean2):
            all_mean_2.append(mean2)
        if not np.isnan(mean3):
            all_mean_3.append(mean3)
        if not np.isnan(med2):
            all_median_2.append(med2)
        if not np.isnan(med3):
            all_median_3.append(med3)

    # === Summary ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    avg_matched_2 = np.mean(all_matched_2) if all_matched_2 else 0
    avg_matched_3 = np.mean(all_matched_3) if all_matched_3 else 0
    avg_mean_2 = np.mean(all_mean_2) if all_mean_2 else float('nan')
    avg_mean_3 = np.mean(all_mean_3) if all_mean_3 else float('nan')
    avg_med_2 = np.mean(all_median_2) if all_median_2 else float('nan')
    avg_med_3 = np.mean(all_median_3) if all_median_3 else float('nan')

    print(f"{'Metric':<25} {'2-stage':<15} {'3-stage':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Avg matched vertices':<25} {avg_matched_2:<15.0f} {avg_matched_3:<15.0f} {'(same)' if avg_matched_2 == avg_matched_3 else '':<15}")
    if not np.isnan(avg_mean_2) and avg_mean_2 > 0:
        imp_mean = (avg_mean_2 - avg_mean_3) / avg_mean_2 * 100
    else:
        imp_mean = 0
    if not np.isnan(avg_med_2) and avg_med_2 > 0:
        imp_med = (avg_med_2 - avg_med_3) / avg_med_2 * 100
    else:
        imp_med = 0
    print(f"{'Avg mean geo error':<25} {avg_mean_2:<15.6f} {avg_mean_3:<15.6f} {imp_mean:<14.2f}%")
    print(f"{'Avg median geo error':<25} {avg_med_2:<15.6f} {avg_med_3:<15.6f} {imp_med:<14.2f}%")
    print(f"{'Avg time (ms)':<25} {avg_2:<15.2f} {avg_3:<15.2f} {avg_3 - avg_2:<14.2f}ms")

    # === UV shift analysis ===
    print("\n" + "=" * 70)
    print("UV SHIFT ANALYSIS (how much stage-3 moved UVs from stage-2)")
    print("=" * 70)
    print(f"{'Sample':<10} {'Refined':<10} {'Mean px':<12} {'Max px':<12} {'Mean UV':<12}")
    print("-" * 70)
    for i in range(args.num_samples):
        uv2, acc2, _, _ = results_2[i]
        uv3, acc3, _, _ = results_3[i]
        both_accepted = (acc2 > 0.5) & (acc3 > 0.5)
        if both_accepted.sum() > 0:
            diff_uv = (uv3[both_accepted] - uv2[both_accepted]).abs()
            diff_px = diff_uv * torch.tensor([1024.0, 1024.0], device=diff_uv.device)
            shift_px = torch.linalg.norm(diff_px, dim=1)
            n_refined = int((shift_px > 1e-6).sum().item())
            mean_px = float(shift_px.mean())
            max_px = float(shift_px.max())
            mean_uv = float(diff_uv.mean())
            print(f"{i:<10} {n_refined:<10} {mean_px:<12.4f} {max_px:<12.4f} {mean_uv:<12.6f}")
        else:
            print(f"{i:<10} {'N/A':<10}")

    # === Temperature sweep ===
    print("\n" + "=" * 70)
    print("TEMPERATURE SWEEP (finding optimal subpixel_temperature)")
    print("=" * 70)
    temperatures = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    print(f"{'Temperature':<14} {'Mean err':<12} {'Median err':<12} {'Mean shift px':<14} {'Time ms':<10}")
    print("-" * 70)

    # 2-stage baseline for reference
    base_errs = []
    for i in range(args.num_samples):
        uv2, acc2, _, _ = results_2[i]
        err2 = evaluate_geo_error_at_pixel(model, pred_geo[i], uv2, acc2)
        if err2.numel() > 0:
            base_errs.append(err2)
    if base_errs:
        base_all = torch.cat(base_errs)
        print(f"{'2-stage':<14} {float(base_all.mean()):<12.6f} {float(base_all.median()):<12.6f} {'N/A':<14} {avg_2:<10.2f}")

    for temp in temperatures:
        model.subpixel_temperature = temp
        temp_errs = []
        temp_shifts = []
        t_start = time.perf_counter()
        with torch.no_grad():
            for i in range(args.num_samples):
                uv3, acc3, _, _ = run_3stage(model, pred_geo[i], pred_mask_logits[i])[:4]
                err3 = evaluate_geo_error_at_pixel(model, pred_geo[i], uv3, acc3)
                if err3.numel() > 0:
                    temp_errs.append(err3)
                uv2, acc2, _, _ = results_2[i]
                both = (acc2 > 0.5) & (acc3 > 0.5)
                if both.sum() > 0:
                    d = (uv3[both] - uv2[both]) * torch.tensor([1024.0, 1024.0], device=uv3.device)
                    temp_shifts.append(torch.linalg.norm(d, dim=1))
        t_elapsed = (time.perf_counter() - t_start) / args.num_samples * 1000

        if temp_errs:
            all_err = torch.cat(temp_errs)
            mean_e = float(all_err.mean())
            med_e = float(all_err.median())
        else:
            mean_e = med_e = float('nan')
        if temp_shifts:
            mean_shift = float(torch.cat(temp_shifts).mean())
        else:
            mean_shift = float('nan')

        print(f"{temp:<14.0e} {mean_e:<12.6f} {med_e:<12.6f} {mean_shift:<14.4f} {t_elapsed:<10.2f}")

    # Restore original temperature
    model.subpixel_temperature = 0.01


if __name__ == "__main__":
    main()
