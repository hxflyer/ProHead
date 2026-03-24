"""
predict_geometry_dataset.py

Run the geometry transformer on real images and save predictions in the same
text format as MetaHuman GT data, so they can be used as pseudo-GT for training.

Uses Align5PtHelper's MediaPipe + jaw-point alignment before geometry inference.

Output per image (mirrors FastGeometryDataset expected layout):
  {output_dir}/Color_{image_id}<orig_ext>         -- hardlink/symlink/copy of input image
  {output_dir}/landmark/landmark_{image_id}.txt   -- N_full × 5: x3d y3d z3d x2d y2d
  {output_dir}/mesh/mesh_{image_id}.txt           -- N_full × 5: x3d y3d z3d x2d y2d

2D (x2d, y2d) are in original image pixel space.
FastGeometryDataset._load_geometry divides by 1024, so for 1024×1024 images
this matches the GT convention exactly.
3D (x3d, y3d, z3d) are model predictions in normalized space (~[-1,1]).
The dataset renormalizes per-sample by bbox, so the shape is preserved.
"""

import argparse
import faulthandler
import os
import signal
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from tqdm import tqdm

from geometry_transformer import GeometryTransformer
from align_5pt_helper import Align5PtHelper
from obj_load_helper import load_uv_obj_file
from train_visualize_helper import load_combined_mesh_uv

# ──────────────────────────────────────────────────────────────────────────────
# Constants matching FastGeometryDataset training-time alignment
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE = 512   # model input size

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# CUDA helper for multi-GPU preprocessing.

def _set_cv_cuda_device(device_id: int) -> None:
    try:
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "setDevice"):
            cv2.cuda.setDevice(int(device_id))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary data (mirrors inference_geometry.py load_aux_data)
# ──────────────────────────────────────────────────────────────────────────────

def _load_combined_mesh_faces(model_dir: str) -> np.ndarray:
    part_files = ["mesh_head.obj", "mesh_eye_l.obj", "mesh_eye_r.obj", "mesh_mouth.obj"]
    tris, offset = [], 0
    for fn in part_files:
        path = os.path.join(model_dir, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        verts, uvs, _, v_faces, _, _ = load_uv_obj_file(path, triangulate=True)
        tris.append(np.asarray(v_faces, dtype=np.int32) + offset)
        offset += len(verts)
    return np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), dtype=np.int32)


def _remap_faces(faces: np.ndarray, kept: np.ndarray, n_orig: int) -> np.ndarray:
    remap = np.full((n_orig,), -1, dtype=np.int64)
    remap[kept] = np.arange(len(kept), dtype=np.int64)
    tri_new = remap[faces.astype(np.int64)]
    return tri_new[np.all(tri_new >= 0, axis=1)].astype(np.int32)


def load_aux_data(model_dir: str, device: torch.device):
    # Landmark template
    lm_template = np.load(os.path.join(model_dir, "landmark_template.npy"))
    lm_restore   = None
    lm_idx_path  = os.path.join(model_dir, "landmark_indices.npy")
    if os.path.exists(lm_idx_path):
        lm_idx = np.load(lm_idx_path)
        lm_inv_path = os.path.join(model_dir, "landmark_inverse.npy")
        if os.path.exists(lm_inv_path):
            lm_restore = np.load(lm_inv_path)
        if lm_idx.max() < lm_template.shape[0]:
            lm_template = lm_template[lm_idx]

    # Mesh template
    mesh_template = np.load(os.path.join(model_dir, "mesh_template.npy"))
    mesh_full_n   = int(mesh_template.shape[0])
    mesh_restore  = None
    mesh_uv       = load_combined_mesh_uv(model_dir=model_dir, copy=True).astype(np.float32)
    mesh_uv_full  = mesh_uv.copy()
    mesh_faces_full = _load_combined_mesh_faces(model_dir)
    mesh_faces      = mesh_faces_full.copy()

    mesh_idx_path = os.path.join(model_dir, "mesh_indices.npy")
    if os.path.exists(mesh_idx_path):
        mesh_idx = np.load(mesh_idx_path)
        mesh_inv_path = os.path.join(model_dir, "mesh_inverse.npy")
        if os.path.exists(mesh_inv_path):
            mesh_restore = np.load(mesh_inv_path)
        if mesh_idx.max() < mesh_template.shape[0]:
            mesh_template = mesh_template[mesh_idx]
            if mesh_idx.max() < mesh_uv.shape[0]:
                mesh_uv = mesh_uv[mesh_idx]
            mesh_faces = _remap_faces(mesh_faces_full, mesh_idx, mesh_full_n)

    # KNN
    lm2kp_idx = np.load(os.path.join(model_dir, "landmark2keypoint_knn_indices.npy"))
    lm2kp_w   = np.load(os.path.join(model_dir, "landmark2keypoint_knn_weights.npy"))
    n_keypoint = int(lm2kp_idx.max()) + 1
    m2lm_idx  = np.load(os.path.join(model_dir, "mesh2landmark_knn_indices.npy"))
    m2lm_w    = np.load(os.path.join(model_dir, "mesh2landmark_knn_weights.npy"))

    if mesh_uv.shape[0] != mesh_template.shape[0]:
        if mesh_template.shape[1] >= 5:
            mesh_uv = mesh_template[:, 3:5].astype(np.float32)
        else:
            mesh_uv = np.zeros((mesh_template.shape[0], 2), dtype=np.float32)

    return {
        "num_lm":         lm_template.shape[0],
        "num_mesh":       mesh_template.shape[0],
        "lm_template":    lm_template,
        "mesh_template":  mesh_template,
        "lm2kp_idx":      lm2kp_idx,
        "lm2kp_w":        lm2kp_w,
        "m2lm_idx":       m2lm_idx,
        "m2lm_w":         m2lm_w,
        "n_keypoint":     n_keypoint,
        "lm_restore":     lm_restore,
        "mesh_restore":   mesh_restore,
        "mesh_uv":        mesh_uv,
        "mesh_uv_full":   mesh_uv_full,
        "mesh_faces":     mesh_faces,
        "mesh_faces_full": mesh_faces_full,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(args, aux: dict, device: torch.device) -> GeometryTransformer:
    model = GeometryTransformer(
        num_landmarks=aux["num_lm"],
        num_mesh=aux["num_mesh"],
        template_landmark=aux["lm_template"],
        template_mesh=aux["mesh_template"],
        landmark2keypoint_knn_indices=aux["lm2kp_idx"],
        landmark2keypoint_knn_weights=aux["lm2kp_w"],
        mesh2landmark_knn_indices=aux["m2lm_idx"],
        mesh2landmark_knn_weights=aux["m2lm_w"],
        n_keypoint=aux["n_keypoint"],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        backbone_weights=args.backbone_weights,
        model_type=args.model_type,
        k_bins=args.k_bins,
        simdr_range_3d=(args.simdr_min_3d, args.simdr_max_3d),
        simdr_range_2d=(args.simdr_min_2d, args.simdr_max_2d),
        use_deformable_attention=(args.use_deformable_attention if args.model_type == "simdr" else False),
        num_deformable_points=args.num_deformable_points,
        template_mesh_uv=aux["mesh_uv"],
        template_mesh_uv_full=aux["mesh_uv_full"],
        template_mesh_faces=aux["mesh_faces"],
        template_mesh_faces_full=aux["mesh_faces_full"],
        mesh_restore_indices=aux["mesh_restore"],
    ).to(device)

    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    # Filter to only keys whose shape matches the current model.
    # Geometric buffers (template_mesh_faces etc.) may differ in size between
    # checkpoint and current template files — skip them and keep local values.
    model_state = model.state_dict()
    compatible   = {}
    skipped_shape = []
    skipped_missing = []
    for k, v in state.items():
        if k not in model_state:
            skipped_missing.append(k)
        elif model_state[k].shape != v.shape:
            skipped_shape.append(f"  {k}: ckpt{tuple(v.shape)} vs model{tuple(model_state[k].shape)}")
        else:
            compatible[k] = v

    print(f"[Checkpoint] loaded {len(compatible)}/{len(state)} keys "
          f"(shape-mismatch: {len(skipped_shape)}, not-in-model: {len(skipped_missing)})")
    if skipped_shape:
        print("[Checkpoint] shape-mismatched keys (SKIPPED — likely wrong vertex count):")
        for s in skipped_shape:
            print(s)

    model.load_state_dict(compatible, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing (ImageNet normalisation)
# ──────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    """Return normalised CHW float32 ready for batching."""
    x = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)  # CHW


# ──────────────────────────────────────────────────────────────────────────────
# Text-file saving
# ──────────────────────────────────────────────────────────────────────────────

def save_geometry_txt(path: str, xyz: np.ndarray, uv_px: np.ndarray) -> None:
    """Save geometry as 'x3d y3d z3d x2d y2d' per line (N_full rows)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.concatenate([xyz[:, :3], uv_px[:, :2]], axis=1).astype(np.float64)
    np.savetxt(path, data, fmt="%.6f")


def _color_output_path(out_dir: str, image_id: str, src_path: str) -> str:
    ext = os.path.splitext(src_path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        ext = ".png"
    return os.path.join(out_dir, f"Color_{image_id}{ext}")


def _materialize_color_image(src_path: str, dst_path: str) -> None:
    """Create Color_{id}<orig_ext> as a hardlink/symlink/copy of the original image when possible."""
    if os.path.exists(dst_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    src_abs = os.path.abspath(src_path)
    dst_abs = os.path.abspath(dst_path)
    if src_abs == dst_abs:
        return

    try:
        os.link(src_abs, dst_abs)
        return
    except OSError:
        pass

    try:
        os.symlink(src_abs, dst_abs)
        return
    except OSError:
        pass

    shutil.copy2(src_abs, dst_abs)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image processing
# ──────────────────────────────────────────────────────────────────────────────

_5PT_COLORS = [
    (255,  80,  80),   # right_eye  — red
    ( 80, 200,  80),   # left_eye   — green
    ( 80, 150, 255),   # nose       — blue
    (255, 200,  50),   # mouth_r    — yellow
    (220,  80, 220),   # mouth_l    — magenta
]
_5PT_LABELS = ["R_Eye", "L_Eye", "Nose", "Mouth_R", "Mouth_L"]


def _draw_5pt(img_rgb: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Return a copy of img_rgb with 5 keypoints drawn."""
    out = img_rgb.copy()
    for i, (pt, col, lbl) in enumerate(zip(pts_px, _5PT_COLORS, _5PT_LABELS)):
        if not np.isfinite(pt).all():
            continue
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        cv2.circle(out, (x, y), 8, col, -1)
        cv2.circle(out, (x, y), 10, (255, 255, 255), 2)
        cv2.putText(out, lbl, (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return out


def _save_test_vis(
    vis_dir:    str,
    image_id:   str,
    img_rgb:    np.ndarray,
    five_pts:   np.ndarray,
    img_aligned: np.ndarray,
    M:          np.ndarray,
) -> None:
    """Save a side-by-side PNG: original (with 5-pt overlay) | aligned image."""
    os.makedirs(vis_dir, exist_ok=True)

    # Draw 5 pts on original (resize to 512 for consistent panel size)
    orig_disp = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    scale_x = IMAGE_SIZE / float(img_rgb.shape[1])
    scale_y = IMAGE_SIZE / float(img_rgb.shape[0])
    pts_disp = five_pts.copy()
    pts_disp[:, 0] *= scale_x
    pts_disp[:, 1] *= scale_y
    orig_disp = _draw_5pt(orig_disp, pts_disp)

    # Draw where 5 pts land on the aligned image
    ones      = np.ones((5, 1), dtype=np.float32)
    pts_aligned = (M @ np.concatenate([five_pts, ones], axis=1).T).T   # [5, 2]
    aligned_disp = _draw_5pt(img_aligned.copy(), pts_aligned)

    # Separator bar
    sep = np.full((IMAGE_SIZE, 4, 3), 200, dtype=np.uint8)

    panel = np.concatenate([orig_disp, sep, aligned_disp], axis=1)
    out_path = os.path.join(vis_dir, f"{image_id}_test.png")
    cv2.imwrite(out_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _save_predictions_vis(
    vis_dir:    str,
    image_id:   str,
    img_orig:   np.ndarray,   # [H, W, 3] RGB, original (unaligned) image
    lm_uv_px:   np.ndarray,   # [N_lm, 2]  in 1024-convention original space
    mesh_uv_px: np.ndarray,   # [N_mesh, 2] in 1024-convention original space
) -> None:
    """Draw predicted 2D landmarks + mesh vertices on the original image and save."""
    os.makedirs(vis_dir, exist_ok=True)
    # Resize original image to 1024×1024; UV is in 1024-convention so maps 1:1
    canvas = cv2.resize(img_orig, (1024, 1024), interpolation=cv2.INTER_LINEAR).copy()
    # Draw mesh as small dots (blue)
    for pt in mesh_uv_px:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(canvas, (x, y), 1, (80, 150, 255), -1)
    # Draw landmarks as larger dots (red)
    for pt in lm_uv_px:
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if 0 <= x < 1024 and 0 <= y < 1024:
            cv2.circle(canvas, (x, y), 3, (255, 80, 80), -1)
    out_path = os.path.join(vis_dir, f"{image_id}_pred2d.png")
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_obj(path: str, vertices_xyz: np.ndarray, faces: np.ndarray) -> None:
    """Save a mesh as a minimal Wavefront OBJ file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for v in vertices_xyz:
            f.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


def _outputs_exist(out_dir: str, image_id: str) -> bool:
    lm_path   = os.path.join(out_dir, "landmark", f"landmark_{image_id}.txt")
    mesh_path = os.path.join(out_dir, "mesh",     f"mesh_{image_id}.txt")
    return os.path.exists(lm_path) and os.path.exists(mesh_path)


def _preprocess_one(img_path: str, image_id: str, align_helper: Align5PtHelper) -> dict | None:
    """CPU preprocessing: read → detect 5-pt → align → normalise tensor.
    Returns a dict with all data needed for saving, or None on failure.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    lm6, _ = align_helper.detect_landmarks(img_rgb, fallback_lm_px=None)
    M = align_helper.estimate_alignment_matrix(lm6, src_w=w_orig, src_h=h_orig, split="val")
    five_pts, five_valid = align_helper.extract_key5_from_lm68(lm6)
    if not bool(five_valid.any()):
        five_pts = np.full((5, 2), np.nan, dtype=np.float32)

    img_aligned = cv2.warpAffine(
        img_rgb,
        M,
        (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return {
        "img_path":    img_path,
        "image_id":   image_id,
        "img_rgb":    img_rgb,
        "img_aligned": img_aligned,
        "five_pts":   five_pts,
        "M":          M,
        "tensor":     torch.from_numpy(preprocess(img_aligned)),  # CHW float32
    }


def process_batch(
    batch_items:  list[tuple[str, str]],   # [(img_path, image_id), ...]
    model:        GeometryTransformer,
    align_helper: Align5PtHelper,
    aux:          dict,
    device:       torch.device,
    args,
    out_dir:      str,
    save_pool:    ThreadPoolExecutor,
    preprocess_workers: int,
    test_vis_dir: str | None = None,
) -> tuple[int, list]:
    """CPU-preprocess batch (parallel), one GPU forward pass, save outputs (parallel).
    Returns #saved.  save_pool is shared by the caller for asynchronous file writes.
    """
    # Phase 1: CPU preprocessing (imread + face detect + warpAffine).
    # Each worker thread uses its own detector instance so preprocessing can scale
    # without sharing native backend state across threads.
    if preprocess_workers <= 1 or len(batch_items) <= 1:
        preprocessed = []
        for p, iid in batch_items:
            result = _preprocess_one(p, iid, align_helper)
            if result is not None:
                preprocessed.append(result)
    else:
        with ThreadPoolExecutor(max_workers=min(preprocess_workers, len(batch_items))) as preprocess_pool:
            futures = [preprocess_pool.submit(_preprocess_one, p, iid, align_helper) for p, iid in batch_items]
            preprocessed = [r for r in (f.result() for f in futures) if r is not None]

    if not preprocessed:
        return 0, []

    # Phase 2: batch GPU inference
    tensors = torch.stack([item["tensor"] for item in preprocessed]).to(device, non_blocking=True)  # [B,3,H,W]
    final_only_mask = [False] * (args.num_layers - 1) + [True]
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda" and bool(getattr(args, "amp", True))),
            dtype=torch.float16,
        ):
            outputs = model(
                tensors,
                return_logits_mask=[False] * args.num_layers,
                predict_layer_mask=final_only_mask,
                decode_texture_mask=[False] * args.num_layers,
            )
    final    = outputs[-1]
    lm_all   = final["landmark"].detach().cpu().float().numpy()   # [B, N_unique, 6]
    mesh_all = final["mesh"].detach().cpu().float().numpy()        # [B, N_unique, 6]

    lm_restore   = aux["lm_restore"]
    mesh_restore = aux["mesh_restore"]

    def _uv_aligned_to_orig_1024(uv_norm: np.ndarray, M_inv: np.ndarray, w_orig: int, h_orig: int) -> np.ndarray:
        px = uv_norm * 512.0
        ones = np.ones((len(px), 1), dtype=np.float32)
        px_orig = (M_inv @ np.concatenate([px, ones], axis=1).T).T
        px_orig[:, 0] *= 1024.0 / w_orig
        px_orig[:, 1] *= 1024.0 / h_orig
        return px_orig.astype(np.float32)

    def _save_one(i: int, item: dict) -> None:
        img_path     = item["img_path"]
        image_id    = item["image_id"]
        img_rgb     = item["img_rgb"]
        img_aligned = item["img_aligned"]
        five_pts    = item["five_pts"]
        M           = item["M"]
        h_orig, w_orig = img_rgb.shape[:2]
        M_inv = cv2.invertAffineTransform(M)

        lm_pred   = lm_all[i]
        mesh_pred = mesh_all[i]
        lm_full   = lm_pred[lm_restore]     if lm_restore   is not None else lm_pred
        mesh_full = mesh_pred[mesh_restore] if mesh_restore is not None else mesh_pred

        lm_uv_orig   = _uv_aligned_to_orig_1024(lm_full[:, 3:5].astype(np.float32),   M_inv, w_orig, h_orig)
        mesh_uv_orig = _uv_aligned_to_orig_1024(mesh_full[:, 3:5].astype(np.float32), M_inv, w_orig, h_orig)

        save_geometry_txt(
            os.path.join(out_dir, "landmark", f"landmark_{image_id}.txt"),
            lm_full[:, :3], lm_uv_orig,
        )
        save_geometry_txt(
            os.path.join(out_dir, "mesh", f"mesh_{image_id}.txt"),
            mesh_full[:, :3], mesh_uv_orig,
        )

        color_dst = _color_output_path(out_dir, image_id, img_path)
        _materialize_color_image(img_path, color_dst)

        if test_vis_dir is not None:
            _save_test_vis(test_vis_dir, image_id, img_rgb, five_pts, img_aligned, M)
            _save_predictions_vis(test_vis_dir, image_id, img_rgb, lm_uv_orig, mesh_uv_orig)
            _save_obj(
                os.path.join(test_vis_dir, f"{image_id}_pred.obj"),
                mesh_full[:, :3],
                aux["mesh_faces_full"],
            )

    # Phase 3: parallel saves
    save_futures = [save_pool.submit(_save_one, i, item) for i, item in enumerate(preprocessed)]
    return len(preprocessed), save_futures


# ──────────────────────────────────────────────────────────────────────────────
# Multi-GPU worker
# ──────────────────────────────────────────────────────────────────────────────

def _gpu_worker(rank: int, num_gpus: int, image_list: list, args, is_test: bool) -> None:
    """One process per GPU. Processes image_list[rank::num_gpus] in batches."""
    pbar = None
    try:
        faulthandler.enable(all_threads=True)
        if torch.cuda.is_available():
            _set_cv_cuda_device(rank)

        device = torch.device(f"cuda:{rank}")
        aux    = load_aux_data(args.model_dir, device)
        model  = load_model(args, aux, device)
        align_helper = Align5PtHelper(
            image_size=IMAGE_SIZE,
        )

        my_images    = image_list[rank::num_gpus]   # interleaved slice for this GPU
        out_dir      = args.output_dir
        test_vis_dir = os.path.join(out_dir, "test_vis") if is_test else None
        bs           = args.batch_size
        per_gpu_cpu_budget = max(1, (os.cpu_count() or 1) // max(1, num_gpus))
        preprocess_workers = int(getattr(args, "preprocess_workers", 0))
        if preprocess_workers <= 0:
            preprocess_workers = max(1, min(bs, 4, per_gpu_cpu_budget))
        save_workers = int(getattr(args, "save_workers", 0))
        if save_workers <= 0:
            save_workers = max(1, min(max(2, bs), 4, per_gpu_cpu_budget))
        max_pending_batches = max(1, int(getattr(args, "max_pending_batches", 4)))

        ok = skip = 0
        pending_saves = []
        pbar = tqdm(total=len(my_images), desc=f"GPU{rank}", position=rank, leave=True)
        print(
            f"[GPU{rank}] preprocess_workers={preprocess_workers} "
            f"save_workers={save_workers} align=mediapipe-jaw"
        )
        with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
            for start in range(0, len(my_images), bs):
                batch = my_images[start : start + bs]
                if not is_test:
                    filtered = [(p, iid) for p, iid in batch if not _outputs_exist(out_dir, iid)]
                    skip += len(batch) - len(filtered)
                    batch = filtered
                if batch:
                    saved_now, save_futures = process_batch(
                        batch, model, align_helper, aux, device, args, out_dir,
                        save_pool, preprocess_workers, test_vis_dir,
                    )
                    ok += saved_now
                    pending_saves.extend(save_futures)
                    if len(pending_saves) >= max_pending_batches * max(1, bs):
                        drain_count = max(1, len(save_futures))
                        for fut in pending_saves[:drain_count]:
                            fut.result()
                        pending_saves = pending_saves[drain_count:]
                pbar.update(len(my_images[start : start + bs]))
            for fut in pending_saves:
                fut.result()
        print(f"[GPU{rank}] saved: {ok}  skipped: {skip}  failed: {len(my_images) - ok - skip}")
    except Exception as e:
        print(f"[GPU{rank}] worker crashed: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if pbar is not None:
            pbar.close()


# ──────────────────────────────────────────────────────────────────────────────
# Collection helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_color_prefix(name: str) -> str:
    """Remove leading 'Color_' prefix from a file stem if present."""
    if name.startswith("Color_"):
        return name[len("Color_"):]
    return name


def collect_images(input_path: str, max_images: int = -1) -> list[tuple[str, str]]:
    """Return list of (image_path, image_id) sorted deterministically."""
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in SUPPORTED_EXTS:
            stem = os.path.splitext(os.path.basename(input_path))[0]
            image_id = _strip_color_prefix(stem)
            return [(input_path, image_id)]
        return []

    # Prefer original source images over generated Color_* aliases so reruns against
    # the same folder do not double-count copied outputs.
    items_by_id = {}
    for fn in sorted(os.listdir(input_path)):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
        is_generated_color = fn.startswith("Color_")
        image_id = _strip_color_prefix(os.path.splitext(fn)[0])
        current = (os.path.join(input_path, fn), image_id)
        existing = items_by_id.get(image_id)
        if existing is None:
            items_by_id[image_id] = current
            continue
        existing_is_generated = os.path.basename(existing[0]).startswith("Color_")
        if existing_is_generated and not is_generated_color:
            items_by_id[image_id] = current

    items = [items_by_id[k] for k in sorted(items_by_id.keys())]

    if max_images > 0:
        items = items[:max_images]
    return items


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing as mp

    faulthandler.enable(all_threads=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="Predict geometry dataset from real images")
    parser.add_argument("--image_dir",  type=str, required=True,  help="Input image file or directory")
    parser.add_argument("--model_path", type=str, default="best_geometry_transformer_dim6.pth")
    parser.add_argument("--output_dir", type=str, default="geometry_pseudo_gt")
    parser.add_argument("--model_dir",  type=str, default="model", help="Directory with template .npy files")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--gpu_id",     type=int, default=0,   help="GPU index when num_gpus=1")
    parser.add_argument("--num_gpus",   type=int, default=1,   help="Number of GPUs to use in parallel")
    parser.add_argument("--batch_size", type=int, default=4,   help="Images per GPU forward pass")
    parser.add_argument("--preprocess_workers", type=int, default=0,
                        help="CPU workers per GPU for image decode + MediaPipe+jaw alignment (0 = auto)")
    parser.add_argument("--save_workers", type=int, default=0,
                        help="CPU workers per GPU for writing output files (0 = auto)")
    parser.add_argument("--max_pending_batches", type=int, default=4,
                        help="How many batches of save work can queue before draining")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: process test_n images, save visualisations")
    parser.add_argument("--test_n", type=int, default=20,
                        help="Number of images to process in test mode (default 20)")
    parser.add_argument("--amp", dest="amp", action="store_true",
                        help="Enable CUDA autocast for faster inference")
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable CUDA autocast")

    # Model architecture (must match checkpoint)
    parser.add_argument("--d_model",    type=int,   default=512)
    parser.add_argument("--nhead",      type=int,   default=8)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--backbone_weights", type=str, default="checkpoint",
                        help="'checkpoint' skips pretrained file load (backbone from .pth); 'imagenet'/'dinov3' load from disk")
    parser.add_argument("--model_type", type=str,   default="regression", choices=["regression", "simdr"])
    parser.add_argument("--k_bins",     type=int,   default=256)
    parser.add_argument("--simdr_min_3d", type=float, default=-0.5)
    parser.add_argument("--simdr_max_3d", type=float, default=0.5)
    parser.add_argument("--simdr_min_2d", type=float, default=-0.5)
    parser.add_argument("--simdr_max_2d", type=float, default=0.5)
    parser.add_argument("--num_deformable_points", type=int, default=16)
    parser.add_argument("--use_deformable_attention",    dest="use_deformable_attention", action="store_true")
    parser.add_argument("--no_deformable_attention",     dest="use_deformable_attention", action="store_false")
    parser.set_defaults(use_deformable_attention=True)
    parser.set_defaults(amp=True)

    args = parser.parse_args()

    # ── Collect images first (single process) ──
    max_collect = args.test_n if args.test else args.max_images
    image_list  = collect_images(args.image_dir, max_images=max_collect)
    if not image_list:
        print(f"No images found in: {args.image_dir}")
        sys.exit(1)
    print(f"Images: {len(image_list)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "landmark"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mesh"),     exist_ok=True)
    if args.test:
        os.makedirs(os.path.join(args.output_dir, "test_vis"), exist_ok=True)

    # ── Decide GPU count ──
    if args.device != "cuda" or not torch.cuda.is_available():
        num_gpus = 0   # CPU fallback — single process
    else:
        available = torch.cuda.device_count()
        num_gpus  = min(args.num_gpus, available)
        if num_gpus == 0:
            num_gpus = 1

    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}")

    if num_gpus == 1:
        # Single-GPU path: run in-process (avoids spawn overhead)
        args.gpu_id = args.gpu_id if args.device == "cuda" else 0
        _gpu_worker(rank=args.gpu_id, num_gpus=1, image_list=image_list,
                    args=args, is_test=args.test)
    else:
        # Multi-GPU path: one subprocess per GPU
        ctx   = mp.get_context("spawn")
        procs = []
        for rank in range(num_gpus):
            p = ctx.Process(
                target=_gpu_worker,
                args=(rank, num_gpus, image_list, args, args.test),
            )
            p.start()
            procs.append(p)
        failed = []
        for rank, p in enumerate(procs):
            p.join()
            if p.exitcode not in (0, None):
                if p.exitcode < 0:
                    sig_num = -p.exitcode
                    try:
                        sig_name = signal.Signals(sig_num).name
                    except ValueError:
                        sig_name = f"SIG{sig_num}"
                    failed.append(f"GPU{rank} exited from signal {sig_name}")
                else:
                    failed.append(f"GPU{rank} exited with code {p.exitcode}")
        if failed:
            print("[Error] One or more GPU workers failed:")
            for item in failed:
                print(f"  {item}")
            print("[Hint] Start with --preprocess_workers 1 and --num_gpus 1 to confirm the failure mode.")
            sys.exit(1)

    print(f"\nOutput: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
