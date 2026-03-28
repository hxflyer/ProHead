"""
Inference Script for Geometry Transformer (Landmark + Mesh)
Loads a trained model and predicts landmarks and mesh for a given image.
Uses RetinaFace for 5-point face alignment before inference.
Saves the visualization (overlay) and 3D OBJ mesh.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

# Import model and helpers
from geometry_transformer import GeometryTransformer
from data_utils.obj_io import load_uv_obj_file
from train_visualize_helper import (
    load_landmark_topology,
    load_mesh_topology,
    create_combined_overlay,
    load_combined_mesh_uv,
    render_mesh_texture_from_2d_pred,
)

# Face detection using dlib (no TensorFlow dependency!)
_dlib_detector = None
_dlib_predictor = None
DLIB_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("鉂?dlib not available. Install with: pip install dlib")

def _init_dlib_detector():
    """Initialize dlib face and landmark detectors."""
    global _dlib_detector, _dlib_predictor, DLIB_AVAILABLE
    
    if not DLIB_AVAILABLE:
        return False
    
    if _dlib_detector is not None:
        return True
    
    try:
        import dlib
        _dlib_detector = dlib.get_frontal_face_detector()
        
        # Try to load the 68-point landmark predictor
        # This model needs to be downloaded separately
        predictor_path = "assets/pretrained/shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            _dlib_predictor = dlib.shape_predictor(predictor_path)
        else:
            # Try common locations
            common_paths = [
                "assets/pretrained/shape_predictor_68_face_landmarks.dat",
                "/usr/share/dlib/shape_predictor_68_face_landmarks.dat",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    _dlib_predictor = dlib.shape_predictor(path)
                    break
        
        if _dlib_predictor is None:
            print("鈿狅笍 dlib landmark predictor model not found. Download from:")
            print("   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            return False
        
        return True
    except Exception as e:
        print(f"鈿狅笍 dlib detector init failed: {e}")
        return False

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
) -> np.ndarray:
    """Remap face indices from original space to filtered (N_unique) space. Drops faces that
    reference filtered-out vertices. Used only for compute_vertex_normals on N_unique vertices."""
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


def load_aux_data(device):
    """Load auxiliary data needed for model initialization."""
    try:
        def _find_existing_path(*candidates: str) -> str | None:
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return None

        # --- Landmark Data ---
        template_landmark = np.load("assets/topology/landmark_template.npy")
        landmark_restore_indices = None

        landmark_indices_path = _find_existing_path(
            os.path.join("assets", "topology", "landmark_indices.npy"),
            os.path.join("model", "landmark_indices.npy"),
        )
        if landmark_indices_path and os.path.exists(landmark_indices_path):
            landmark_indices = np.load(landmark_indices_path)

            landmark_inverse_path = _find_existing_path(
                os.path.join("assets", "topology", "landmark_inverse.npy"),
                os.path.join("model", "landmark_inverse.npy"),
            )
            if landmark_inverse_path and os.path.exists(landmark_inverse_path):
                print(f"[Info] Loading landmark restoration map from {landmark_inverse_path}...")
                landmark_restore_indices = np.load(landmark_inverse_path)

            if landmark_indices.max() < template_landmark.shape[0]:
                template_landmark = template_landmark[landmark_indices]
                print(f"[Info] Filtered template landmarks: {len(landmark_indices)} points kept.")

        # --- Mesh Data ---
        template_mesh = np.load("assets/topology/mesh_template.npy")
        template_mesh_full_count = int(template_mesh.shape[0])
        template_mesh_uv = load_combined_mesh_uv(model_dir="assets/topology", copy=True).astype(np.float32, copy=False)
        template_mesh_uv_full = template_mesh_uv.copy()
        # Full (N_full indexed) faces — kept intact for seamless rendering via mesh_restore_indices expansion
        template_mesh_faces_full = load_combined_mesh_triangle_faces(model_dir="assets/topology")
        template_mesh_faces = template_mesh_faces_full.copy()  # will be remapped to N_unique

        mesh_restore_indices = None
        mesh_indices_path = _find_existing_path(
            os.path.join("assets", "topology", "mesh_indices.npy"),
            os.path.join("model", "mesh_indices.npy"),
        )
        if mesh_indices_path and os.path.exists(mesh_indices_path):
            mesh_indices = np.load(mesh_indices_path)
            mesh_inverse_path = _find_existing_path(
                os.path.join("assets", "topology", "mesh_inverse.npy"),
                os.path.join("model", "mesh_inverse.npy"),
            )
            if mesh_inverse_path and os.path.exists(mesh_inverse_path):
                print(f"[Info] Loading mesh restoration map from {mesh_inverse_path}...")
                mesh_restore_indices = np.load(mesh_inverse_path)
            if mesh_indices.max() < template_mesh.shape[0]:
                template_mesh = template_mesh[mesh_indices]
                print(f"[Info] Filtered template mesh: {len(mesh_indices)} points kept.")
                if mesh_indices.max() < template_mesh_uv.shape[0]:
                    template_mesh_uv = template_mesh_uv[mesh_indices]
                template_mesh_faces = remap_triangle_faces_after_vertex_filter(
                    template_mesh_faces_full,
                    mesh_indices,
                    original_vertex_count=template_mesh_full_count,
                )

        # --- KNN Mappings ---
        landmark2keypoint_idx = np.load("assets/topology/landmark2keypoint_knn_indices.npy")
        landmark2keypoint_w = np.load("assets/topology/landmark2keypoint_knn_weights.npy")
        n_keypoint = int(landmark2keypoint_idx.max()) + 1

        mesh2landmark_idx = np.load("assets/topology/mesh2landmark_knn_indices.npy")
        mesh2landmark_w = np.load("assets/topology/mesh2landmark_knn_weights.npy")
        num_landmarks = template_landmark.shape[0]
        num_mesh = template_mesh.shape[0]

        if template_mesh_uv.shape[0] != num_mesh:
            print(
                f"[Warn] template_mesh_uv length mismatch ({template_mesh_uv.shape[0]} vs {num_mesh}). "
                "Falling back to template_mesh[:,3:5] when available."
            )
            if template_mesh.shape[1] >= 5:
                template_mesh_uv = template_mesh[:, 3:5].astype(np.float32, copy=True)
            else:
                template_mesh_uv = np.zeros((num_mesh, 2), dtype=np.float32)

        if template_mesh_faces.shape[0] == 0:
            raise ValueError("template_mesh_faces is empty after filtering.")
        if template_mesh_faces.max() >= num_mesh or template_mesh_faces.min() < 0:
            raise ValueError(
                f"template_mesh_faces index range invalid for current mesh size {num_mesh}"
            )

        return (
            num_landmarks,
            num_mesh,
            template_landmark,
            template_mesh,
            landmark2keypoint_idx,
            landmark2keypoint_w,
            mesh2landmark_idx,
            mesh2landmark_w,
            n_keypoint,
            landmark_restore_indices,
            mesh_restore_indices,
            template_mesh_uv,
            template_mesh_uv_full,
            template_mesh_faces,
            template_mesh_faces_full,
        )

    except Exception as e:
        print(f"[Error] Failed to load geometry auxiliary data: {e}")
        raise e

def detect_face_keypoints_dlib(img_path):
    """
    Detect face and extract 5 keypoints using dlib (68-point landmarks).
    No TensorFlow dependency.
    """
    if not _init_dlib_detector():
        return None, False
    
    try:
        import dlib
        img = cv2.imread(img_path)
        if img is None:
            return None, False
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = _dlib_detector(gray, 1)
        
        if len(faces) == 0:
            return None, False
        
        # Use the first face
        face = faces[0]
        
        # Get 68 facial landmarks
        shape = _dlib_predictor(gray, face)
        
        # Extract 5 key points from 68-point model:
        # Right eye: average of points 36-41
        # Left eye: average of points 42-47
        # Nose tip: point 30
        # Right mouth corner: point 48
        # Left mouth corner: point 54
        
        right_eye = np.mean([[shape.part(i).x, shape.part(i).y] for i in range(36, 42)], axis=0)
        left_eye = np.mean([[shape.part(i).x, shape.part(i).y] for i in range(42, 48)], axis=0)
        nose = np.array([shape.part(30).x, shape.part(30).y])
        mouth_right = np.array([shape.part(48).x, shape.part(48).y])
        mouth_left = np.array([shape.part(54).x, shape.part(54).y])
        
        five_points = np.array([
            right_eye,
            left_eye,
            nose,
            mouth_right,
            mouth_left
        ], dtype=np.float32)
        
        return five_points, True
        
    except Exception as e:
        print(f"鈿狅笍 dlib detection failed: {e}")
        return None, False


def detect_face_keypoints(img_path):
    """
    Detect face and extract 5 keypoints using dlib 68-point landmark detector.
    Returns: (five_points_px, success) where five_points_px is [5, 2] array
    Order: [right_eye, left_eye, nose, mouth_right, mouth_left]
    """
    pts, success = detect_face_keypoints_dlib(img_path)
    if success:
        return pts, True
    
    print(f"鈿狅笍 No face detected in {img_path}")
    return None, False


def estimate_alignment_matrix(
    five_pts_px,
    src_w,
    src_h,
    output_size=512,
    output_scale=0.75,
    direction_shift=0.08
):
    """
    Estimate similarity transform from 5 facial keypoints to canonical pose.
    Based on geometry_dataset.py alignment logic.
    
    Args:
        five_pts_px: [5, 2] array in pixel coords [right_eye, left_eye, nose, mouth_right, mouth_left]
        src_w, src_h: source image dimensions
        output_size: target image size
        output_scale: post-alignment scale factor
        direction_shift: extra shift toward face direction
    """
    # Canonical 5-point targets in normalized [0,1] space
    target_norm = np.array([
        [0.35, 0.38],  # right_eye
        [0.65, 0.38],  # left_eye
        [0.50, 0.56],  # nose
        [0.40, 0.72],  # mouth_right
        [0.60, 0.72],  # mouth_left
    ], dtype=np.float32)
    
    target_px = target_norm * float(output_size)
    
    # Make left/right assignment robust
    src = five_pts_px.copy()
    if src[0, 0] > src[1, 0]:  # if right_eye is more left than left_eye
        src[[0, 1]] = src[[1, 0]]
    if src[3, 0] > src[4, 0]:  # if mouth_right is more left than mouth_left
        src[[3, 4]] = src[[4, 3]]
    
    # Estimate affine transformation
    M, _ = cv2.estimateAffinePartial2D(src, target_px, method=cv2.LMEDS)
    
    if M is None:
        # Fallback: simple scaling
        scale = float(output_size) / float(max(src_w, src_h))
        M = np.array([
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0]
        ], dtype=np.float32)
    
    # Apply center scale
    center = np.array([0.5 * output_size, 0.5 * output_size], dtype=np.float32)
    A = M[:, :2]
    b = M[:, 2]
    M[:, :2] = output_scale * A
    M[:, 2] = output_scale * b + (1.0 - output_scale) * center
    
    # Apply direction shift for extreme poses
    if direction_shift > 0:
        # Estimate face direction from keypoints
        nose = five_pts_px[2]
        context = np.vstack([five_pts_px[0], five_pts_px[1], five_pts_px[3], five_pts_px[4]])
        context_center = np.mean(context, axis=0)
        
        extent = np.max(np.linalg.norm(five_pts_px[:, None, :] - five_pts_px[None, :, :], axis=2))
        extent = max(extent, 1.0)
        
        face_dir = (nose - context_center) / extent
        face_dir = np.clip(face_dir, -1.0, 1.0)
        
        M[:, 2] += direction_shift * output_size * face_dir
    
    return M


def transform_geometry_2d(geom, M_inv, aligned_size, original_w, original_h):
    """
    Transform 2D coordinates (u, v) from aligned space back to original image space.
    
    Args:
        geom: [N, 5] array (x, y, z, u, v)
        M_inv: [2, 3] inverse affine matrix
        aligned_size: size of aligned image
        original_w, original_h: original image dimensions
    """
    result = geom.copy()
    
    # Extract 2D coords (u, v) from normalized [0, 1] space
    uv_norm = geom[:, 3:5]
    
    # Convert to aligned pixel space
    uv_px = uv_norm * float(aligned_size)
    
    # Apply inverse transform
    ones = np.ones((len(uv_px), 1), dtype=np.float32)
    uv_h = np.concatenate([uv_px, ones], axis=1)  # [N, 3]
    uv_orig_px = (M_inv @ uv_h.T).T  # [N, 2]
    
    # Normalize to original image space [0, 1]
    result[:, 3] = uv_orig_px[:, 0] / float(original_w)
    result[:, 4] = uv_orig_px[:, 1] / float(original_h)
    
    return result


def save_obj(filepath, vertices, topology, restore_indices=None, vertex_colors=None):
    """Save 3D mesh to OBJ file. If provided, writes per-vertex RGB as `v x y z r g b`."""
    
    # Restore full landmarks if mapping is provided
    if restore_indices is not None:
        vertices = vertices[restore_indices]
        if vertex_colors is not None:
            vertex_colors = vertex_colors[restore_indices]
        
    with open(filepath, 'w') as f:
        f.write(f"# Prediction\n")
        
        # Write vertices
        if vertex_colors is not None:
            vc = np.asarray(vertex_colors, dtype=np.float32)
            vc = np.clip(vc, 0.0, 1.0)
            if vc.shape[0] != len(vertices):
                raise ValueError(f"vertex_colors count {vc.shape[0]} != vertices count {len(vertices)}")
            for v, c in zip(vertices, vc):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        else:
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write faces
        for name, data in topology.items():
            start_idx = data['start_idx']
            faces = data['faces']
            
            f.write(f"g {name}\n")
            for face in faces:
                # OBJ indices are 1-based
                indices_str = " ".join([str(idx + start_idx + 1) for idx in face])
                f.write(f"f {indices_str}\n")

def run_inference(args):
    if args.device == 'cuda':
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device(args.device)
        
    print(f"馃殌 Running inference on {device}")
    
    # 1. Load Data & Topology
    (num_landmarks, num_mesh, template_landmark, template_mesh,
     landmark2keypoint_idx, landmark2keypoint_w,
     mesh2landmark_idx, mesh2landmark_w,
     n_keypoint, landmark_restore_indices, mesh_restore_indices,
     template_mesh_uv, template_mesh_uv_full, template_mesh_faces, template_mesh_faces_full) = load_aux_data(device)
    
    landmark_topology = load_landmark_topology()
    mesh_topology = load_mesh_topology()
    
    # 2. Initialize Model
    print(f"馃 Loading model from {args.model_path}...")
    model = GeometryTransformer(
        num_landmarks=num_landmarks,
        num_mesh=num_mesh,
        template_landmark=template_landmark,
        template_mesh=template_mesh,
        landmark2keypoint_knn_indices=landmark2keypoint_idx,
        landmark2keypoint_knn_weights=landmark2keypoint_w,
        mesh2landmark_knn_indices=mesh2landmark_idx,
        mesh2landmark_knn_weights=mesh2landmark_w,
        n_keypoint=n_keypoint,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        backbone_weights=args.backbone_weights,
        model_type=args.model_type,
        k_bins=args.k_bins,
        simdr_range_3d=(args.simdr_min_3d, args.simdr_max_3d),
        simdr_range_2d=(args.simdr_min_2d, args.simdr_max_2d),
        use_deformable_attention=(args.use_deformable_attention if args.model_type == 'simdr' else False),
        num_deformable_points=args.num_deformable_points,
        template_mesh_uv=template_mesh_uv,
        template_mesh_uv_full=template_mesh_uv_full,
        template_mesh_faces=template_mesh_faces,
        template_mesh_faces_full=template_mesh_faces_full,
        mesh_restore_indices=mesh_restore_indices,
    ).to(device)
    
    # Load Weights
    if not os.path.exists(args.model_path):
        print(f"鉂?Model file not found: {args.model_path}")
        return

    ckpt = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']

    # Handle checkpoints saved from wrapped DDP modules.
    if any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    # 3. Process Image(s)
    if os.path.isdir(args.image_path):
        image_files = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_files = [args.image_path]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # Load Image
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"鈿狅笍 Could not read {img_path}")
                continue
                
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]
            
            # Step 1: Detect face keypoints
            five_pts_px, success = detect_face_keypoints(img_path)
            
            if not success or five_pts_px is None:
                print(f"鈿狅笍 Skipping {img_path} - face detection failed")
                continue
            
            # Save original image with 5 keypoints drawn
            basename = os.path.splitext(os.path.basename(img_path))[0]
            img_with_kp = img_rgb.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            labels = ['R_Eye', 'L_Eye', 'Nose', 'R_Mouth', 'L_Mouth']
            
            # First, try to draw all 68 dlib landmarks if available
            if DLIB_AVAILABLE and _dlib_detector is not None and _dlib_predictor is not None:
                try:
                    import dlib
                    img_bgr_copy = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(img_bgr_copy, cv2.COLOR_BGR2GRAY)
                    faces = _dlib_detector(gray, 1)
                    if len(faces) > 0:
                        shape = _dlib_predictor(gray, faces[0])
                        # Draw all 68 points in light gray
                        for i in range(68):
                            pt = (shape.part(i).x, shape.part(i).y)
                            cv2.circle(img_with_kp, pt, 2, (180, 180, 180), -1)
                            cv2.putText(img_with_kp, str(i), (pt[0]+3, pt[1]-3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
                except Exception as e:
                    print(f"鈿狅笍 Failed to draw 68 points: {e}")
            
            # Then draw the 5 key points on top
            for i, (pt, color, label) in enumerate(zip(five_pts_px, colors, labels)):
                cv2.circle(img_with_kp, (int(pt[0]), int(pt[1])), 7, color, -1)
                cv2.circle(img_with_kp, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)
                cv2.putText(img_with_kp, label, (int(pt[0])+12, int(pt[1])-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out_kp_orig_path = os.path.join(args.output_dir, f"{basename}_keypoints_original.png")
            cv2.imwrite(out_kp_orig_path, cv2.cvtColor(img_with_kp, cv2.COLOR_RGB2BGR))
            
            
            # Step 2: Compute alignment matrix
            input_size = 512
            M = estimate_alignment_matrix(
                five_pts_px,
                w_orig,
                h_orig,
                output_size=input_size,
                output_scale=args.align_output_scale,
                direction_shift=args.align_direction_shift
            )
            
            # Step 3: Align image to canonical pose
            img_aligned = cv2.warpAffine(
                img_rgb,
                M,
                (input_size, input_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            # Save aligned image with transformed keypoints drawn
            img_aligned_vis = img_aligned.copy()
            
            # Transform the original detected keypoints through M to see where they ended up
            ones = np.ones((len(five_pts_px), 1), dtype=np.float32)
            pts_h = np.concatenate([five_pts_px, ones], axis=1)  # [5, 3]
            transformed_pts = (M @ pts_h.T).T  # [5, 2]
            
            # Draw the transformed keypoints on aligned image
            for i, (pt, color, label) in enumerate(zip(transformed_pts, colors, labels)):
                cv2.circle(img_aligned_vis, (int(pt[0]), int(pt[1])), 7, color, -1)
                cv2.circle(img_aligned_vis, (int(pt[0]), int(pt[1])), 9, (255, 255, 255), 2)
                cv2.putText(img_aligned_vis, label, (int(pt[0])+12, int(pt[1])-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out_kp_aligned_path = os.path.join(args.output_dir, f"{basename}_keypoints_aligned.png")
            cv2.imwrite(out_kp_aligned_path, cv2.cvtColor(img_aligned_vis, cv2.COLOR_RGB2BGR))
            
            # To Tensor
            rgb_tensor = torch.from_numpy(img_aligned).permute(2, 0, 1).float() / 255.0
            
            # Normalize (ImageNet)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_tensor = (rgb_tensor - mean) / std
            
            rgb_tensor = rgb_tensor.unsqueeze(0).to(device) # [1, 3, 512, 512]
            
            # Inference
            with torch.no_grad():
                if args.model_type == 'simdr':
                    outputs = model(
                        rgb_tensor,
                        return_logits_mask=[False] * args.num_layers,
                        predict_layer_mask=([False] * (args.num_layers - 1) + [True]),
                    )
                else:
                    outputs = model(rgb_tensor)
                final_output = outputs[-1]
                
                lm_pred = final_output['landmark'].detach().cpu().numpy().reshape(rgb_tensor.shape[0], -1, 6)
                mesh_pred = final_output['mesh'].detach().cpu().numpy().reshape(rgb_tensor.shape[0], -1, 6)
                mesh_color_pred = final_output.get('mesh_color', None)
                mesh_texture_pred = final_output.get('mesh_texture', None)
                if mesh_color_pred is not None:
                    mesh_color_pred = mesh_color_pred.detach().cpu().numpy().reshape(rgb_tensor.shape[0], -1, 3)
                if mesh_texture_pred is not None:
                    mesh_texture_pred = mesh_texture_pred.detach().cpu().numpy()
            
            # 4. Transform predictions back to original image space
            # Process first (and only) batch element
            lm_pred = lm_pred[0]  # [N, 5]
            mesh_pred = mesh_pred[0]  # [M, 5]
            if mesh_color_pred is not None:
                mesh_color_pred = mesh_color_pred[0]  # [M, 3]
            if mesh_texture_pred is not None:
                mesh_texture_pred = mesh_texture_pred[0]  # [3, Ht, Wt]
            
            # Compute inverse transform
            M_inv = cv2.invertAffineTransform(M)
            
            # Transform 2D coordinates back to original space
            lm_pred_orig = transform_geometry_2d(lm_pred, M_inv, input_size, w_orig, h_orig)
            mesh_pred_orig = transform_geometry_2d(mesh_pred, M_inv, input_size, w_orig, h_orig)
            
            # 5. Visualization & Saving (use original image, not aligned)
            vis_size = 1024
            img_vis = cv2.resize(img_rgb, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            
            # Process 2D coordinates for visualization
            # For landmarks
            current_lm = lm_pred_orig.copy()
            current_lm_2d = current_lm[:, 3:5]  # Use u, v for 2D
            current_lm_2d = np.clip(current_lm_2d, 0.0, 1.0)
            if landmark_restore_indices is not None:
                current_lm_2d = current_lm_2d[landmark_restore_indices]
            current_lm_2d[:, 0] *= vis_size
            current_lm_2d[:, 1] *= vis_size
            
            # For mesh
            current_mesh = mesh_pred_orig.copy()
            current_mesh_2d = current_mesh[:, 3:5]  # Use u, v for 2D
            current_mesh_2d = np.clip(current_mesh_2d, 0.0, 1.0)
            if mesh_restore_indices is not None:
                current_mesh_2d = current_mesh_2d[mesh_restore_indices]
            current_mesh_2d[:, 0] *= vis_size
            current_mesh_2d[:, 1] *= vis_size
            
            # Process 3D coordinates for OBJ export
            # For landmarks (use original space predictions)
            lm_verts = lm_pred_orig[:, :3]  # x, y, z
            if landmark_restore_indices is not None:
                lm_verts = lm_verts[landmark_restore_indices]
            
            # For mesh
            mesh_verts = mesh_pred_orig[:, :3]  # x, y, z
            mesh_vertex_colors = None
            if mesh_color_pred is not None:
                mesh_vertex_colors = np.clip(mesh_color_pred.astype(np.float32), 0.0, 1.0)
            
            if mesh_restore_indices is not None:
                mesh_verts = mesh_verts[mesh_restore_indices]
                if mesh_vertex_colors is not None:
                    mesh_vertex_colors = mesh_vertex_colors[mesh_restore_indices]
            
            # Draw Overlays
            overlay_mesh = create_combined_overlay(img_vis, current_mesh_2d, mesh_topology)
            overlay_landmark = create_combined_overlay(img_vis, current_lm_2d, landmark_topology)
            
            # Save Mesh Overlay
            out_mesh_img_path = os.path.join(args.output_dir, f"{basename}_mesh_vis.png")
            cv2.imwrite(out_mesh_img_path, cv2.cvtColor(overlay_mesh, cv2.COLOR_RGB2BGR))
            
            # Save Landmark Overlay
            out_lm_img_path = os.path.join(args.output_dir, f"{basename}_landmark_vis.png")
            cv2.imwrite(out_lm_img_path, cv2.cvtColor(overlay_landmark, cv2.COLOR_RGB2BGR))
            
            # Save 3D OBJs (vertices are already restored, so don't restore again)
            out_lm_obj_path = os.path.join(args.output_dir, f"{basename}_landmark.obj")
            save_obj(out_lm_obj_path, lm_verts, landmark_topology, restore_indices=None)
            
            out_mesh_obj_path = os.path.join(args.output_dir, f"{basename}_mesh.obj")
            save_obj(out_mesh_obj_path, mesh_verts, mesh_topology, restore_indices=None, vertex_colors=mesh_vertex_colors)

            if mesh_texture_pred is not None:
                texture_img = np.transpose(mesh_texture_pred, (1, 2, 0)).astype(np.float32, copy=False)
                texture_img = np.nan_to_num(texture_img, nan=0.0, posinf=1.0, neginf=0.0)
                texture_img = np.clip(texture_img, 0.0, 1.0)
                texture_u8 = (texture_img * 255.0).astype(np.uint8)
                out_texture_path = os.path.join(args.output_dir, f"{basename}_pred_texture.png")
                cv2.imwrite(out_texture_path, cv2.cvtColor(texture_u8, cv2.COLOR_RGB2BGR))

                render_img, _ = render_mesh_texture_from_2d_pred(
                    mesh_pred=mesh_pred_orig,
                    mesh_texture=texture_img,
                    template_mesh_uv=template_mesh_uv,
                    template_mesh_faces=template_mesh_faces,
                    out_h=h_orig,
                    out_w=w_orig,
                    device=device,
                    flip_uv_v=True,
                )
                if render_img is not None:
                    render_u8 = (np.clip(render_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                    out_render_path = os.path.join(args.output_dir, f"{basename}_pred_render.png")
                    cv2.imwrite(out_render_path, cv2.cvtColor(render_u8, cv2.COLOR_RGB2BGR))
                else:
                    print(f"[Warn] Skip render output for {basename} (nvdiffrast unavailable or render failed).")
            
            print(f"鉁?Processed {basename}")
            
        except Exception as e:
            print(f"鉂?Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Geometry Transformer')
    parser.add_argument('--image_path', type=str, default='samples', help='Path to image or directory')
    parser.add_argument('--model_path', type=str, default='artifacts/checkpoints/best_geometry_transformer_dim6.pth', help='Path to .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default='artifacts/test_result_geometry', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (if device is cuda)')
    
    # Model args (must match training)
    parser.add_argument('--d_model', type=int, default=512, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--backbone_weights', type=str, default='imagenet', help='Backbone weights type')
    parser.add_argument('--model_type', type=str, default='simdr', choices=['regression', 'simdr'], help='Model type: regression or simdr')

    # SimDR model args (must match training when --model_type simdr)
    parser.add_argument('--k_bins', type=int, default=256, help='Number of bins for SimDR')
    parser.add_argument('--simdr_min_3d', type=float, default=-0.5, help='SimDR 3D offset min range')
    parser.add_argument('--simdr_max_3d', type=float, default=0.5, help='SimDR 3D offset max range')
    parser.add_argument('--simdr_min_2d', type=float, default=-0.5, help='SimDR 2D offset min range')
    parser.add_argument('--simdr_max_2d', type=float, default=0.5, help='SimDR 2D offset max range')
    parser.add_argument('--num_deformable_points', type=int, default=16, help='Deformable attention sampling points per head per level (SimDR model)')
    parser.add_argument('--use_deformable_attention', dest='use_deformable_attention', action='store_true', help='Enable deformable cross-attention in SimDR transformer')
    parser.add_argument('--no_deformable_attention', dest='use_deformable_attention', action='store_false', help='Disable deformable cross-attention and use standard decoder attention')
    parser.set_defaults(use_deformable_attention=True)
    
    # Face alignment args (should match training if --align_5pt was used)
    parser.add_argument('--align_output_scale', type=float, default=0.75, help='Post-alignment center scale')
    parser.add_argument('--align_direction_shift', type=float, default=0.08, help='Extra shift toward face direction')

    args = parser.parse_args()
    
    # Initialize dlib detector
    if not _init_dlib_detector():
        print("鉂?dlib detector initialization failed.")
        print("   Install dlib: pip install dlib")
        print("   Download model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        exit(1)
    
    print("鉁?Using dlib for accurate facial landmark detection (no TensorFlow dependency)")
    
    run_inference(args)
