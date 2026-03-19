import os
import random
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from obj_load_helper import load_uv_obj_file
try:
    import nvdiffrast.torch as dr
    _NVDIFFRAST_AVAILABLE = True
except Exception:
    dr = None
    _NVDIFFRAST_AVAILABLE = False

# Static combined UV layout (same as test_combine_uv_layout.py and dataset compose).
EYE_BOX_SIZE = 0.20
EYE_L_U_START = 0.005
EYE_R_U_START = 0.795
BOTTOM_MARGIN = 0.005

MOUTH_SPLIT_V = 0.61
MOUTH_BOX_SIZE = 0.25
MOUTH_V_GT_U_START = 0.50
MOUTH_V_LE_U_START = 0.22

HEAD_SCALE = 1.0
HEAD_TX = 0.0
HEAD_TY = 0.0

_MESH_UV_COMBINED_CACHE = {}


def _transform_uv_center(uv: np.ndarray, scale: float, tx: float, ty: float) -> np.ndarray:
    center = uv.mean(axis=0, keepdims=True)
    out = (uv - center) * float(scale) + center
    out[:, 0] += float(tx)
    out[:, 1] += float(ty)
    return out.astype(np.float32, copy=False)


def _place_uv_in_box(uv: np.ndarray, u_start: float, v_start: float, box_size: float, align_bottom: bool) -> np.ndarray:
    if uv.shape[0] == 0:
        return uv.copy().astype(np.float32, copy=False)

    uv_min = uv.min(axis=0)
    uv_max = uv.max(axis=0)
    span = np.maximum(uv_max - uv_min, 1e-8)
    scale = float(box_size) / float(max(span[0], span[1]))

    uv_local = (uv - uv_min) * scale
    local_min = uv_local.min(axis=0)
    local_max = uv_local.max(axis=0)
    local_size = local_max - local_min

    tx = float(u_start) - float(local_min[0]) + 0.5 * (float(box_size) - float(local_size[0]))
    if align_bottom:
        ty = float(v_start) - float(local_min[1])
    else:
        ty = float(v_start) - float(local_min[1]) + 0.5 * (float(box_size) - float(local_size[1]))

    out = uv_local.copy()
    out[:, 0] += tx
    out[:, 1] += ty
    return out.astype(np.float32, copy=False)


def _load_combined_mesh_uv(model_dir: str = "model") -> np.ndarray:
    cache_key = os.path.abspath(model_dir)
    if cache_key in _MESH_UV_COMBINED_CACHE:
        return _MESH_UV_COMBINED_CACHE[cache_key]

    part_files = {
        "head": "mesh_head.obj",
        "eye_l": "mesh_eye_l.obj",
        "eye_r": "mesh_eye_r.obj",
        "mouth": "mesh_mouth.obj",
    }

    part_uv = {}
    for part in ["head", "eye_l", "eye_r", "mouth"]:
        obj_path = os.path.join(model_dir, part_files[part])
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Missing OBJ for UV load: {obj_path}")
        verts, uvs, _, _, _, _ = load_uv_obj_file(obj_path, triangulate=False)
        if uvs is None:
            raise ValueError(f"OBJ has no UVs: {obj_path}")
        if verts is not None and len(verts) != len(uvs):
            raise ValueError(
                f"Vertex/UV count mismatch in {obj_path}: verts={len(verts)}, uvs={len(uvs)}"
            )
        part_uv[part] = np.asarray(uvs[:, :2], dtype=np.float32)

    head_uv = _transform_uv_center(part_uv["head"], HEAD_SCALE, HEAD_TX, HEAD_TY)
    eye_l_uv = _place_uv_in_box(
        part_uv["eye_l"],
        u_start=EYE_L_U_START,
        v_start=BOTTOM_MARGIN,
        box_size=EYE_BOX_SIZE,
        align_bottom=False,
    )
    eye_r_uv = _place_uv_in_box(
        part_uv["eye_r"],
        u_start=EYE_R_U_START,
        v_start=BOTTOM_MARGIN,
        box_size=EYE_BOX_SIZE,
        align_bottom=False,
    )

    mouth_uv_src = part_uv["mouth"]
    mouth_uv = mouth_uv_src.copy()
    mouth_high_mask = mouth_uv_src[:, 1] > float(MOUTH_SPLIT_V)
    mouth_low_mask = ~mouth_high_mask
    if np.any(mouth_high_mask):
        mouth_uv[mouth_high_mask] = _place_uv_in_box(
            mouth_uv_src[mouth_high_mask],
            u_start=MOUTH_V_GT_U_START,
            v_start=BOTTOM_MARGIN,
            box_size=MOUTH_BOX_SIZE,
            align_bottom=True,
        )
    if np.any(mouth_low_mask):
        mouth_uv[mouth_low_mask] = _place_uv_in_box(
            mouth_uv_src[mouth_low_mask],
            u_start=MOUTH_V_LE_U_START,
            v_start=BOTTOM_MARGIN,
            box_size=MOUTH_BOX_SIZE,
            align_bottom=True,
        )

    uv_combined = np.concatenate([head_uv, eye_l_uv, eye_r_uv, mouth_uv], axis=0).astype(np.float32, copy=False)
    uv_combined = np.clip(uv_combined, 0.0, 1.0)
    _MESH_UV_COMBINED_CACHE[cache_key] = uv_combined
    return uv_combined


def load_combined_mesh_uv(model_dir: str = "model", copy: bool = True) -> np.ndarray:
    """
    Public helper to load the static combined UV layout used by texture GT composition.
    """
    uv = _load_combined_mesh_uv(model_dir=model_dir)
    if copy:
        return uv.copy()
    return uv


def draw_uv_points_on_texture(
    texture_rgb: np.ndarray,
    uv_coords: np.ndarray,
    mesh_topology: dict | None = None,
    point_radius: int = 1,
    point_step: int = 1,
) -> np.ndarray:
    """
    Draw UV points on RGB texture for mapping inspection.
    Uses the same UV->pixel convention as model rasterization (flip V).
    """
    if texture_rgb is None or uv_coords is None:
        raise ValueError("texture_rgb and uv_coords are required.")
    if texture_rgb.ndim != 3 or texture_rgb.shape[2] != 3:
        raise ValueError(f"texture_rgb must be [H,W,3], got {texture_rgb.shape}")

    out = np.ascontiguousarray(texture_rgb.copy())
    h, w = out.shape[:2]
    uv = np.asarray(uv_coords, dtype=np.float32)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv_coords must be [N,2], got {uv.shape}")

    uv = np.clip(uv, 0.0, 1.0)
    x = np.rint(uv[:, 0] * float(max(w - 1, 1))).astype(np.int32)
    y = np.rint((1.0 - uv[:, 1]) * float(max(h - 1, 1))).astype(np.int32)
    step = max(1, int(point_step))

    # BGR for OpenCV drawing.
    color_by_part = {
        "head": (192, 192, 192),
        "eye_l": (60, 220, 60),
        "eye_r": (60, 60, 220),
        "mouth": (220, 60, 60),
    }

    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if mesh_topology is not None:
        for part_name, part in mesh_topology.items():
            start = int(part.get("start_idx", 0))
            count = int(part.get("count", 0))
            if count <= 0:
                continue
            end = min(start + count, uv.shape[0])
            color = color_by_part.get(part_name, (255, 255, 255))
            for px, py in zip(x[start:end:step], y[start:end:step]):
                cv2.circle(out_bgr, (int(px), int(py)), int(point_radius), color, thickness=-1, lineType=cv2.LINE_AA)
    else:
        for px, py in zip(x[::step], y[::step]):
            cv2.circle(out_bgr, (int(px), int(py)), int(point_radius), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    # Mark vertex 0 for quick sanity check.
    if len(x) > 0:
        cv2.circle(out_bgr, (int(x[0]), int(y[0])), max(2, int(point_radius) + 1), (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def derive_depth_from_3d_to_2d(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Derive view-space depth by fitting an affine projection from 3D to 2D.

    Solves for a 3x4 affine matrix P such that:
        [u, v, depth]^T ≈ P @ [x, y, z, 1]^T
    using least squares on the u,v correspondences, then uses the
    fitted transform to produce a consistent depth per vertex.

    Args:
        xyz: [N, 3] predicted 3D coordinates.
        uv:  [N, 2] predicted 2D screen coordinates (in [0,1]).

    Returns:
        depth: [N] normalised depth in [0, 1], suitable for rasterisation z-order.
    """
    N = xyz.shape[0]
    if N < 4:
        return np.zeros(N, dtype=np.float32)

    # Build [N, 4] homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float64)
    A = np.hstack([xyz.astype(np.float64), ones])  # [N, 4]

    # Solve for rows 1 & 2 of P via least squares: uv = A @ P12^T
    uv64 = uv.astype(np.float64)
    P12, _, _, _ = np.linalg.lstsq(A, uv64, rcond=None)  # [4, 2]

    # Residual — how well the affine model fits
    uv_fit = A @ P12  # [N, 2]
    residual = np.sqrt(((uv64 - uv_fit) ** 2).sum(axis=1).mean())

    # Build the 3rd row (depth) orthogonal to the first two rows.
    # The first two rows of P span a 2D subspace in 4D; the depth direction
    # should be the component of the z-axis that is orthogonal to this subspace.
    r1 = P12[:3, 0]  # first 3 elements of row 1
    r2 = P12[:3, 1]  # first 3 elements of row 2
    # Cross product gives a direction perpendicular to both rows in 3D
    depth_dir = np.cross(r1, r2)
    depth_norm = np.linalg.norm(depth_dir)
    if depth_norm < 1e-12:
        return np.full(N, 0.5, dtype=np.float32)
    depth_dir = depth_dir / depth_norm

    # Project 3D coords onto the depth direction
    raw_depth = xyz.astype(np.float64) @ depth_dir  # [N]

    # Normalise to [0, 1]
    d_min, d_max = raw_depth.min(), raw_depth.max()
    d_range = d_max - d_min
    if d_range < 1e-8:
        return np.full(N, 0.5, dtype=np.float32)

    depth = ((raw_depth - d_min) / d_range).astype(np.float32)
    return np.clip(depth, 0.0, 1.0)


def derive_depth_from_3d_to_2d_torch(xyz: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Batched torch version of derive_depth_from_3d_to_2d.

    Args:
        xyz: [B, N, 3] predicted 3D coordinates.
        uv:  [B, N, 2] predicted 2D screen coordinates.

    Returns:
        depth: [B, N] normalised depth in [0, 1].
    """
    B, N, _ = xyz.shape
    device = xyz.device
    dtype = xyz.dtype

    # Build homogeneous coords [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=torch.float32)
    A = torch.cat([xyz.float(), ones], dim=-1)  # [B, N, 4]
    uv_f = uv.float()  # [B, N, 2]

    # Solve least squares per batch: A @ P = uv  =>  P = (A^T A)^{-1} A^T uv
    AtA = torch.bmm(A.transpose(1, 2), A)  # [B, 4, 4]
    Atb = torch.bmm(A.transpose(1, 2), uv_f)  # [B, 4, 2]
    try:
        P = torch.linalg.solve(AtA, Atb)  # [B, 4, 2]
    except Exception:
        return torch.full((B, N), 0.5, device=device, dtype=dtype)

    # Extract the 3D part of the two rows
    r1 = P[:, :3, 0]  # [B, 3]
    r2 = P[:, :3, 1]  # [B, 3]

    # Cross product gives depth direction orthogonal to both projection rows
    depth_dir = torch.cross(r1, r2, dim=-1)  # [B, 3]
    depth_norm = depth_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    depth_dir = depth_dir / depth_norm  # [B, 3]

    # Project 3D coords onto depth direction
    raw_depth = torch.bmm(xyz.float(), depth_dir.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # Normalise to [0, 1] per batch
    d_min = raw_depth.min(dim=1, keepdim=True).values
    d_max = raw_depth.max(dim=1, keepdim=True).values
    d_range = (d_max - d_min).clamp(min=1e-8)
    depth = ((raw_depth - d_min) / d_range).clamp(0.0, 1.0)

    return depth.to(dtype=dtype)


def render_mesh_texture_from_2d_pred(
    mesh_pred: np.ndarray,
    mesh_texture: np.ndarray,
    template_mesh_uv: np.ndarray,
    template_mesh_faces: np.ndarray,
    out_h: int,
    out_w: int,
    device: torch.device | str | None = None,
    flip_uv_v: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Render RGB image from predicted 2D mesh coordinates and predicted UV texture.
    Returns (render_rgb, coverage), each in float32 [0,1], or (None, None) on failure.
    """
    if not _NVDIFFRAST_AVAILABLE:
        return None, None

    if mesh_pred is None or mesh_texture is None:
        return None, None

    mesh_pred = np.asarray(mesh_pred, dtype=np.float32)
    if mesh_pred.ndim != 2 or mesh_pred.shape[1] < 5:
        return None, None

    uv = np.asarray(template_mesh_uv, dtype=np.float32)
    tri = np.asarray(template_mesh_faces, dtype=np.int32)
    if uv.ndim != 2 or uv.shape[1] != 2 or tri.ndim != 2 or tri.shape[1] != 3:
        return None, None
    if uv.shape[0] != mesh_pred.shape[0]:
        return None, None

    tex = np.asarray(mesh_texture, dtype=np.float32)
    if tex.ndim != 3:
        return None, None
    if tex.shape[0] == 3 and tex.shape[2] != 3:
        tex = np.transpose(tex, (1, 2, 0))
    if tex.shape[2] != 3:
        return None, None
    tex = np.nan_to_num(tex, nan=0.0, posinf=1.0, neginf=0.0)
    tex = np.clip(tex, 0.0, 1.0)

    if device is None:
        if not torch.cuda.is_available():
            return None, None
        dev = torch.device("cuda:0")
    else:
        dev = torch.device(device)
    if dev.type != "cuda":
        return None, None

    xy = mesh_pred[:, 3:5].astype(np.float32)
    clip_pos = np.zeros((xy.shape[0], 4), dtype=np.float32)
    clip_pos[:, 0] = xy[:, 0] * 2.0 - 1.0
    clip_pos[:, 1] = 1.0 - (xy[:, 1] * 2.0)
    clip_pos[:, :2] = np.nan_to_num(clip_pos[:, :2], nan=0.0, posinf=2.0, neginf=-2.0)

    # Derive depth from 3D→2D alignment instead of using predicted depth
    depth_scale = 0.8
    depth_bias = 0.1
    if mesh_pred.shape[1] >= 5:
        xyz_3d = mesh_pred[:, :3]
        uv_2d = mesh_pred[:, 3:5]
        derived_depth = derive_depth_from_3d_to_2d(xyz_3d, uv_2d)
        clip_pos[:, 2] = -(derived_depth * depth_scale + depth_bias)
    else:
        clip_pos[:, 2] = 0.0
    clip_pos[:, 3] = 1.0

    pos_t = torch.from_numpy(clip_pos).to(dev)[None, ...].contiguous()
    tri_t = torch.from_numpy(tri).to(device=dev, dtype=torch.int32).contiguous()
    uv_t = torch.from_numpy(np.clip(uv, 0.0, 1.0)).to(dev)[None, ...].contiguous()
    tex_t = torch.from_numpy(tex).to(dev)[None, ...].contiguous()

    render_h = int(out_h) * 2
    render_w = int(out_w) * 2

    try:
        ctx = dr.RasterizeCudaContext(device=dev)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            rast, _ = dr.rasterize(ctx, pos_t, tri_t, resolution=[render_h, render_w])
            uv_pix, _ = dr.interpolate(uv_t.float(), rast, tri_t)
            if flip_uv_v:
                uv_pix = torch.stack([uv_pix[..., 0], 1.0 - uv_pix[..., 1]], dim=-1)
            uv_pix = uv_pix.clamp(0.0, 1.0)

            color = dr.texture(tex_t.float(), uv_pix, filter_mode="linear", boundary_mode="clamp")
            color = dr.antialias(color, rast, pos_t, tri_t)
            cov = (rast[..., 3:4] > 0).to(dtype=color.dtype)
            color = color * cov

        color = color.permute(0, 3, 1, 2).contiguous()
        cov = cov.permute(0, 3, 1, 2).contiguous()
        color = F.interpolate(color, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
        cov = F.interpolate(cov, size=(int(out_h), int(out_w)), mode='bilinear', align_corners=False)
        color = torch.flip(color, dims=[2])
        cov = torch.flip(cov, dims=[2])

        render = color[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        coverage = cov[0, 0].detach().cpu().numpy().astype(np.float32)
        return render, coverage
    except Exception:
        return None, None

# --- Landmark Visualization Helpers ---

def load_landmark_topology():
    """Load the landmark 3D model topology for visualization."""
    models = {
        'head': 'model/landmark_head.obj',
        'eye_l': 'model/landmark_eye_l.obj',
        'eye_r': 'model/landmark_eye_r.obj',
        'mouth': 'model/landmark_mouth.obj'
    }
    
    topology = {}
    total_verts = 0
    
    # Colors (B, G, R)
    colors = {
        'head': (200, 200, 200),
        'eye_l': (0, 255, 0),
        'eye_r': (0, 255, 0),
        'mouth': (0, 0, 255)
    }
    
    # Load each part
    # Assumed order: head, eye_l, eye_r, mouth
    for name in ['head', 'eye_l', 'eye_r', 'mouth']:
        path = models[name]
        if os.path.exists(path):
            # Load with triangulate=False to keep quads
            verts, _, _, faces, _, _ = load_uv_obj_file(path, triangulate=False)
            num_verts = len(verts)
            
            topology[name] = {
                'faces': faces,
                'start_idx': total_verts,
                'count': num_verts,
                'color': colors[name]
            }
            total_verts += num_verts
            print(f"Loaded {name}: {num_verts} verts, {len(faces)} faces (Offset: {topology[name]['start_idx']})")
        else:
            print(f"Warning: Model file not found: {path}")
            
    print(f"Total landmark topology vertices: {total_verts}")
    return topology


def load_mesh_topology():
    """Load the mesh 3D model topology for visualization."""
    models = {
        'head': 'model/mesh_head.obj',
        'eye_l': 'model/mesh_eye_l.obj',
        'eye_r': 'model/mesh_eye_r.obj',
        'mouth': 'model/mesh_mouth.obj'
    }
    
    topology = {}
    total_verts = 0
    
    # Colors (B, G, R)
    colors = {
        'head': (180, 180, 180),
        'eye_l': (0, 200, 0),
        'eye_r': (0, 200, 0),
        'mouth': (0, 0, 200)
    }
    
    # Load each part
    for name in ['head', 'eye_l', 'eye_r', 'mouth']:
        path = models[name]
        if os.path.exists(path):
            verts, _, _, faces, _, _ = load_uv_obj_file(path, triangulate=False)
            num_verts = len(verts)
            
            topology[name] = {
                'faces': faces,
                'start_idx': total_verts,
                'count': num_verts,
                'color': colors[name]
            }
            total_verts += num_verts
            print(f"Loaded mesh {name}: {num_verts} verts, {len(faces)} faces (Offset: {topology[name]['start_idx']})")
        else:
            print(f"Warning: Mesh model file not found: {path}")
            
    print(f"Total mesh topology vertices: {total_verts}")
    return topology


def load_topology():
    """Load the landmark 3D model topology (backward compatibility)."""
    return load_landmark_topology()

def create_combined_overlay(rgb_image, landmarks, topology):
    """
    Draw wireframe overlay on RGB image.
    rgb_image: numpy array (H, W, 3) in uint8 [0, 255] (RGB format, will convert to BGR for cv2)
    landmarks: numpy array (N, 2) in pixel coordinates
    topology: dict of mesh info
    """
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    overlay = img.copy()
    
    for name, data in topology.items():
        start = data['start_idx']
        count = data['count']
        faces = data['faces']
        color = data['color']
        
        # Get relevant landmarks for this part
        # Check bounds
        if start + count <= len(landmarks):
            part_landmarks = landmarks[start : start+count]
            
            for face in faces:
                pts = []
                valid = True
                for idx in face:
                    if idx < len(part_landmarks):
                        # Landmark coords
                        px, py = part_landmarks[idx][0], part_landmarks[idx][1]
                        
                        # Check for NaN/Inf
                        if not np.isfinite(px) or not np.isfinite(py):
                            valid = False; break
                            
                        x, y = int(px), int(py)
                        # Loose bounds check for drawing
                        if -1000 <= x < 10000 and -1000 <= y < 10000:
                            pts.append((x, y))
                        else:
                            valid = False; break
                    else:
                        valid = False; break
                
                # Draw edges for any polygon type (triangles, quads, n-gons)
                if valid and len(pts) >= 3:
                    num_pts = len(pts)
                    for i in range(num_pts):
                        pt1 = pts[i]
                        pt2 = pts[(i + 1) % num_pts]
                        cv2.line(overlay, pt1, pt2, color, 1, cv2.LINE_AA)
    
    # Blend
    alpha = 0.6
    combined = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out

def save_landmark_visualizations(landmark_model, batch, epoch, device, output_dir, topology, output_dim=2, restore_indices=None):
    """Save visualizations including wireframe overlay."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Use up to 8 samples
    max_samples = 8
    
    # Run Inference
    landmark_model.eval()
    with torch.no_grad():
        # Resize RGB to model's expected input size (512x512)
        rgb = batch['rgb'][:max_samples]
        rgb = F.interpolate(rgb, size=(512, 512), mode='bilinear', align_corners=True).to(device)
        
        # Predict
        lm_out = landmark_model(rgb)[-1]
        lm_pred = lm_out.detach().cpu().numpy().reshape(lm_out.shape[0], -1, output_dim)
    
    # Create grid
    num_samples = lm_pred.shape[0]
    
    # Prepare images
    images = []
    rgb_tensor = batch['rgb'][:max_samples]
    for i in range(num_samples):
        # ALWAYS use augmented RGB from batch (what the model actually saw during inference)
        # This ensures landmarks align correctly even when augmentations are applied
        img_np = (rgb_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Force resize to 1024x1024 for visualization
        img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        H, W = img_np.shape[:2]
        
        # Draw overlay
        current_lm = lm_pred[i].copy()
        
        # If geometry includes UV (>=5D), use UV for 2D overlay.
        if output_dim >= 5:
            current_lm = current_lm[:, 3:5]
        # If 3D (XYZ), slice X, Y (might not align if world space, but best effort)
        elif output_dim > 2:
            current_lm = current_lm[:, :2]
            
        current_lm = np.clip(current_lm, 0.0, 1.0)
        
        # Restore full landmarks if mapping is provided
        if restore_indices is not None:
            current_lm = current_lm[restore_indices]

        # Coordinate transformation
        # Model predictions are normalized [0, 1] relative to the full image FOV
        # (Since we removed cropping and just resize to 1024/512)
        current_lm[:, 0] *= W
        current_lm[:, 1] *= H
        
        img_vis = create_combined_overlay(img_np, current_lm, topology)
        images.append(img_vis)
    
    # Combine into grid (4 cols)
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    # Calculate grid size
    grid_h = rows * images[0].shape[0]
    grid_w = cols * images[0].shape[1]
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        h, w = img.shape[:2]
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
        
    # Save
    path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_landmark_overlay.png')
    cv2.imwrite(path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"Saved landmark overlay: {path}")

    # Save 3D OBJ for all samples
    for i in range(num_samples):
        obj_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_sample_{i}.obj')
        
        # Get 3D coords
        # lm_pred is [B, N, 5]
        # output_dim 5 -> x, y, z, u, v
        # We need x, y, z
        sample_lm = lm_pred[i] # [N, 5]
        
        vertices = sample_lm[:, :3] # [N, 3]
        
        # Restore full landmarks if mapping is provided
        if restore_indices is not None:
            vertices = vertices[restore_indices]
            
        with open(obj_path, 'w') as f:
            f.write(f"# Epoch {epoch+1} Sample {i} Prediction\n")
            
            # Write vertices
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
                    
        #print(f"Saved 3D mesh: {obj_path}")


def save_geometry_visualizations(geometry_model, batch, epoch, device, output_dir, landmark_topology, mesh_topology, output_dim=5, landmark_restore_indices=None, mesh_restore_indices=None):
    """Save visualizations for GeometryTransformer (both landmark and mesh)."""
    os.makedirs(output_dir, exist_ok=True)
    
    max_samples = 8
    
    # Run Inference
    geometry_model.eval()
    with torch.no_grad():
        rgb = batch['rgb'][:max_samples]
        rgb = F.interpolate(rgb, size=(512, 512), mode='bilinear', align_corners=True).to(device)
        
        # Normalize inputs
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        rgb_normalized = (rgb - mean) / std
        
        # Predict (returns list of dicts)
        outputs = geometry_model(rgb_normalized)
        final_output = outputs[-1]  # Use final layer output
        
        lm_pred = final_output['landmark'].detach().cpu().numpy().reshape(rgb.shape[0], -1, output_dim)
        mesh_pred = final_output['mesh'].detach().cpu().numpy().reshape(rgb.shape[0], -1, output_dim)
        mesh_color_pred = None
        mesh_texture_pred = None
        if 'mesh_color' in final_output and final_output['mesh_color'] is not None:
            mesh_color_pred = final_output['mesh_color'].detach().cpu().numpy().reshape(rgb.shape[0], -1, 3)
        if 'mesh_texture' in final_output and final_output['mesh_texture'] is not None:
            mesh_texture_pred = final_output['mesh_texture'].detach().cpu().numpy()
    
    num_samples = lm_pred.shape[0]
    rgb_tensor = batch['rgb'][:max_samples]
    
    # Save landmark visualizations
    landmark_images = []
    for i in range(num_samples):
        img_np = (rgb_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        H, W = img_np.shape[:2]
        
        current_lm = lm_pred[i].copy()
        if output_dim >= 5:
            current_lm = current_lm[:, 3:5]
        elif output_dim > 2:
            current_lm = current_lm[:, :2]
            
        current_lm = np.clip(current_lm, 0.0, 1.0)
        
        if landmark_restore_indices is not None:
            current_lm = current_lm[landmark_restore_indices]
        
        current_lm[:, 0] *= W
        current_lm[:, 1] *= H
        
        img_vis = create_combined_overlay(img_np, current_lm, landmark_topology)
        landmark_images.append(img_vis)
    
    # Load template data for rendering
    combined_mesh_uv = None
    template_mesh_uv = None
    template_mesh_faces = None
    try:
        combined_mesh_uv = _load_combined_mesh_uv(model_dir="model")
    except Exception as e:
        print(f"Warning: failed to load combined mesh UV layout for OBJ export: {e}")
    try:
        template_mesh_uv = geometry_model.template_mesh_uv.detach().cpu().numpy().astype(np.float32, copy=False)
        template_mesh_faces = geometry_model.template_mesh_faces.detach().cpu().numpy().astype(np.int32, copy=False)
    except Exception as e:
        print(f"Warning: failed to load template mesh UV/faces for render export: {e}")

    # Save mesh visualizations: combined grid with overlay, pred texture, pred render
    vis_size = 512  # size for each cell in the combined grid
    mesh_overlay_images = []
    texture_images = []
    render_images = []
    for i in range(num_samples):
        img_np = (rgb_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        H, W = img_np.shape[:2]

        current_mesh = mesh_pred[i].copy()
        if output_dim >= 5:
            current_mesh = current_mesh[:, 3:5]
        elif output_dim > 2:
            current_mesh = current_mesh[:, :2]

        current_mesh = np.clip(current_mesh, 0.0, 1.0)

        if mesh_restore_indices is not None:
            current_mesh = current_mesh[mesh_restore_indices]

        current_mesh[:, 0] *= W
        current_mesh[:, 1] *= H

        img_vis = create_combined_overlay(img_np, current_mesh, mesh_topology)
        img_vis = cv2.resize(img_vis, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        mesh_overlay_images.append(img_vis)

        # Pred texture
        if mesh_texture_pred is not None:
            tex_img = mesh_texture_pred[i].transpose(1, 2, 0)
            tex_img = np.nan_to_num(tex_img, nan=0.0, posinf=1.0, neginf=0.0)
            tex_img = np.clip(tex_img, 0.0, 1.0)
            tex_u8 = (tex_img * 255.0).astype(np.uint8)
            tex_u8 = cv2.resize(tex_u8, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
            texture_images.append(tex_u8)
        else:
            texture_images.append(np.zeros((vis_size, vis_size, 3), dtype=np.uint8))

        # Pred render
        if mesh_texture_pred is not None and template_mesh_uv is not None and template_mesh_faces is not None:
            tex_for_render = mesh_texture_pred[i].transpose(1, 2, 0)
            tex_for_render = np.nan_to_num(tex_for_render, nan=0.0, posinf=1.0, neginf=0.0)
            tex_for_render = np.clip(tex_for_render, 0.0, 1.0)
            render_img, _ = render_mesh_texture_from_2d_pred(
                mesh_pred=mesh_pred[i],
                mesh_texture=tex_for_render,
                template_mesh_uv=template_mesh_uv,
                template_mesh_faces=template_mesh_faces,
                out_h=vis_size,
                out_w=vis_size,
                device=device,
                flip_uv_v=True,
            )
            if render_img is not None:
                render_u8 = (np.clip(render_img, 0.0, 1.0) * 255.0).astype(np.uint8)
                render_images.append(render_u8)
            else:
                render_images.append(np.zeros((vis_size, vis_size, 3), dtype=np.uint8))
        else:
            render_images.append(np.zeros((vis_size, vis_size, 3), dtype=np.uint8))

    # Landmark grid (4 cols)
    lm_cols = 4
    lm_rows = (num_samples + lm_cols - 1) // lm_cols
    lm_cell_h = landmark_images[0].shape[0]
    lm_cell_w = landmark_images[0].shape[1]
    lm_grid = np.zeros((lm_rows * lm_cell_h, lm_cols * lm_cell_w, 3), dtype=np.uint8)

    for idx, img in enumerate(landmark_images):
        r = idx // lm_cols
        c = idx % lm_cols
        lm_grid[r*lm_cell_h:(r+1)*lm_cell_h, c*lm_cell_w:(c+1)*lm_cell_w] = img

    lm_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_landmark_overlay.png')
    cv2.imwrite(lm_path, cv2.cvtColor(lm_grid, cv2.COLOR_RGB2BGR))
    print(f"Saved landmark overlay: {lm_path}")

    # Combined mesh grid: each row = [overlay | texture | render] per sample
    mesh_cols = 3
    mesh_grid = np.zeros((num_samples * vis_size, mesh_cols * vis_size, 3), dtype=np.uint8)
    for i in range(num_samples):
        mesh_grid[i*vis_size:(i+1)*vis_size, 0*vis_size:1*vis_size] = mesh_overlay_images[i]
        mesh_grid[i*vis_size:(i+1)*vis_size, 1*vis_size:2*vis_size] = texture_images[i]
        mesh_grid[i*vis_size:(i+1)*vis_size, 2*vis_size:3*vis_size] = render_images[i]

    mesh_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_mesh_combined.png')
    cv2.imwrite(mesh_path, cv2.cvtColor(mesh_grid, cv2.COLOR_RGB2BGR))
    print(f"Saved mesh combined grid (overlay|texture|render): {mesh_path}")
    
    # Save 3D OBJ files
    for i in range(num_samples):
        # Landmark OBJ
        lm_obj_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_sample_{i}_landmark.obj')
        lm_verts = lm_pred[i, :, :3]
        
        if landmark_restore_indices is not None:
            lm_verts = lm_verts[landmark_restore_indices]
        
        with open(lm_obj_path, 'w') as f:
            f.write(f"# Epoch {epoch+1} Sample {i} Landmark Prediction\n")
            for v in lm_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for name, data in landmark_topology.items():
                start_idx = data['start_idx']
                faces = data['faces']
                f.write(f"g {name}\n")
                for face in faces:
                    indices_str = " ".join([str(idx + start_idx + 1) for idx in face])
                    f.write(f"f {indices_str}\n")
        
        # Mesh OBJ (optional vertex colors)
        mesh_obj_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_sample_{i}_mesh.obj')
        mesh_verts = mesh_pred[i, :, :3]
        mesh_colors = None
        if mesh_color_pred is not None:
            mesh_colors = np.clip(mesh_color_pred[i], 0.0, 1.0)
        
        if mesh_restore_indices is not None:
            mesh_verts = mesh_verts[mesh_restore_indices]
            if mesh_colors is not None:
                mesh_colors = mesh_colors[mesh_restore_indices]
        
        with open(mesh_obj_path, 'w') as f:
            f.write(f"# Epoch {epoch+1} Sample {i} Mesh Prediction\n")
            if mesh_colors is not None:
                if len(mesh_colors) != len(mesh_verts):
                    raise ValueError("mesh_colors count does not match mesh_verts")
                for v, c in zip(mesh_verts, mesh_colors):
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
            else:
                for v in mesh_verts:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for name, data in mesh_topology.items():
                start_idx = data['start_idx']
                faces = data['faces']
                f.write(f"g {name}\n")
                for face in faces:
                    indices_str = " ".join([str(idx + start_idx + 1) for idx in face])
                    f.write(f"f {indices_str}\n")

        texture_png_name = None
        if mesh_texture_pred is not None:
            tex_img = mesh_texture_pred[i].transpose(1, 2, 0)
            tex_img = np.nan_to_num(tex_img, nan=0.0, posinf=1.0, neginf=0.0)
            tex_img = np.clip(tex_img, 0.0, 1.0)
            tex_u8 = (tex_img * 255.0).astype(np.uint8)
            texture_png_name = f'epoch_{epoch+1:02d}_sample_{i}_pred_texture.png'
            texture_png_path = os.path.join(output_dir, texture_png_name)
            cv2.imwrite(texture_png_path, cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR))

        if combined_mesh_uv is not None and len(combined_mesh_uv) == len(mesh_verts):
            combined_obj_name = f'epoch_{epoch+1:02d}_sample_{i}_mesh_combined_uv.obj'
            combined_obj_path = os.path.join(output_dir, combined_obj_name)
            combined_mtl_name = f'epoch_{epoch+1:02d}_sample_{i}_mesh_combined_uv.mtl'
            combined_mtl_path = os.path.join(output_dir, combined_mtl_name)

            if texture_png_name is not None:
                with open(combined_mtl_path, 'w') as f:
                    f.write("newmtl CombinedTexture\n")
                    f.write("Ka 0.000000 0.000000 0.000000\n")
                    f.write("Kd 1.000000 1.000000 1.000000\n")
                    f.write("Ks 0.000000 0.000000 0.000000\n")
                    f.write("d 1.000000\n")
                    f.write("illum 1\n")
                    f.write(f"map_Kd {texture_png_name}\n")

            with open(combined_obj_path, 'w') as f:
                f.write(f"# Epoch {epoch+1} Sample {i} Mesh Prediction with Combined UV Layout\n")
                if texture_png_name is not None:
                    f.write(f"mtllib {combined_mtl_name}\n")
                    f.write("usemtl CombinedTexture\n")

                if mesh_colors is not None and len(mesh_colors) == len(mesh_verts):
                    for v, c in zip(mesh_verts, mesh_colors):
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                else:
                    for v in mesh_verts:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                for uv in combined_mesh_uv:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

                for name, data in mesh_topology.items():
                    start_idx = data['start_idx']
                    faces = data['faces']
                    f.write(f"g {name}\n")
                    for face in faces:
                        tokens = []
                        for idx in face:
                            vi = int(idx + start_idx + 1)
                            tokens.append(f"{vi}/{vi}")
                        f.write("f " + " ".join(tokens) + "\n")
        elif combined_mesh_uv is not None and i == 0:
            print(
                "Warning: skip combined UV OBJ export due to vertex count mismatch "
                f"(combined_uv={len(combined_mesh_uv)}, mesh_verts={len(mesh_verts)})."
            )


def save_dense2geometry_visualizations(
    model,
    batch,
    epoch,
    device,
    output_dir,
    mesh_topology,
    mesh_restore_indices=None,
):
    """Save mesh-focused visualizations for Dense2Geometry training."""
    os.makedirs(output_dir, exist_ok=True)

    max_samples = 8
    model.eval()
    with torch.no_grad():
        rgb = batch["rgb"][:max_samples].to(device)
        outputs = model(rgb)

    rgb_tensor = batch["rgb"][:max_samples]
    gt_mesh = batch["mesh"][:max_samples].detach().cpu().numpy()
    pred_mesh = outputs["mesh"][:max_samples].detach().cpu().numpy()
    searched_uv = outputs["searched_uv"][:max_samples].detach().cpu().numpy()
    match_mask = outputs["match_mask"][:max_samples].detach().cpu().numpy()
    pred_geo = outputs["pred_geo"][:max_samples].detach().cpu().numpy()

    num_samples = int(rgb_tensor.shape[0])
    vis_size = 512
    rows = []

    def _restore(values: np.ndarray | None):
        if values is None:
            return None
        out = np.asarray(values).copy()
        if mesh_restore_indices is not None:
            out = out[mesh_restore_indices]
        return out

    def _prepare_overlay_points(coords_uv: np.ndarray, width: int, height: int) -> np.ndarray:
        pts = np.asarray(coords_uv, dtype=np.float32).copy()
        invalid = ~np.isfinite(pts).all(axis=1)
        invalid |= (pts[:, 0] < 0.0) | (pts[:, 1] < 0.0)
        pts[invalid] = np.nan
        pts[:, 0] *= float(width)
        pts[:, 1] *= float(height)
        return pts

    def _draw_points_only(rgb_image: np.ndarray, coords_uv: np.ndarray, matched_mask: np.ndarray | None) -> np.ndarray:
        out = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)
        pts = _prepare_overlay_points(coords_uv, rgb_image.shape[1], rgb_image.shape[0])
        matched = None if matched_mask is None else np.asarray(matched_mask, dtype=bool).reshape(-1)
        for idx, (px, py) in enumerate(pts):
            if not np.isfinite(px) or not np.isfinite(py):
                continue
            is_matched = matched is None or (idx < matched.shape[0] and matched[idx])
            color = (0, 220, 0) if is_matched else (0, 80, 255)
            cv2.circle(
                out,
                (int(round(float(px))), int(round(float(py)))),
                1,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    for i in range(num_samples):
        img_np = (rgb_tensor[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        H, W = img_np.shape[:2]

        gt_uv = gt_mesh[i, :, 3:5]
        pred_uv = pred_mesh[i, :, 3:5]
        searched_uv_i = searched_uv[i].copy()
        match_mask_i = match_mask[i] > 0.5
        searched_uv_i[~match_mask_i] = -1.0

        gt_uv = _restore(gt_uv)
        pred_uv = _restore(pred_uv)
        searched_uv_i = _restore(searched_uv_i)
        match_mask_i = _restore(match_mask_i)
        if match_mask_i is not None:
            searched_uv_i[~np.asarray(match_mask_i, dtype=bool)] = -1.0

        gt_overlay = create_combined_overlay(img_np, _prepare_overlay_points(gt_uv, W, H), mesh_topology)
        searched_overlay = _draw_points_only(img_np, searched_uv_i, match_mask_i)
        pred_overlay = create_combined_overlay(img_np, _prepare_overlay_points(pred_uv, W, H), mesh_topology)

        aligned_rgb_vis = cv2.resize(img_np, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        geo_vis = np.transpose(pred_geo[i], (1, 2, 0))
        geo_vis = np.nan_to_num(geo_vis, nan=0.0, posinf=1.0, neginf=0.0)
        geo_vis = np.clip(geo_vis, 0.0, 1.0)
        pred_mask = outputs["pred_mask_logits"][i : i + 1].detach().cpu()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = F.interpolate(pred_mask, size=geo_vis.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()
        geo_vis = geo_vis * np.clip(pred_mask[..., None], 0.0, 1.0)
        geo_vis = cv2.resize((geo_vis * 255.0).astype(np.uint8), (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        gt_overlay = cv2.resize(gt_overlay, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        searched_overlay = cv2.resize(searched_overlay, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)
        pred_overlay = cv2.resize(pred_overlay, (vis_size, vis_size), interpolation=cv2.INTER_LINEAR)

        panels = [
            add_panel_title(aligned_rgb_vis, "Aligned RGB"),
            add_panel_title(geo_vis, "Pred Geo"),
            add_panel_title(gt_overlay, "GT Mesh Overlay"),
            add_panel_title(searched_overlay, "Searched 2D"),
            add_panel_title(pred_overlay, "Aligned Pred Mesh"),
        ]
        row = np.concatenate(panels, axis=1)
        rows.append(row)

    if rows:
        grid = np.concatenate(rows, axis=0)
        grid_path = os.path.join(output_dir, f"epoch_{epoch + 1:02d}_dense2geometry_mesh.png")
        cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Saved dense2geometry mesh visualization: {grid_path}")


# --- Geo/DPT Visualization Helpers ---

def get_encoding_image(tensor, k, n):
    # tensor: [3 + 6*n, H, W]
    idx0 = 3 + k
    idx1 = 3 + 2*n + k
    idx2 = 3 + 4*n + k
    
    c0 = tensor[idx0]
    c1 = tensor[idx1]
    c2 = tensor[idx2]
    
    img = np.stack([c0, c1, c2], axis=0)
    # Normalize -1..1 to 0..1
    img = (img + 1) / 2
    return np.clip(img.transpose(1, 2, 0), 0, 1)

def get_base_geo(tensor):
    img = tensor[0:3]
    # Data is normalized to [-1, 1]. Un-normalize to [0, 1] for visualization.
    img = (img + 1) / 2
    return np.clip(img.transpose(1, 2, 0), 0, 1)

def save_geo_visualizations(model, batch, epoch, device, output_dir="training_samples"):
    """
    Save sample predictions for DPT (Geo/Mask).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get samples (up to 8 if available)
    max_samples = 8
    rgb = batch['rgb'][:max_samples].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(rgb)
    
    # Infer n from geo channels
    # Total channels = 3 + 6 * n
    geo_channels = outputs['geo'].shape[1]
    n_encodings = (geo_channels - 3) // 6
    
    # Cols: RGB, MaskGT, MaskPred, (GeoGT_Base, GeoPred_Base), (GeoGT_Enc_i, GeoPred_Enc_i) * n
    num_cols = 3 + 2 + 2 * n_encodings
    
    # Rows: Samples
    num_samples = rgb.shape[0]
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 2, num_samples * 2))
    if num_samples == 1: axes = axes.reshape(1, -1)
    if num_cols == 1: axes = axes.reshape(-1, 1)
        
    fig.suptitle(f'Geo Predictions - Epoch {epoch+1} (n={n_encodings})', fontsize=16, y=0.98)
    
    for row in range(num_samples):
        col = 0
        
        # RGB
        rgb_vis = batch['rgb'][row].cpu().numpy().transpose(1, 2, 0)
        rgb_vis = np.clip(rgb_vis, 0, 1)
        axes[row, col].imshow(rgb_vis)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("RGB", fontsize=8)
        col += 1
        
        # Mask GT
        mask_gt = batch['face_mask'][row].cpu().numpy().squeeze()
        if mask_gt.ndim == 3 and mask_gt.shape[0] == 1: mask_gt = mask_gt.squeeze(0)
        axes[row, col].imshow(mask_gt, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Mask GT", fontsize=8)
        col += 1
        
        # Mask Pred
        alpha_logits = outputs['alpha_logits'][row].detach()
        alpha_pred = torch.sigmoid(alpha_logits).cpu().numpy().squeeze()
        axes[row, col].imshow(alpha_pred, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Mask Pred", fontsize=8)
        col += 1
        
        # Geo Data
        geo_gt = batch['geo'][row].cpu().numpy()
        geo_pred = outputs['geo'][row].detach().cpu().numpy()
        alpha_3ch = np.stack([alpha_pred] * 3, axis=2)
        
        # Base Geo GT
        axes[row, col].imshow(get_base_geo(geo_gt))
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Geo Base GT", fontsize=8)
        col += 1
        
        # Base Geo Pred
        base_pred = get_base_geo(geo_pred)
        axes[row, col].imshow(np.clip(base_pred * alpha_3ch, 0, 1))
        axes[row, col].axis('off')
        if row == 0: axes[row, col].set_title("Geo Base Pred", fontsize=8)
        col += 1
        
        # Encodings
        for k in range(n_encodings):
            # GT
            gt_img = get_encoding_image(geo_gt, k, n_encodings)
            axes[row, col].imshow(gt_img)
            axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f"Enc GT {k}", fontsize=8)
            col += 1
            
            # Pred
            pred_img = get_encoding_image(geo_pred, k, n_encodings)
            pred_masked = pred_img * alpha_3ch
            axes[row, col].imshow(np.clip(pred_masked, 0, 1))
            axes[row, col].axis('off')
            if row == 0: axes[row, col].set_title(f"Enc Pred {k}", fontsize=8)
            col += 1

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    sample_path = os.path.join(output_dir, f'epoch_{epoch+1:02d}_geo_samples.png')
    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved geo samples: {sample_path}")


def save_random_landmark_training_samples(
    data_roots,
    num_samples: int = 2,
    split: str = "train",
    output_dir: str = "training_samples",
    geo_encoding_n: int = 4,
):
    """Save landmark overlays for a few random samples from the dataset.

    Draws ground-truth landmarks as wireframe on top of the color image.
    """
    os.makedirs(output_dir, exist_ok=True)

    from metahuman_dataset2 import MetahumanDataset2

    dataset = MetahumanDataset2(
        data_roots=data_roots,
        split=split,
        image_size=512,
        train_ratio=0.95,
        augment=False,
        geo_encoding_n=geo_encoding_n,
    )

    if len(dataset) == 0:
        print("No samples found in dataset.")
        return

    topology = load_topology()

    num = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num)
    print(f"Selected training indices for GT landmark overlay: {indices}")

    for idx in indices:
        sample = dataset[idx]

        if "landmarks" not in sample:
            print(f"Sample {idx} has no landmarks, skipping.")
            continue

        landmarks = sample["landmarks"].cpu().numpy().copy()

        img_np = None
        if "image_path" in sample and sample["image_path"] and os.path.exists(sample["image_path"]):
            try:
                img_bgr = cv2.imread(sample["image_path"])
                if img_bgr is not None:
                    img_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Warning: failed to load high-res image for {idx}: {e}")

        if img_np is None:
            rgb = sample["rgb"]
            img_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # Force resize to 1024x1024
        if img_np.shape[0] != 1024 or img_np.shape[1] != 1024:
            img_np = cv2.resize(img_np, (1024, 1024), interpolation=cv2.INTER_LINEAR)

        h, w = img_np.shape[:2]

        lm_px = landmarks
        lm_px[:, 0] *= w
        lm_px[:, 1] *= h

        overlay = create_combined_overlay(img_np, lm_px, topology)

        out_path = os.path.join(output_dir, f"train_landmark_gt_{idx:06d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved GT landmark overlay: {out_path}")


