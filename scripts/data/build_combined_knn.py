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
import numpy as np
import json
from data_utils.obj_io import load_simple_obj_file, load_uv_obj_file

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_obj(path, vertices, faces=None):
    with open(path, 'w') as f:
        f.write(f"# Exported by compute_combined_knn.py\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        if faces is not None:
            for face in faces:
                # OBJ indices are 1-based
                f.write("f")
                for idx in face:
                    f.write(f" {idx + 1}")
                f.write("\n")
    print(f"Saved OBJ to {path}")

def load_mesh_data_full(path):
    """
    Load vertices, UVs, and faces from an OBJ file.
    Maps UVs to vertices (First-Hit) to ensure alignment without splitting vertices.
    This preserves the original vertex count/indexing for the inverse map.
    Returns:
        vertices: (N, 3) float32
        uvs: (N, 2) float32 (Aligned with vertices)
        faces: list of lists (indices)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")
    
    # Use load_uv_obj_file for everything if possible
    try:
        # returns: vertices, uvs, normals, vertex_faces, uv_faces, normal_faces
        res = load_uv_obj_file(path)
        raw_vertices = res[0]
        raw_uvs = res[1]
        vertex_faces = res[3]
        uv_faces = res[4]
        
        # If we have valid vertices
        if raw_vertices is not None and len(raw_vertices) > 0:
            vertices = raw_vertices
            
            # Align UVs: Create an array aligned with vertices
            # Initialize with -1.0 (invalid)
            aligned_uvs = np.full((len(vertices), 2), -1.0, dtype=np.float32)
            
            if raw_uvs is not None and uv_faces is not None and len(raw_uvs) > 0:
                # Ensure lists
                v_faces_list = vertex_faces.tolist() if hasattr(vertex_faces, 'tolist') else vertex_faces
                u_faces_list = uv_faces.tolist() if hasattr(uv_faces, 'tolist') else uv_faces
                
                # To assign UVs, we iterate faces. 
                # If a vertex has multiple UVs, we pick the first one encountered.
                visited = np.zeros(len(vertices), dtype=bool)
                
                for i in range(len(v_faces_list)):
                    v_face = v_faces_list[i]
                    u_face = u_faces_list[i]
                    
                    for v_idx, u_idx in zip(v_face, u_face):
                        if not visited[v_idx]:
                            if u_idx is not None and 0 <= u_idx < len(raw_uvs):
                                aligned_uvs[v_idx] = raw_uvs[u_idx]
                                visited[v_idx] = True
            
            faces = vertex_faces.tolist() if vertex_faces is not None else []
            return vertices, aligned_uvs, faces
            
    except Exception as e:
        print(f"Warning: load_uv_obj_file failed for {path}: {e}")
    
    # Fallback to simple loader
    try:
        verts_list, faces_list = load_simple_obj_file(path)
        if len(verts_list) > 0:
            vertices = np.array(verts_list, dtype=np.float32)
            faces = faces_list
            return vertices, None, faces
    except Exception:
        pass

    return np.empty((0, 3), dtype=np.float32), None, []

def remove_duplicates_and_remap(vertices, uvs, faces):
    """
    Deduplicate vertices, keep UVs (first one), remap faces.
    Returns:
        sorted_unique_verts: (M, 3)
        new_uvs: (M, 2)
        new_faces: List of lists
        kept_original_indices: (M,) Indices into original 'vertices' array that were kept
    """
    if len(vertices) == 0:
        return vertices, uvs, faces, np.empty(0, dtype=int)

    # unique_inverse: indices to reconstruct original from unique
    # Use rounding to merge close vertices (within 1e-2)
    rounded_vertices = np.round(vertices, decimals=3)
    _, unique_indices, unique_inverse = np.unique(
        rounded_vertices, axis=0, return_index=True, return_inverse=True
    )
    
    # Sort by index to preserve order of first appearance
    sort_order = np.argsort(unique_indices)
    
    # Original indices that correspond to the sorted unique vertices
    kept_original_indices = unique_indices[sort_order]
    
    # Retrieve original precision vertices
    sorted_unique_verts = vertices[kept_original_indices]
    
    # Map for faces: Lexicographical -> Appearance Sorted
    lex_to_sorted = np.empty_like(sort_order)
    lex_to_sorted[sort_order] = np.arange(len(sort_order))
    
    # Final map: Original -> Sorted Unique Index
    final_inverse_map = lex_to_sorted[unique_inverse]
    
    # Remap faces
    new_faces = []
    for face in faces:
        new_face = [int(final_inverse_map[idx]) for idx in face]
        new_faces.append(new_face)
        
    # UVs
    if uvs is not None:
        new_uvs = uvs[kept_original_indices]
    else:
        new_uvs = None
        
    return sorted_unique_verts, new_uvs, new_faces, kept_original_indices, final_inverse_map

def compute_inverse_distance_weights(landmark_vertices: np.ndarray,
                                     keypoint_vertices: np.ndarray,
                                     k: int = 8,
                                     chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    
    if landmark_vertices.shape[1] != 3 or keypoint_vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    num_landmark = landmark_vertices.shape[0]
    num_keypoint = keypoint_vertices.shape[0]

    if num_keypoint < k:
        print(f"Warning: requested k={k} but only have {num_keypoint} keypoint vertices. Reducing k.")
        k = num_keypoint

    indices_all = np.empty((num_landmark, k), dtype=np.int32)
    weights_all = np.empty((num_landmark, k), dtype=np.float32)

    for start in range(0, num_landmark, chunk_size):
        end = min(start + chunk_size, num_landmark)
        lv = landmark_vertices[start:end]

        diff = lv[:, None, :] - keypoint_vertices[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)

        if num_keypoint == k:
             idx_k = np.arange(k)[None, :].repeat(end-start, axis=0)
             dist_sq_k = dist_sq
        else:
            idx_k = np.argpartition(dist_sq, k - 1, axis=1)[:, :k]
            dist_sq_k = np.take_along_axis(dist_sq, idx_k, axis=1)

        order = np.argsort(dist_sq_k, axis=1)
        idx_sorted = np.take_along_axis(idx_k, order, axis=1)
        dist_sq_sorted = np.take_along_axis(dist_sq_k, order, axis=1)

        dist = np.sqrt(dist_sq_sorted)
        zero_mask = dist <= 1e-8
        has_zero = np.any(zero_mask, axis=1)

        inv_dist = 1.0 / (dist + 1e-8)
        inv_sum = np.sum(inv_dist, axis=1, keepdims=True)
        weights = inv_dist / inv_sum

        if np.any(has_zero):
            idx_rows = np.where(has_zero)[0]
            zero_indices = np.argmax(zero_mask[has_zero], axis=1)
            w_rows = weights[idx_rows]
            w_rows[:] = 0.0
            w_rows[np.arange(len(idx_rows)), zero_indices] = 1.0
            weights[idx_rows] = w_rows

        indices_all[start:end] = idx_sorted.astype(np.int32)
        weights_all[start:end] = weights.astype(np.float32)

    return indices_all, weights_all

def generate_mask_labels(uvs, is_head, mask_tex_path, output_img_path, output_txt_path):
    print(f"\nGenerating mask labels...")
    if uvs is None or len(uvs) == 0:
        print("No UVs provided for mask generation.")
        return
    
    width = 1024
    height = 1024
    
    try:
        import cv2
        mask_img = cv2.imread(mask_tex_path)
        if mask_img is None: raise ImportError("No Image")
        
        mask_h, mask_w = mask_img.shape[:2]
        out_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        mask_labels = []
        
        for i, uv in enumerate(uvs):
            u, v = uv[0], uv[1]

            # Non-head vertices always get label 1
            if not is_head[i]:
                mask_labels.append(1)
                if 0 <= u <= 1 and 0 <= v <= 1:
                    x, y = int(u * width), int((1 - v) * height)
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(out_img, (x, y), 2, (0, 0, 255), -1)
                continue

            if u < 0 or u > 1 or v < 0 or v > 1:
                mask_labels.append(1)
                continue
            
            mx = int(u * mask_w)
            my = int((1 - v) * mask_h)
            mx = max(0, min(mx, mask_w - 1))
            my = max(0, min(my, mask_h - 1))
            
            pixel = mask_img[my, mx]
            if np.any(pixel > 0):
                label = 0
                draw_color = (0, 0, 0)
            else:
                label = 1
                draw_color = (0, 0, 255)
            
            mask_labels.append(label)
            x, y = int(u * width), int((1 - v) * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(out_img, (x, y), 2, draw_color, -1)
                
        cv2.imwrite(output_img_path, out_img)
        with open(output_txt_path, 'w') as f:
            for lbl in mask_labels:
                f.write(f"{lbl}\n")
        print(f"Saved mask labels and image.")
        
    except Exception:
        print("Could not generate mask (cv2 missing or image not found).")

def main():
    base_dir = "assets/topology"
    output_dir = "visualization"
    ensure_dir(output_dir)
    
    mask_tex_path = 'assets/topology/hair_mask_tex.png'
    
    parts = [
        ("head", "mesh_head.obj", "landmark_head.obj", "keypoint_head.obj"),
        ("eye_l", "mesh_eye_l.obj", "landmark_eye_l.obj", "keypoint_eye_l.obj"),
        ("eye_r", "mesh_eye_r.obj", "landmark_eye_r.obj", "keypoint_eye_r.obj"),
        ("mouth", "mesh_mouth.obj", "landmark_mouth.obj", "keypoint_mouth.obj"),
    ]

    # Containers
    mesh_verts_list = []
    mesh_uvs_list = []
    mesh_is_head_list = []
    mesh_faces_list = []
    
    landmark_verts_list = []
    landmark_uvs_list = []
    landmark_is_head_list = []
    landmark_faces_list = []
    
    keypoint_verts_list = []
    keypoint_faces_list = []
    
    mesh_offset = 0
    landmark_offset = 0
    keypoint_offset = 0
    
    print("Loading all meshes...")
    for part_name, mesh_file, landmark_file, keypoint_file in parts:
        mv, muvs, mfaces = load_mesh_data_full(os.path.join(base_dir, mesh_file))
        mfaces_offset = [[idx + mesh_offset for idx in face] for face in mfaces]
        
        if muvs is None:
            muvs = np.full((len(mv), 2), -1.0, dtype=np.float32)
        if len(muvs) != len(mv):
            if len(muvs) < len(mv):
                pad = np.full((len(mv) - len(muvs), 2), -1.0, dtype=np.float32)
                muvs = np.vstack([muvs, pad])
            else:
                muvs = muvs[:len(mv)]
        
        mesh_verts_list.append(mv)
        mesh_uvs_list.append(muvs)
        mesh_is_head_list.append(np.full(len(mv), part_name == "head", dtype=bool))
        mesh_faces_list.extend(mfaces_offset)
        mesh_offset += len(mv)
        
        # Load Landmark
        lv, luvs, lfaces = load_mesh_data_full(os.path.join(base_dir, landmark_file))
        
        # Handle UVs (now guaranteed to be aligned if present)
        if luvs is None:
            luvs = np.full((len(lv), 2), -1.0, dtype=np.float32)
        
        # Verify alignment
        if len(luvs) != len(lv):
            print(f"Warning: UV count {len(luvs)} != Vertex count {len(lv)} for {part_name}. Fixing...")
            if len(luvs) < len(lv):
                pad = np.full((len(lv) - len(luvs), 2), -1.0, dtype=np.float32)
                luvs = np.vstack([luvs, pad])
            else:
                luvs = luvs[:len(lv)]
        
        # Offset Landmark Faces
        lfaces_offset = [[idx + landmark_offset for idx in face] for face in lfaces]
        
        landmark_verts_list.append(lv)
        landmark_uvs_list.append(luvs)
        landmark_is_head_list.append(np.full(len(lv), part_name == "head", dtype=bool))
        landmark_faces_list.extend(lfaces_offset)
        landmark_offset += len(lv)
        
        # Load Keypoint
        kv, kuvs, kfaces = load_mesh_data_full(os.path.join(base_dir, keypoint_file))
        kfaces_offset = [[idx + keypoint_offset for idx in face] for face in kfaces]
        
        keypoint_verts_list.append(kv)
        keypoint_faces_list.extend(kfaces_offset)
        keypoint_offset += len(kv)
        
        print(f"Loaded {part_name}")

    # Combine
    combined_mesh_verts = np.concatenate(mesh_verts_list, axis=0)
    combined_mesh_uvs = np.concatenate(mesh_uvs_list, axis=0)
    combined_mesh_is_head = np.concatenate(mesh_is_head_list, axis=0)
    combined_landmark_verts = np.concatenate(landmark_verts_list, axis=0)
    combined_landmark_uvs = np.concatenate(landmark_uvs_list, axis=0)
    combined_landmark_is_head = np.concatenate(landmark_is_head_list, axis=0)
    combined_keypoint_verts = np.concatenate(keypoint_verts_list, axis=0)
    
    # Deduplicate Mesh (removes UV-induced duplicates)
    print("Deduplicating Mesh...")
    unique_mesh, unique_mesh_uvs, new_mesh_faces, kept_mesh_indices, mesh_inverse_map = remove_duplicates_and_remap(
        combined_mesh_verts, combined_mesh_uvs, mesh_faces_list
    )
    
    # Deduplicate Landmark
    print("Deduplicating Landmark...")
    unique_landmark, unique_landmark_uvs, new_landmark_faces, kept_landmark_indices, landmark_inverse_map = remove_duplicates_and_remap(
        combined_landmark_verts, combined_landmark_uvs, landmark_faces_list
    )
    
    # Deduplicate Keypoint
    print("Deduplicating Keypoint...")
    unique_keypoint, _, new_keypoint_faces, _, _ = remove_duplicates_and_remap(
        combined_keypoint_verts, None, keypoint_faces_list
    )
    
    print(f"Mesh: Original {len(combined_mesh_verts)} -> Unique {len(unique_mesh)} (Removed {len(combined_mesh_verts) - len(unique_mesh)})")
    print(f"Landmark: Original {len(combined_landmark_verts)} -> Unique {len(unique_landmark)} (Removed {len(combined_landmark_verts) - len(unique_landmark)})")
    print(f"Keypoint: Original {len(combined_keypoint_verts)} -> Unique {len(unique_keypoint)} (Removed {len(combined_keypoint_verts) - len(unique_keypoint)})")
    
    # Save mesh index mappings
    mesh_indices_path = os.path.join(base_dir, "mesh_indices.npy")
    np.save(mesh_indices_path, kept_mesh_indices)
    print(f"Saved mesh indices mapping to {mesh_indices_path}")

    mesh_inverse_path = os.path.join(base_dir, "mesh_inverse.npy")
    np.save(mesh_inverse_path, mesh_inverse_map)
    print(f"Saved mesh inverse mapping to {mesh_inverse_path}")
    
    # Save landmark index mappings
    landmark_indices_path = os.path.join(base_dir, "landmark_indices.npy")
    np.save(landmark_indices_path, kept_landmark_indices)
    print(f"Saved landmark indices mapping to {landmark_indices_path}")

    landmark_inverse_path = os.path.join(base_dir, "landmark_inverse.npy")
    np.save(landmark_inverse_path, landmark_inverse_map)
    print(f"Saved landmark inverse mapping to {landmark_inverse_path}")
    
    # KNN: Mesh → Landmark
    print("Computing Mesh → Landmark KNN...")
    mesh2landmark_indices, mesh2landmark_weights = compute_inverse_distance_weights(unique_mesh, unique_landmark, k=8)
    
    # KNN: Landmark → Keypoint
    print("Computing Landmark → Keypoint KNN...")
    landmark2keypoint_indices, landmark2keypoint_weights = compute_inverse_distance_weights(unique_landmark, unique_keypoint, k=8)
    
    # Save NPYs to assets/topology/
    np.save("assets/topology/mesh2landmark_knn_indices.npy", mesh2landmark_indices)
    np.save("assets/topology/mesh2landmark_knn_weights.npy", mesh2landmark_weights)
    np.save("assets/topology/landmark2keypoint_knn_indices.npy", landmark2keypoint_indices)
    np.save("assets/topology/landmark2keypoint_knn_weights.npy", landmark2keypoint_weights)
    
    # Save OBJs to assets/demo/
    save_obj(os.path.join(output_dir, "combined_mesh.obj"), unique_mesh, new_mesh_faces)
    save_obj(os.path.join(output_dir, "combined_landmark.obj"), unique_landmark, new_landmark_faces)
    save_obj(os.path.join(output_dir, "combined_keypoint.obj"), unique_keypoint, new_keypoint_faces)
    
    # Save JSON to assets/demo/ with VERTICES
    # Flatten arrays for JSON size efficiency
    knn_data = {
        "mesh2landmark_indices": mesh2landmark_indices.tolist(),
        "mesh2landmark_weights": mesh2landmark_weights.tolist(),
        "landmark2keypoint_indices": landmark2keypoint_indices.tolist(),
        "landmark2keypoint_weights": landmark2keypoint_weights.tolist(),
        "mesh_vertices": unique_mesh.flatten().tolist(),
        "landmark_vertices": unique_landmark.flatten().tolist(),
        "keypoint_vertices": unique_keypoint.flatten().tolist()
    }
    with open(os.path.join(output_dir, "knn_data.json"), 'w') as f:
        json.dump(knn_data, f)
    print(f"Saved JSON data.")
    
    # Build is_head flags for deduplicated vertices
    unique_landmark_is_head = combined_landmark_is_head[kept_landmark_indices]
    unique_mesh_is_head = combined_mesh_is_head[kept_mesh_indices]
    
    # Generate Masks
    generate_mask_labels(unique_landmark_uvs, unique_landmark_is_head, mask_tex_path, "assets/topology/head_uv_points.png", "assets/topology/landmark_mask.txt")
    generate_mask_labels(unique_mesh_uvs, unique_mesh_is_head, mask_tex_path, "assets/topology/mesh_uv_points.png", "assets/topology/mesh_mask.txt")

if __name__ == "__main__":
    main()
