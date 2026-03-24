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
from data_utils import obj_io

def load_raw_faces(obj_path):
    """
    Reads the OBJ file and returns a list of face lines (strings) 
    and a list of normal lines (strings) from the new topology file.
    """
    faces = []
    normals = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                faces.append(line.strip())
            elif line.startswith('vn '):
                normals.append(line.strip())
    return faces, normals

def find_mapping(source_points, target_points, tolerance=1e-5):
    """
    Finds index mapping from source to target.
    map[i] = j means source_points[i] corresponds to target_points[j].
    """
    if len(source_points) != len(target_points):
        return None
    
    # Use a simple brute force with early exit for small N, or sort for better perf?
    # Given N ~ 2000, brute force O(N^2) is 4M comparisons, which takes ~1-2 seconds in Python.
    # Optimization: Sort by x coordinate to reduce search space.
    
    mapping = np.zeros(len(source_points), dtype=int) - 1
    
    # Create indexed list for target
    # target_indexed = list(enumerate(target_points))
    # Sort by X
    # target_indexed.sort(key=lambda p: p[1][0])
    
    # Just use brute force for simplicity and robustness against order, 
    # but use numpy broadcasting for speed.
    
    # Process in chunks to avoid memory issues if N is large (though 2000 is small)
    
    # Let's try a KDTree approach or simple distance matrix if scipy is available.
    # If not, simple numpy broadcasting.
    
    src = np.array(source_points)
    tgt = np.array(target_points)
    
    # For each src point, find closest tgt point
    for i in range(len(src)):
        diff = tgt - src[i]
        dist_sq = np.sum(diff**2, axis=1)
        min_idx = np.argmin(dist_sq)
        if dist_sq[min_idx] > tolerance:
            print(f"Warning: vertex {i} match distance {dist_sq[min_idx]} > tolerance")
        mapping[i] = min_idx
        
    # Check for duplicates in mapping (should be 1-to-1)
    if len(np.unique(mapping)) != len(target_points):
        print("Error: Mapping is not 1-to-1. Some vertices mapped to same target.")
        return None
        
    return mapping

def process_pair(model_path, new_topo_path):
    print(f"Processing {model_path}...")
    
    # Load model data (Target)
    try:
        v_tgt, vt_tgt, _, _, _, _ = obj_io.load_uv_obj_file(model_path)
    except Exception as e:
        # Fallback for simple obj if load_uv_obj_file fails or returns None for everything
        # Actually load_uv_obj_file handles simple files fine (returns None for missing attribs)
        # But let's handle the case where it might crash on empty faces?
        print(f"Error loading {model_path}: {e}")
        return

    # Load new topology data (Source)
    try:
        v_src, vt_src, vn_src, _, _, _ = obj_io.load_uv_obj_file(new_topo_path)
    except Exception as e:
        print(f"Error loading {new_topo_path}: {e}")
        return

    # Verify counts
    if len(v_tgt) != len(v_src):
        print(f"Vertex count mismatch: {len(v_tgt)} vs {len(v_src)}. Skipping.")
        return

    # Build Vertex Map
    print("  Mapping vertices...")
    v_map = find_mapping(v_src, v_tgt)
    if v_map is None:
        print("  Vertex mapping failed.")
        return

    # Build UV Map
    vt_map = None
    if vt_tgt is not None and vt_src is not None:
        if len(vt_tgt) == len(vt_src):
            print("  Mapping UVs...")
            
            # Ensure UVs are 2D for comparison
            vt_src_comp = vt_src
            vt_tgt_comp = vt_tgt
            
            if vt_src.shape[1] > 2:
                vt_src_comp = vt_src[:, :2]
            if vt_tgt.shape[1] > 2:
                vt_tgt_comp = vt_tgt[:, :2]
                
            # If one is 2D and other is 3D, we must compare 2D. 
            # If src has 3 cols and tgt has 2, src[:,:2] vs tgt is fine.
            # But we need to make sure we don't pass incompatible shapes to find_mapping.
            
            # Check if dimensions match now
            if vt_src_comp.shape[1] != vt_tgt_comp.shape[1]:
                 # Force both to 2D if possible
                 if vt_src_comp.shape[1] >= 2 and vt_tgt_comp.shape[1] >= 2:
                     vt_src_comp = vt_src_comp[:, :2]
                     vt_tgt_comp = vt_tgt_comp[:, :2]
            
            vt_map = find_mapping(vt_src_comp, vt_tgt_comp, tolerance=1e-4)
            if vt_map is None:
                print("  UV mapping failed (not 1-to-1). Ignoring UVs from source faces.")
        else:
             print(f"  UV count mismatch: {len(vt_tgt)} vs {len(vt_src)}. Ignoring UVs.")

    # Get raw faces and normals from new topology
    faces_lines, normals_lines = load_raw_faces(new_topo_path)
    
    # Write output
    print(f"  Writing to {model_path}...")
    with open(model_path, 'w') as f:
        f.write(f"# Processed by replace_topology.py\n")
        
        # Write Target Vertices
        for v in v_tgt:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            
        # Write Target UVs (if any)
        if vt_tgt is not None:
            for vt in vt_tgt:
                f.write(f"vt {vt[0]} {vt[1]}\n")
                
        # Write Source Normals (if any)
        # We prefer Source normals because topology changed.
        if normals_lines:
            for line in normals_lines:
                f.write(f"{line}\n")
        # Note: If source has no normals, we write nothing. 
        # Target normals are discarded as they apply to old topology.
        
        # Write Remapped Faces
        for line in faces_lines:
            # line is like "f v/vt/vn v/vt/vn ..."
            parts = line.split()
            new_line = ["f"]
            for part in parts[1:]:
                indices = part.split('/')
                
                # Vertex Index (1-based)
                v_idx = int(indices[0]) - 1
                new_v_idx = v_map[v_idx] + 1
                
                # UV Index (optional)
                new_vt_idx_str = ""
                if len(indices) > 1 and indices[1]:
                    vt_idx = int(indices[1]) - 1
                    if vt_map is not None:
                        new_vt_idx = vt_map[vt_idx] + 1
                        new_vt_idx_str = str(new_vt_idx)
                    # If no map or mapping failed, we skip writing vt index but keep the slot empty if vn follows
                
                # Normal Index (optional)
                new_vn_idx_str = ""
                if len(indices) > 2 and indices[2]:
                    # Keep original normal index from Source, as we wrote Source normals
                    if normals_lines:
                        new_vn_idx_str = indices[2]
                
                # Construct index string
                if new_vn_idx_str:
                    index_str = f"{new_v_idx}/{new_vt_idx_str}/{new_vn_idx_str}"
                elif new_vt_idx_str:
                    index_str = f"{new_v_idx}/{new_vt_idx_str}"
                else:
                    index_str = f"{new_v_idx}"
                    
                new_line.append(index_str)
            
            f.write(" ".join(new_line) + "\n")

    print(f"  Done.")

def main():
    base_dir = r"D:\Source\DAViD\model"
    new_topo_dir = os.path.join(base_dir, "new_topology")
    
    if not os.path.exists(new_topo_dir):
        print(f"New topology directory not found: {new_topo_dir}")
        return

    for filename in os.listdir(new_topo_dir):
        if not filename.lower().endswith(".obj"):
            continue
            
        new_topo_path = os.path.join(new_topo_dir, filename)
        model_path = os.path.join(base_dir, filename)
        
        if os.path.exists(model_path):
            process_pair(model_path, new_topo_path)
        else:
            print(f"Skipping {filename} (no match in model dir)")

if __name__ == "__main__":
    main()
