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
from data_utils.obj_io import load_simple_obj_file

def load_vertices_from_obj(path):
    """Load vertices from a simple OBJ file as (N, 3) float32 array."""
    verts_list, _ = load_simple_obj_file(path)
    if len(verts_list) == 0:
        raise FileNotFoundError(f"No vertices loaded from {path}")
    verts = np.stack(verts_list, axis=0).astype(np.float32)
    return verts

def remove_duplicate_vertices(vertices: np.ndarray) -> np.ndarray:
    """Remove exact duplicate vertices (same xyz)."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    unique_vertices, unique_indices = np.unique(vertices, axis=0, return_index=True)
    # Keep order of first appearance
    order = np.argsort(unique_indices)
    return unique_vertices[order]

def main():
    base_dir = "assets/topology"
    files = ["landmark_head.obj", "landmark_eye_l.obj", "landmark_eye_r.obj", "landmark_mouth.obj"]
    
    all_vertices = []
    
    for filename in files:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
            
        verts = load_vertices_from_obj(path)
        print(f"Loaded {filename}: {len(verts)} vertices")
        all_vertices.append(verts)
    
    if not all_vertices:
        print("No vertices loaded.")
        return

    # Combine all vertices
    combined_vertices = np.concatenate(all_vertices, axis=0)
    print(f"Total vertices before stripping: {len(combined_vertices)}")
    
    # Strip vertices at same position
    unique_vertices = remove_duplicate_vertices(combined_vertices)
    print(f"Total vertices after stripping: {len(unique_vertices)}")

if __name__ == "__main__":
    main()
