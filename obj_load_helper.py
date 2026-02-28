import numpy as np

def load_uv_obj_file(obj_file_path, triangulate=True):
    """
    Load an OBJ file with vertices, UVs, normals, and faces.
    Handles triangles, quads (splits into triangles), and n-gons (fan triangulation).
    
    Args:
        obj_file_path: Path to .obj file
        triangulate: If True, split polygons into triangles. If False, keep original polygons.
    
    Returns:
        vertices: numpy array of vertex positions (N, 3)
        uvs: numpy array of UV coordinates (M, 2)
        normals: numpy array of vertex normals (K, 3) or None if not present
        vertex_faces: numpy array of vertex face indices (F, 3) or List of lists if not triangulate
        uv_faces: numpy array of UV face indices (F, 3) or List of lists
        normal_faces: numpy array of normal face indices (F, 3) or List of lists
    """
    # Initialize lists for storing vertices, UVs, normals, and faces
    vertices = []
    uvs = []
    normals = []
    faces = []

    with open(obj_file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Parse vertex coordinates from line
                vertex_coords = [float(coord) for coord in line.split()[1:]]
                vertices.append(vertex_coords)
                
            elif line.startswith('vt '):
                # Parse UV coordinates from line
                uv_coords = [float(coord) for coord in line.split()[1:]]
                uvs.append(uv_coords)
                
            elif line.startswith('vn '):
                # Parse vertex normal from line
                normal_coords = [float(coord) for coord in line.split()[1:]]
                normals.append(normal_coords)
                
            elif line.startswith('f '):
                # Parse face vertex indices, UV indices, and normal indices from line
                face_indices = []
                uv_indices = []
                normal_indices = []
                
                for index in line.split()[1:]:
                    indices = index.split('/')
                    face_indices.append(int(indices[0])-1)
                    
                    if len(indices) > 1 and indices[1]:
                        uv_indices.append(int(indices[1])-1)
                    else:
                        uv_indices.append(None)
                        
                    if len(indices) > 2 and indices[2]:
                        normal_indices.append(int(indices[2])-1)
                    else:
                        normal_indices.append(None)
                
                if triangulate:
                    # Handle quads by splitting them into triangles
                    if len(face_indices) == 4:
                        # Split quad into two triangles: (0,1,2) and (0,2,3)
                        faces.append(([face_indices[0], face_indices[1], face_indices[2]], 
                                     [uv_indices[0], uv_indices[1], uv_indices[2]],
                                     [normal_indices[0], normal_indices[1], normal_indices[2]]))
                        faces.append(([face_indices[0], face_indices[2], face_indices[3]], 
                                     [uv_indices[0], uv_indices[2], uv_indices[3]],
                                     [normal_indices[0], normal_indices[2], normal_indices[3]]))
                    elif len(face_indices) == 3:
                        # Triangle - keep as is
                        faces.append((face_indices, uv_indices, normal_indices))
                    else:
                        # Handle other polygon types (5+ vertices) by fan triangulation
                        for i in range(1, len(face_indices) - 1):
                            faces.append(([face_indices[0], face_indices[i], face_indices[i+1]], 
                                         [uv_indices[0], uv_indices[i], uv_indices[i+1]],
                                         [normal_indices[0], normal_indices[i], normal_indices[i+1]]))
                else:
                    # Keep original polygon
                    faces.append((face_indices, uv_indices, normal_indices))
    
    # Convert to numpy arrays
    vertices = np.array(vertices, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
    normals = np.array(normals, dtype=np.float32) if normals else None
    
    # Convert faces to separate vertex, UV, and normal face arrays
    vertex_faces = []
    uv_faces = []
    normal_faces = []
    
    for face_verts, face_uvs, face_normals in faces:
        vertex_faces.append(face_verts)
        uv_faces.append(face_uvs)
        normal_faces.append(face_normals)
    
    if triangulate:
        vertex_faces = np.array(vertex_faces, dtype=np.int32)
        uv_faces = np.array(uv_faces, dtype=np.int32) if any(uv is not None for face in uv_faces for uv in face) else None
        normal_faces = np.array(normal_faces, dtype=np.int32) if any(n is not None for face in normal_faces for n in face) else None
    else:
        # Keep as list of lists because lengths might differ
        # Check if UVs/Normals exist at all
        has_uvs = any(uv is not None for face in uv_faces for uv in face)
        has_normals = any(n is not None for face in normal_faces for n in face)
        
        if not has_uvs: uv_faces = None
        if not has_normals: normal_faces = None
    
    return vertices, uvs, normals, vertex_faces, uv_faces, normal_faces


def load_simple_obj_file(obj_file_path):
    """
    Load a simple OBJ file with only vertices and faces (no UVs or normals).
    
    Returns:
        vertices: list of vertex positions as numpy arrays
        faces: list of face vertex indices
    """
    vertices = []
    faces = []
    
    try:
        with open(obj_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    if len(parts) == 4:
                        vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        vertices.append(vertex)
                elif line.startswith('f '):
                    parts = line.split()
                    face_indices = []
                    for i in range(1, 4):
                        vertex_data = parts[i].split('/')
                        vertex_idx = int(vertex_data[0]) - 1
                        face_indices.append(vertex_idx)
                    faces.append(face_indices)
    except FileNotFoundError:
        print(f"Error: Could not find OBJ file at {obj_file_path}")
        return [], []
    
    return vertices, faces
