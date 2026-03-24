import numpy as np
import re
import math

def parse_vector(vector_str):
    """Parse Unreal Engine vector string like 'X=1.0 Y=2.0 Z=3.0'"""
    pattern = r'X=([-\d.]+)\s+Y=([-\d.]+)\s+Z=([-\d.]+)'
    match = re.search(pattern, vector_str)
    if match:
        return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    return None

def parse_rotation(rotation_str):
    """Parse Unreal Engine rotation string like 'P=1.0 Y=2.0 R=3.0'"""
    pattern = r'P=([-\d.]+)\s+Y=([-\d.]+)\s+R=([-\d.]+)'
    match = re.search(pattern, rotation_str)
    if match:
        return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    return None

def parse_matrix(matrix_str):
    """Parse 4x4 matrix from string format"""
    lines = matrix_str.strip().split('\n')
    matrix = np.zeros((4, 4))
    for i, line in enumerate(lines):
        numbers = re.findall(r'([-\d.]+)', line)
        if len(numbers) == 4:
            for j in range(4):
                matrix[i][j] = float(numbers[j])
    return matrix

def degrees_to_radians(degrees):
    """Convert degrees to radians"""
    return degrees * math.pi / 180.0

def create_rotation_matrix(pitch, yaw, roll):
    """Create rotation matrix from Unreal Engine pitch, yaw, roll (in degrees)"""
    pitch_rad = degrees_to_radians(pitch)
    yaw_rad = degrees_to_radians(yaw) 
    roll_rad = degrees_to_radians(roll)
    
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
    
    rotation_matrix = np.array([
        [cos_p * cos_y, cos_p * sin_y, sin_p],
        [sin_r * sin_p * cos_y - cos_r * sin_y, sin_r * sin_p * sin_y + cos_r * cos_y, -sin_r * cos_p],
        [-cos_r * sin_p * cos_y - sin_r * sin_y, -cos_r * sin_p * sin_y + sin_r * cos_y, cos_r * cos_p]
    ])
    
    return rotation_matrix

def create_transform_matrix(location, rotation, scale):
    """Create 4x4 transformation matrix from location, rotation, and scale"""
    transform_matrix = np.eye(4)
    
    rotation_matrix = create_rotation_matrix(rotation[0], rotation[1], rotation[2])
    scaled_rotation = rotation_matrix * scale.reshape(1, 3)
    
    transform_matrix[:3, :3] = scaled_rotation
    transform_matrix[3, :3] = location
    
    return transform_matrix

def load_matrix_data(file_path):
    """Load and parse the MatrixData.txt file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    data = {}
    
    # Parse Head Transform
    head_section = re.search(r'Head Transform:(.*?)(?=Camera Transform:|Head 4x4 Matrix:)', content, re.DOTALL)
    if head_section:
        head_text = head_section.group(1)
        data['head_location'] = parse_vector(re.search(r'Location: (.+)', head_text).group(1))
        data['head_rotation'] = parse_rotation(re.search(r'Rotation: (.+)', head_text).group(1))
        data['head_scale'] = parse_vector(re.search(r'Scale: (.+)', head_text).group(1))
    
    # Parse Camera Transform
    camera_section = re.search(r'Camera Transform:(.*?)(?=Head 4x4 Matrix:|Camera 4x4 Matrix:)', content, re.DOTALL)
    if camera_section:
        camera_text = camera_section.group(1)
        data['camera_location'] = parse_vector(re.search(r'Location: (.+)', camera_text).group(1))
        data['camera_rotation'] = parse_rotation(re.search(r'Rotation: (.+)', camera_text).group(1))
        data['camera_scale'] = parse_vector(re.search(r'Scale: (.+)', camera_text).group(1))
    
    # Parse Head 4x4 Matrix
    head_matrix_section = re.search(r'Head 4x4 Matrix:\n((?:\[.*?\]\n){4})', content)
    if head_matrix_section:
        data['head_matrix'] = parse_matrix(head_matrix_section.group(1))
    
    # Parse Camera 4x4 Matrix
    camera_matrix_section = re.search(r'Camera 4x4 Matrix:\n((?:\[.*?\]\n){4})', content)
    if camera_matrix_section:
        data['camera_matrix'] = parse_matrix(camera_matrix_section.group(1))
    
    # Parse Camera Info
    fov_match = re.search(r'FOV: ([\d.]+)', content)
    if fov_match:
        data['fov'] = float(fov_match.group(1))
    
    resolution_match = re.search(r'Resolution: (\d+)x(\d+)', content)
    if resolution_match:
        data['resolution'] = (int(resolution_match.group(1)), int(resolution_match.group(2)))
    
    # Parse Near Clipping Plane
    near_plane_match = re.search(r'NearClippingPlane: ([\d.]+)', content)
    if near_plane_match:
        data['near_clipping_plane'] = float(near_plane_match.group(1))
    
    return data

def get_world_to_view_rotation(matrix_data) -> np.ndarray:
    """Return the 3x3 rotation matrix that maps world-space direction vectors to view/camera space.

    Uses row-vector convention (same as the rest of the projection pipeline):
        N_view = N_world @ R   (shape [N, 3] @ [3, 3])

    This is purely rotational — no translation, no perspective — so it is correct
    for transforming surface normals from world space to screen/camera space.
    """
    camera_location = matrix_data['camera_location']
    camera_rotation = matrix_data['camera_rotation']
    camera_scale = np.array([1.0, 1.0, 1.0])

    camera_matrix = create_transform_matrix(camera_location, camera_rotation, camera_scale)
    view_matrix = np.linalg.inv(camera_matrix)

    coord_conversion_matrix = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    view_matrix = view_matrix @ coord_conversion_matrix
    return view_matrix[:3, :3].astype(np.float32)


def create_view_projection_matrices_from_cpp(matrix_data):
    """Create view and projection matrices matching C++ SaveLandmarks function"""
    camera_location = matrix_data['camera_location']
    camera_rotation = matrix_data['camera_rotation'] 
    camera_scale = np.array([1.0, 1.0, 1.0])
    
    camera_matrix = create_transform_matrix(camera_location, camera_rotation, camera_scale)
    view_matrix = np.linalg.inv(camera_matrix)
    
    coord_conversion_matrix = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    view_matrix = view_matrix @ coord_conversion_matrix
    
    fov = matrix_data['fov']
    fov_radians_half = np.radians(fov) * 0.5
    near_plane = matrix_data.get('near_clipping_plane', 10.0)
    
    f = 1.0 / np.tan(fov_radians_half)
    projection_matrix = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 0, 1],
        [0, 0, near_plane, 0]
    ])
    
    view_projection_matrix = view_matrix @ projection_matrix
    view_projection_matrix = view_projection_matrix.T  # Column-major to row-major conversion
    
    return view_projection_matrix


def project_3d_to_2d_cpp_exact(vertices, matrix_data):
    """Project 3D landmarks to 2D screen coordinates"""
    view_projection_matrix = create_view_projection_matrices_from_cpp(matrix_data)
    resolution = matrix_data['resolution']
    head_matrix = matrix_data.get('head_matrix', None)
    
    # vertices is (N, 3)
    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    local_pos_homo = np.hstack([vertices, ones]) # (N, 4)
    
    # Transform to world space
    if head_matrix is not None:
        # matrix multiplication: (N, 4) @ (4, 4) -> (N, 4)
        world_pos_homo = local_pos_homo @ head_matrix
    else:
        world_pos_homo = local_pos_homo
        
    # Project world space to screen space
    view_proj_pos = (view_projection_matrix @ world_pos_homo.T).T
    
    # Perspective division
    # Avoid division by zero
    w = view_proj_pos[:, 3]
    w[w == 0] = 1e-5
    rhw = 1.0 / w
    
    pos_in_screen_space_x = view_proj_pos[:, 0] * rhw
    pos_in_screen_space_y = view_proj_pos[:, 1] * rhw
    
    normalized_x = (pos_in_screen_space_x / 2.0) + 0.5
    normalized_y = 1.0 - (pos_in_screen_space_y / 2.0) - 0.5
    
    screen_x = normalized_x * resolution[0]
    screen_y = normalized_y * resolution[1]
    
    return np.stack([screen_x, screen_y], axis=1)


def compute_vertex_depth(vertices, matrix_data):
    """
    Compute normalized z-depth for each vertex for rendering order.
    Returns depth values (positive = in front of camera).
    """
    view_projection_matrix = create_view_projection_matrices_from_cpp(matrix_data)
    head_matrix = matrix_data.get('head_matrix', None)
    
    # vertices is (N, 3)
    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    local_pos_homo = np.hstack([vertices, ones])
    
    # Transform to world space
    if head_matrix is not None:
        world_pos_homo = local_pos_homo @ head_matrix
    else:
        world_pos_homo = local_pos_homo
        
    # Project to clip space
    view_proj_pos = (view_projection_matrix @ world_pos_homo.T).T
    
    # Get z and w components
    z = view_proj_pos[:, 2]
    w = view_proj_pos[:, 3]
    
    # Ensure w is never too close to zero to avoid extreme values
    w_safe = np.where(np.abs(w) < 1e-5, np.sign(w) * 1e-5, w)
    
    # Normalized depth after perspective division
    depth_normalized = z / w_safe
    
    # Clip to reasonable range to prevent extreme values
    depth_normalized = np.clip(depth_normalized, -100.0, 100.0)
    
    return depth_normalized
