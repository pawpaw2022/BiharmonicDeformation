"""
Test utilities for HarmonicDeformation tests
"""
import torch
import numpy as np
import igl
import os
import sys

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_plane_mesh(n: int, device=torch.device('cpu')):
    """
    Create a plane mesh with n×n vertices
    
    Parameters:
        n: int
            Number of vertices in each dimension
        device: torch.device
            Device to place tensors on
            
    Returns:
        vertices: torch.Tensor, shape (n*n, 3)
            Mesh vertices
        faces: torch.Tensor, shape (2*(n-1)*(n-1), 3)
            Mesh faces
    """
    x = torch.linspace(-1, 1, n, device=device)
    y = torch.linspace(-1, 1, n, device=device)
    
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    vertices = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.zeros_like(grid_x.flatten())], dim=1)
    
    # Create faces (2 triangles per grid cell)
    faces = []
    for i in range(n-1):
        for j in range(n-1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j
            v3 = (i + 1) * n + j + 1
            
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    return vertices, faces

def create_cube_mesh(device=torch.device('cpu')):
    """
    Create a simple cube mesh with 8 vertices and 12 faces (2 per side)
    
    Parameters:
        device: torch.device
            Device to place tensors on
            
    Returns:
        vertices: torch.Tensor, shape (8, 3)
            Mesh vertices
        faces: torch.Tensor, shape (12, 3)
            Mesh faces
    """
    vertices = torch.tensor([
        [-0.5, -0.5, -0.5],  # 0 bottom
        [0.5, -0.5, -0.5],   # 1 bottom
        [0.5, 0.5, -0.5],    # 2 top
        [-0.5, 0.5, -0.5],   # 3 top
        [-0.5, -0.5, 0.5],   # 4 bottom
        [0.5, -0.5, 0.5],    # 5 bottom
        [0.5, 0.5, 0.5],     # 6 top
        [-0.5, 0.5, 0.5],    # 7 top
    ], dtype=torch.float32, device=device)
    
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [5, 6, 7], [5, 7, 4],  # back
        [4, 7, 3], [4, 3, 0],  # left
        [3, 7, 6], [3, 6, 2],  # top
        [4, 0, 1], [4, 1, 5],  # bottom
    ], dtype=torch.int64, device=device)
    
    return vertices, faces

def create_deformation_test_case(mesh_type='cube', device=torch.device('cpu')):
    """
    Create a test case for deformation tests
    
    Parameters:
        mesh_type: str, default 'cube'
            Type of mesh to create ('cube' or 'plane')
        device: torch.device
            Device to place tensors on
            
    Returns:
        vertices: torch.Tensor
            Original mesh vertices
        faces: torch.Tensor
            Mesh faces
        boundary_vertices: torch.Tensor
            Indices of boundary vertices
        boundary_values: torch.Tensor
            Target positions for boundary vertices
    """
    if mesh_type == 'cube':
        vertices, faces = create_cube_mesh(device)
        boundary_vertices = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7], device=device)
        
        # Create a deformation that stretches the cube in the y direction
        boundary_values = vertices[boundary_vertices].clone()
        boundary_values[4:, 1] += 0.3  # Move top vertices up
        
    elif mesh_type == 'plane':
        n = 10  # 10x10 grid
        vertices, faces = create_plane_mesh(n, device)
        
        # Use the outer ring of vertices as boundary
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        boundary_mask = (x_coords == -1) | (x_coords == 1) | (y_coords == -1) | (y_coords == 1)
        boundary_vertices = torch.nonzero(boundary_mask).squeeze()
        
        # Create a deformation that raises the center of the plane
        boundary_values = vertices[boundary_vertices].clone()
        
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    
    return vertices, faces, boundary_vertices, boundary_values

def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to NumPy array."""
    return tensor.detach().cpu().numpy()

def numpy_to_torch(array, device=torch.device('cpu')):
    """Convert a NumPy array to PyTorch tensor."""
    return torch.tensor(array, device=device)

def compute_error(a, b):
    """Compute relative error between two tensors or arrays."""
    if isinstance(a, torch.Tensor):
        a = torch_to_numpy(a)
    if isinstance(b, torch.Tensor):
        b = torch_to_numpy(b)
    
    return np.linalg.norm(a - b) / np.linalg.norm(a) 