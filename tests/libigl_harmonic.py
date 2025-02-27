"""
Harmonic and biharmonic deformation using libigl
"""
import igl
import numpy as np
import time
from typing import Tuple, List, Optional, Dict

def libigl_harmonic_deformation(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    boundary_values: np.ndarray,
    k: int = 1
) -> np.ndarray:
    """
    Compute k-harmonic deformation of a mesh using libigl.
    
    Parameters:
        vertices: np.ndarray, shape (V, 3)
            Input mesh vertices
        faces: np.ndarray, shape (F, 3)
            Input mesh faces
        boundary_vertices: np.ndarray, shape (B)
            Indices of boundary vertices
        boundary_values: np.ndarray, shape (B, 3)
            Target positions for boundary vertices
        k: int, default 1
            Order of harmonic equation (1 for harmonic, 2 for biharmonic)
            
    Returns:
        np.ndarray, shape (V, 3)
            Deformed mesh vertices
    """
    # Convert to 0-indexed if not already
    faces = faces.astype(np.int32)
    boundary_vertices = boundary_vertices.astype(np.int32)
    
    # Ensure vertices and boundary_values have the same data type
    vertices_dtype = vertices.dtype
    boundary_values = boundary_values.astype(vertices_dtype)
    
    # Create boundary conditions with the same dtype as vertices
    bc = np.zeros((len(boundary_vertices), vertices.shape[1]), dtype=vertices_dtype)
    bc[:] = boundary_values
    
    # Solve using libigl
    deformed = np.zeros_like(vertices)
    
    for d in range(vertices.shape[1]):  # For each coordinate (x, y, z)
        Z = igl.harmonic(vertices, faces, boundary_vertices, bc[:, d], k)
        deformed[:, d] = Z
    
    return deformed

def benchmark_libigl_harmonic(
    test_cases: List[Dict],
    repeat: int = 5
) -> Dict:
    """
    Benchmark libigl harmonic deformation on various test cases.
    
    Parameters:
        test_cases: List[Dict]
            List of test cases, each containing 'vertices', 'faces', 
            'boundary_vertices', 'boundary_values', and 'k'
        repeat: int, default 5
            Number of times to repeat each test for timing
            
    Returns:
        Dict
            Dictionary with timing information
    """
    results = {}
    
    for i, case in enumerate(test_cases):
        vertices = case['vertices']
        faces = case['faces']
        boundary_vertices = case['boundary_vertices']
        boundary_values = case['boundary_values']
        k = case.get('k', 1)
        name = case.get('name', f"Case {i+1}")
        
        # Convert to numpy if needed
        if not isinstance(vertices, np.ndarray):
            vertices = vertices.detach().cpu().numpy()
        if not isinstance(faces, np.ndarray):
            faces = faces.detach().cpu().numpy()
        if not isinstance(boundary_vertices, np.ndarray):
            boundary_vertices = boundary_vertices.detach().cpu().numpy()
        if not isinstance(boundary_values, np.ndarray):
            boundary_values = boundary_values.detach().cpu().numpy()
        
        # Ensure consistent data types
        vertices = vertices.astype(np.float32)
        boundary_values = boundary_values.astype(np.float32)
        
        times = []
        for _ in range(repeat):
            start_time = time.time()
            _ = libigl_harmonic_deformation(
                vertices, faces, boundary_vertices, boundary_values, k
            )
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        std_time = np.std(times)
        
        results[name] = {
            "avg_time": avg_time,
            "std_time": std_time,
            "times": times,
            "vertices_count": len(vertices),
            "faces_count": len(faces),
            "boundary_count": len(boundary_vertices),
            "k": k
        }
    
    return results 