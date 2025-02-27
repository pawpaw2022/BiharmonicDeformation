"""
Benchmarking utilities for PyTorch harmonic deformation implementation
"""
import torch
import time
import numpy as np
from typing import List, Dict, Optional
import sys
import os

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.HarmonicDeformation import harmonic_deformation

def benchmark_pytorch_harmonic(
    test_cases: List[Dict],
    repeat: int = 5,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Benchmark PyTorch harmonic deformation on various test cases.
    
    Parameters:
        test_cases: List[Dict]
            List of test cases, each containing 'vertices', 'faces', 
            'boundary_vertices', 'boundary_values', and 'k'
        repeat: int, default 5
            Number of times to repeat each test for timing
        device: torch.device
            Device to run the tests on
            
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
        
        # Convert to PyTorch if needed
        if not isinstance(vertices, torch.Tensor):
            vertices = torch.tensor(vertices, device=device)
        if not isinstance(faces, torch.Tensor):
            faces = torch.tensor(faces, device=device, dtype=torch.int64)
        if not isinstance(boundary_vertices, torch.Tensor):
            boundary_vertices = torch.tensor(boundary_vertices, device=device, dtype=torch.int64)
        if not isinstance(boundary_values, torch.Tensor):
            boundary_values = torch.tensor(boundary_values, device=device)
        
        # Move to specified device
        vertices = vertices.to(device)
        faces = faces.to(device)
        boundary_vertices = boundary_vertices.to(device)
        boundary_values = boundary_values.to(device)
        
        # Warm-up run
        if device.type == 'cuda':
            torch.cuda.synchronize()
            _ = harmonic_deformation(vertices, faces, boundary_vertices, boundary_values, k)
            torch.cuda.synchronize()
        else:
            _ = harmonic_deformation(vertices, faces, boundary_vertices, boundary_values, k)
        
        times = []
        for _ in range(repeat):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start_time = time.time()
                _ = harmonic_deformation(vertices, faces, boundary_vertices, boundary_values, k)
                torch.cuda.synchronize()
                times.append(time.time() - start_time)
            else:
                start_time = time.time()
                _ = harmonic_deformation(vertices, faces, boundary_vertices, boundary_values, k)
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
            "k": k,
            "device": str(device)
        }
        
        # Clean up to avoid memory issues
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results 