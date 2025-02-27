"""
Unit tests for the individual functions in HarmonicDeformation.py
"""
import unittest
import torch
import numpy as np
import sys
import os
import math

# Add the parent directory to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.HarmonicDeformation import (
    compute_angles,
    compute_cotangent_laplacian,
    compute_mass_matrix,
    solve_linear_system,
    harmonic_deformation
)
from tests.test_utils import create_cube_mesh, create_plane_mesh, compute_error

class TestHarmonicFunctions(unittest.TestCase):
    """Test cases for the individual functions in HarmonicDeformation.py"""
    
    def setUp(self):
        """Set up test cases."""
        self.device = torch.device('cpu')
        self.vertices_cube, self.faces_cube = create_cube_mesh(self.device)
        self.vertices_plane, self.faces_plane = create_plane_mesh(10, self.device)
        
        # For angle computation test
        self.v1 = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        self.v2 = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        self.v3 = torch.tensor([[0.0, 1.0, 0.0]], device=self.device)
        
        # For linear system test
        self.A = torch.tensor([
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 2.0]
        ], device=self.device)
        self.b = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], device=self.device)
    
    def test_compute_angles(self):
        """Test the angle computation function."""
        # Test right angle (90 degrees)
        cos_angle = compute_angles(self.v1, self.v2, self.v3)
        self.assertAlmostEqual(cos_angle.item(), 0.0, places=5)
        
        # Test 45 degree angle
        v3_45 = torch.tensor([[1.0, 1.0, 0.0]], device=self.device)
        cos_angle = compute_angles(self.v1, self.v2, v3_45)
        self.assertAlmostEqual(cos_angle.item(), math.cos(math.pi/4), places=5)
        
        # Test 60 degree angle
        v3_60 = torch.tensor([[0.5, math.sqrt(3)/2, 0.0]], device=self.device)
        cos_angle = compute_angles(self.v1, self.v2, v3_60)
        self.assertAlmostEqual(cos_angle.item(), 0.5, places=5)  # cos(60°) = 0.5
    
    def test_compute_cotangent_laplacian(self):
        """Test the cotangent Laplacian computation."""
        # Create a planar triangle mesh (4 vertices, 2 triangles forming a square)
        # This is a simpler test case where we can expect more predictable behavior
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ], device=self.device)
        
        faces = torch.tensor([
            [0, 1, 2],
            [0, 2, 3]
        ], device=self.device)
        
        L = compute_cotangent_laplacian(vertices, faces)
        
        # For this simple square mesh:
        # 1. The Laplacian matrix should be symmetric
        self.assertTrue(torch.allclose(L, L.T, rtol=1e-5))
        
        # 2. The row sums should be zero
        row_sums = torch.sum(L, dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5))
        
        # 3. The off-diagonal entries should be negative (attractive forces)
        mask = ~torch.eye(L.shape[0], dtype=torch.bool, device=self.device)
        self.assertTrue(torch.all(L[mask] <= 0))
        
        # 4. The diagonal entries should be positive (repulsive forces)
        self.assertTrue(torch.all(torch.diag(L) >= 0))
        
        # 5. For a uniform mesh like this, weights should be roughly equal
        # First, get the non-zero off-diagonal entries
        off_diag_weights = L[mask].reshape(-1).nonzero()
        if len(off_diag_weights) > 0:
            off_diag_weights = L[mask][off_diag_weights]
            # Check that the std deviation is small relative to the mean
            rel_std = torch.std(off_diag_weights) / (torch.mean(torch.abs(off_diag_weights)) + 1e-10)
            self.assertLess(rel_std.item(), 0.5)
    
    def test_compute_mass_matrix(self):
        """Test the mass matrix computation."""
        M = compute_mass_matrix(self.vertices_cube, self.faces_cube)
        
        # Mass matrix should be diagonal
        eye = torch.eye(M.shape[0], device=self.device)
        M_normalized = M / torch.max(torch.abs(M))
        diag_mask = eye > 0
        self.assertTrue(torch.all(M_normalized[~diag_mask] == 0))
        
        # All diagonal entries should be positive (positive masses)
        self.assertTrue(torch.all(torch.diag(M) > 0))
        
        # Total mass should equal the mesh surface area
        total_mass = torch.sum(torch.diag(M))
        
        # Calculate surface area of cube manually (6 faces * 1² area per face)
        v1, v2, v3 = self.vertices_cube[self.faces_cube[:, 0]], self.vertices_cube[self.faces_cube[:, 1]], self.vertices_cube[self.faces_cube[:, 2]]
        face_areas = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1, dim=1), dim=1)
        total_area = torch.sum(face_areas)
        
        self.assertAlmostEqual(total_mass.item(), total_area.item(), places=5)
    
    def test_solve_linear_system(self):
        """Test the linear system solver."""
        # Test with a simple system that has a known solution
        x = solve_linear_system(self.A, self.b)
        
        # Verify Ax = b
        b_computed = torch.matmul(self.A, x)
        # Use higher tolerance for numerical stability
        self.assertTrue(torch.allclose(b_computed, self.b, rtol=1e-4, atol=1e-4))
        
        # Test with singular matrix to check robustness
        A_singular = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], device=self.device)
        
        b_singular = torch.tensor([
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ], device=self.device)
        
        # This should not crash, though solution may be a valid least-squares one
        x_singular = solve_linear_system(A_singular + torch.eye(3, device=self.device) * 1e-6, b_singular)
        self.assertTrue(torch.all(torch.isfinite(x_singular)))
    
    def test_harmonic_deformation_k1(self):
        """Test harmonic deformation (k=1) on a cube."""
        # Define boundary vertices (corners of the cube)
        boundary_vertices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.device)
        
        # Define boundary values (move the top vertices up)
        boundary_values = self.vertices_cube[boundary_vertices].clone()
        boundary_values[4:, 1] += 0.3  # Move top vertices up
        
        # Apply harmonic deformation
        deformed = harmonic_deformation(
            self.vertices_cube,
            self.faces_cube,
            boundary_vertices,
            boundary_values,
            k=1
        )
        
        # Check shape
        self.assertEqual(deformed.shape, self.vertices_cube.shape)
        
        # Check boundary conditions are satisfied
        self.assertTrue(torch.allclose(
            deformed[boundary_vertices],
            boundary_values,
            rtol=1e-5
        ))
        
        # Ensure deformation is reasonable by checking min/max bounds
        self.assertTrue(torch.all(deformed[:, 1] >= self.vertices_cube[:, 1] - 1e-5))
        self.assertTrue(torch.all(deformed[:, 1] <= boundary_values[:, 1].max() + 1e-5))
    
    def test_harmonic_deformation_k2(self):
        """Test biharmonic deformation (k=2) on a cube."""
        # Define boundary vertices (corners of the cube)
        boundary_vertices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device=self.device)
        
        # Define boundary values (move the top vertices up)
        boundary_values = self.vertices_cube[boundary_vertices].clone()
        boundary_values[4:, 1] += 0.3  # Move top vertices up
        
        # Apply biharmonic deformation
        deformed = harmonic_deformation(
            self.vertices_cube,
            self.faces_cube,
            boundary_vertices,
            boundary_values,
            k=2
        )
        
        # Check shape
        self.assertEqual(deformed.shape, self.vertices_cube.shape)
        
        # Check boundary conditions are satisfied
        self.assertTrue(torch.allclose(
            deformed[boundary_vertices],
            boundary_values,
            rtol=1e-5
        ))
        
        # Special case: For our small cube mesh with all corners used as boundary,
        # the harmonic and biharmonic solutions may be numerically similar.
        # Let's add a test with a different mesh to verify this behavior
        
        # Create a larger plane mesh for better testing biharmonic behavior
        vertices_plane, faces_plane = create_plane_mesh(20, self.device)
        
        # Define boundary vertices (outer ring)
        x_coords = vertices_plane[:, 0]
        y_coords = vertices_plane[:, 1]
        boundary_mask = (x_coords == -1) | (x_coords == 1) | (y_coords == -1) | (y_coords == 1)
        plane_boundary_vertices = torch.nonzero(boundary_mask).squeeze()
        
        # Define boundary values (tent shape)
        plane_boundary_values = vertices_plane[plane_boundary_vertices].clone()
        # Make a tent shape
        for i, idx in enumerate(plane_boundary_vertices):
            x, y = vertices_plane[idx, 0], vertices_plane[idx, 1]
            plane_boundary_values[i, 2] = 0.2 * (1 - abs(x)) * (1 - abs(y))
        
        # Apply both harmonic and biharmonic deformations
        deformed_plane_k1 = harmonic_deformation(
            vertices_plane, faces_plane, plane_boundary_vertices, plane_boundary_values, k=1
        )
        
        deformed_plane_k2 = harmonic_deformation(
            vertices_plane, faces_plane, plane_boundary_vertices, plane_boundary_values, k=2
        )
        
        # For this larger mesh, we expect differences between harmonic and biharmonic
        max_diff = torch.max(torch.abs(deformed_plane_k1 - deformed_plane_k2))
        self.assertGreater(max_diff.item(), 1e-4)

if __name__ == '__main__':
    unittest.main() 