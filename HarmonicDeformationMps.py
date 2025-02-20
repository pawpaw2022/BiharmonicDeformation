import torch
import numpy as np
from scipy.sparse import coo_matrix
import igl  # Only used to compute cotangent Laplacian for comparison


def compute_cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the cotangent Laplacian matrix using dense tensors for MPS compatibility.
    """
    device = vertices.device
    
    # Move to CPU, compute Laplacian
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    L_np = -igl.cotmatrix(vertices_np, faces_np)
    
    # Convert sparse matrix to dense numpy array first
    L_dense_np = L_np.todense()
    
    # Convert numpy array to tensor
    L_dense = torch.tensor(L_dense_np, dtype=torch.float32, device=device)
    
    # Clear numpy arrays
    del vertices_np, faces_np, L_np, L_dense_np
    
    return L_dense

def solve_linear_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve linear system Ax = b with multiple fallback options for robustness.
    """
    device = A.device
    try:
        # Try using default solver
        return torch.linalg.solve(A, b)
    except RuntimeError:
        try:
            # Try QR decomposition
            Q, R = torch.linalg.qr(A)
            y = torch.matmul(Q.T, b)
            return torch.triangular_solve(y, R, upper=True)[0]
        except RuntimeError:
            # Fallback to CPU if MPS solvers fail
            A_cpu = A.cpu()
            b_cpu = b.cpu()
            x_cpu = torch.linalg.solve(A_cpu, b_cpu)
            return x_cpu.to(device)

def harmonic_deformation(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    boundary_vertices: torch.Tensor,
    boundary_values: torch.Tensor,
    k: int = 1
) -> torch.Tensor:
    """
    Compute k-harmonic deformation of a mesh using dense tensors for MPS compatibility.
    """
    device = vertices.device
    V = vertices.shape[0]
    
    # Compute Laplacian matrix
    L = compute_cotangent_laplacian(vertices, faces)
    
    # For biharmonic (k=2), square the Laplacian
    if k == 2:
        L = torch.mm(L, L)
    
    # Create mask for interior vertices
    interior_mask = torch.ones(V, dtype=torch.bool, device=device)
    interior_mask[boundary_vertices] = False
    interior_vertices = torch.nonzero(interior_mask).squeeze()
    
    # Initialize solution
    solution = vertices.clone()
    solution[boundary_vertices] = boundary_values
    
    # Create system matrix A and right-hand side b
    b = -torch.mm(L, solution)
    
    # Separate system into known and unknown parts
    A_ii = L[interior_mask][:, interior_mask]
    A_ib = L[interior_mask][:, ~interior_mask]
    b_i = b[interior_mask]
    
    # Clear L to free memory
    del L
    
    # Add regularization
    eps = 1e-8
    A_ii = A_ii + eps * torch.eye(A_ii.shape[0], device=device)
    
    # Solve the system for each coordinate separately
    rhs = b_i - torch.mm(A_ib, solution[boundary_vertices])  # Shape: [n_interior, 3]
    del A_ib, b_i
    
    # Solve for each coordinate
    x_i = torch.zeros_like(rhs, device=device)
    for i in range(3):  # x, y, z coordinates
        x_i[:, i] = solve_linear_system(A_ii, rhs[:, i])
    
    # Update solution
    solution[interior_vertices] = x_i
    
    # print(f"Solution shape: {solution.shape}")
    # print(f"Solution: {solution[0]}")
    
    return solution

def demo() -> None:
    """
    Interactive demo showing a deforming cube using Polyscope visualization.
    """
    import polyscope as ps
    
    # Initialize polyscope
    ps.init()
    ps.set_program_name("Cube Deformation Demo")
    ps.set_up_dir("y_up")
    
    # Create a simple cube mesh
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create vertices for a cube (8 vertices) - scaled up by 0.5
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
    
    # Create faces (12 triangles) - ensure correct winding order
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [5, 6, 7], [5, 7, 4],  # back
        [4, 7, 3], [4, 3, 0],  # left
        [3, 7, 6], [3, 6, 2],  # top
        [4, 0, 1], [4, 1, 5],  # bottom
    ], dtype=torch.int64, device=device)
    
    # Define both top and bottom vertices as boundary
    boundary_vertices = torch.tensor([0,1,4,5, 2,3,6,7], device=device)  # bottom + top vertices
    
    # Original and target positions
    V_bc = vertices[boundary_vertices].clone()
    U_bc = V_bc.clone()
    
    # Only move top vertices up (indices 4-7 in boundary_vertices correspond to vertices 2,3,6,7)
    U_bc[4:, 1] += 0.3  # Move only top vertices up by 0.3
    
    # Register initial mesh with better visualization options
    ps_mesh = ps.register_surface_mesh(
        "cube",
        vertices.cpu().numpy(),
        faces.cpu().numpy(),
        smooth_shade=True,
        edge_width=2.0,
        material="clay",
        color=[0.8, 0.8, 1.0],
        transparency=0.5,
        enabled=True
    )

    # Color the boundary vertices more visibly
    vertex_colors = torch.zeros(len(vertices), 3, device=device)
    vertex_colors[boundary_vertices] = torch.tensor([1.0, 0.0, 0.0], device=device)  # Bright red
    ps_mesh.add_color_quantity("boundary", vertex_colors.cpu().numpy(), enabled=True)
    
    # Add boundary vertices as points for better visibility
    boundary_points = vertices[boundary_vertices].cpu().numpy()
    ps_points = ps.register_point_cloud("boundary_points", boundary_points)
    ps_points.set_color([1.0, 0.0, 0.0])  # Bright red
    ps_points.set_radius(0.02)  # Make points larger
    
    # Add height visualization
    vertex_y = vertices.cpu().numpy()[:, 1]  # Y coordinates
    ps_mesh.add_scalar_quantity(
        "height",
        vertex_y,
        enabled=True,
        cmap='viridis',
        vminmax=(vertex_y.min(), vertex_y.max())
    )
    
    # Animation state
    bc_frac = 0.0
    bc_dir = 0.03
    animate = False
    
    def callback():
        nonlocal bc_frac, bc_dir, animate
        
        # Add ImGui button for animation control
        if ps.imgui.Button("Toggle Animation"):
            animate = not animate
        
        # Show animation state
        ps.imgui.Text(f"Animation {'Running' if animate else 'Paused'}")
        ps.imgui.Text(f"Progress: {bc_frac:.2f}")
        
        # Only update if animation is enabled
        if animate:
            # Update animation parameter
            bc_frac += bc_dir
            if bc_frac >= 1.0 or bc_frac <= 0.0:
                bc_dir *= -1
            
            # Interpolate boundary conditions
            current_boundary = V_bc + bc_frac * (U_bc - V_bc)
            
            # Compute deformation
            deformed = harmonic_deformation(
                vertices, faces, boundary_vertices, current_boundary, k=2
            )
            
            # Update mesh
            vertices_np = deformed.cpu().numpy()
            ps_mesh.update_vertex_positions(vertices_np)
            
            # Ensure visualization settings remain consistent
            ps_mesh.set_material("clay")
            ps_mesh.set_color([0.3, 0.5, 1.0])
            ps_mesh.set_edge_color([0.0, 0.0, 0.0])
            ps_mesh.set_smooth_shade(True)
            
            # Update height visualization
            vertex_y = vertices_np[:, 1]  # Y coordinates
            ps_mesh.add_scalar_quantity(
                "height",
                vertex_y,
                enabled=True,
                cmap='viridis',
                vminmax=(vertex_y.min(), vertex_y.max())
            )
            
            # Clear memory
            del deformed
    
    # Set the callback
    ps.set_user_callback(callback)
    
    # Show the window
    ps.show()

if __name__ == "__main__":
    demo() 