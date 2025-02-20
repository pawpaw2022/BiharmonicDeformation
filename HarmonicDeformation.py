import torch
import numpy as np
from scipy.sparse import coo_matrix
import igl  # Only used to compute cotangent Laplacian for comparison


def compute_cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the cotangent Laplacian matrix with memory optimization.
    
    Parameters:
        vertices: torch.Tensor, shape (V, 3)
            Vertex positions in 3D space
        faces: torch.Tensor, shape (F, 3)
            Face indices defining the mesh topology
            
    Returns:
        torch.Tensor: Sparse tensor of shape (V, V)
            The cotangent Laplacian matrix in sparse format
    """
    device = vertices.device
    
    # Move to CPU, compute Laplacian, and clean up immediately
    with torch.cuda.device(device):
        vertices_np = vertices.cpu().numpy()
        faces_np = faces.cpu().numpy()
        L_np = -igl.cotmatrix(vertices_np, faces_np)
        
        # Convert to sparse tensor efficiently
        L_coo = coo_matrix(L_np)
        indices = torch.LongTensor(np.vstack((L_coo.row, L_coo.col))).to(device)
        values = torch.FloatTensor(L_coo.data).to(device)
        
        # Clear numpy arrays
        del vertices_np, faces_np, L_np
        torch.cuda.empty_cache()
        
        return torch.sparse_coo_tensor(indices, values, L_coo.shape, device=device)

def solve_linear_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve linear system Ax = b with multiple fallback options for robustness.
    
    Parameters:
        A: torch.Tensor, shape (N, N)
            Square coefficient matrix
        b: torch.Tensor, shape (N,)
            Right-hand side vector
            
    Returns:
        torch.Tensor: shape (N,)
            Solution vector x that satisfies Ax = b
            
    Notes:
        Uses three different solution methods in order:
        1. Direct solver (torch.linalg.solve)
        2. QR decomposition if direct solver fails
        3. CPU fallback if GPU methods fail
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
            # Fallback to CPU if GPU solvers fail
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
    Compute k-harmonic deformation of a mesh with memory optimization.
    
    Parameters:
        vertices: torch.Tensor, shape (V, 3)
            Initial vertex positions
        faces: torch.Tensor, shape (F, 3)
            Face indices defining the mesh topology
        boundary_vertices: torch.Tensor, shape (B,)
            Indices of boundary vertices to be constrained
        boundary_values: torch.Tensor, shape (B, 3)
            Target positions for boundary vertices
        k: int, default=1
            Order of harmonic deformation (1=harmonic, 2=biharmonic)
            
    Returns:
        torch.Tensor: shape (V, 3)
            Deformed vertex positions satisfying the boundary conditions
            
    Notes:
        - Uses cotangent Laplacian for better geometric properties
        - Implements memory-efficient GPU computation
        - Handles both harmonic (k=1) and biharmonic (k=2) deformation
        - Uses sparse matrices where possible to reduce memory usage
    """
    device = vertices.device
    V = vertices.shape[0]
    
    # Compute Laplacian matrix
    L = compute_cotangent_laplacian(vertices, faces)
    
    # For biharmonic (k=2), square the Laplacian
    if k == 2:
        L = L @ L
    
    # Create mask for interior vertices (on GPU)
    interior_mask = torch.ones(V, dtype=torch.bool, device=device)
    interior_mask[boundary_vertices] = False
    interior_vertices = torch.nonzero(interior_mask).squeeze()
    
    # Initialize solution
    solution = vertices.clone()
    solution[boundary_vertices] = boundary_values
    
    # Create system matrix A and right-hand side b
    with torch.cuda.device(device):
        A = L.to_dense()  # Necessary for solver
        b = -torch.sparse.mm(L, solution)
        
        # Clear L to free memory
        del L
        torch.cuda.empty_cache()
        
        # Separate system into known and unknown parts
        A_ii = A[interior_mask][:, interior_mask]
        A_ib = A[interior_mask][:, ~interior_mask]
        b_i = b[interior_mask]
        
        # Clear A and b to free memory
        del A
        torch.cuda.empty_cache()
        
        # Add regularization
        eps = 1e-8
        A_ii = A_ii + eps * torch.eye(A_ii.shape[0], device=device)
        
        # Solve the system for each coordinate separately
        rhs = b_i - A_ib @ solution[boundary_vertices]  # Shape: [n_interior, 3]
        del A_ib, b_i
        torch.cuda.empty_cache()
        
        # Solve for each coordinate
        x_i = torch.zeros_like(rhs, device=device)
        for i in range(3):  # x, y, z coordinates
            x_i[:, i] = solve_linear_system(A_ii, rhs[:, i])
        
        # Update solution
        solution[interior_vertices] = x_i
        
        return solution

def demo() -> None:
    """
    Interactive demo showing a deforming cube using Polyscope visualization.
    
    Features:
        - Creates a simple cube mesh
        - Implements real-time harmonic deformation
        - Provides interactive animation controls
        - Uses GPU acceleration when available
        - Visualizes:
            * Mesh geometry with smooth shading
            * Boundary vertices in red
            * Height-based color mapping
            * Edge highlighting
            * Back face differentiation
        
    Controls:
        - Toggle Animation button to start/stop deformation
        - Shows animation progress
        - Interactive camera control via mouse
        
    Notes:
        - Uses CUDA GPU if available
        - Implements memory-efficient computation
        - Maintains consistent visualization settings
    """
    import polyscope as ps
    
    # Initialize polyscope
    ps.init()
    ps.set_program_name("Cube Deformation Demo")
    ps.set_up_dir("y_up")
    
    # Set better camera position and view
    # ps.set_ground_plane_mode("shadow_only")
    # Fix look_at parameters: camera_location, target, fly_to
    # ps.look_at(
    #     camera_location=[1.0, 1.0, 1.0],  # Position camera above and to the side
    #     target=[0.0, 0.0, 0.0],           # Look at center
    #     fly_to=True                        # Animate the camera movement
    # )
    
    # Create a simple cube mesh
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    
    # vertices = torch.tensor([
    #     [-0.0, -0.0, -0.0],
    #     [0.0,  0.0,  0.0],
    #     [0.5,  0.8, -0.5],
    #     [-0.5, 0.8, -0.5],
    #     [0.0,  0.0,  0.0],
    #     [0.0,  0.0,  0.0],
    #     [0.5,  0.8,  0.5],
    #     [-0.5, 0.8,  0.5]
    # ], dtype=torch.float32, device=device)
    
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
        edge_width=2.0,          # Thicker edges
        material="clay",         # Use clay material
        color=[0.8, 0.8, 1.0],  # Light blue color
        transparency=0.5,  # Add transparency value (0.0 = fully transparent, 1.0 = fully opaque)
        enabled=True
    )

    # Color the boundary vertices more visibly
    vertex_colors = torch.zeros(len(vertices), 3, device=device)
    vertex_colors[boundary_vertices] = torch.tensor([1.0, 0.2, 0.2], device=device)  # Brighter red
    ps_mesh.add_color_quantity("boundary", vertex_colors.cpu().numpy(), enabled=True)
    
    # Add height visualization
    vertex_y = vertices.cpu().numpy()[:, 1]  # Y coordinates
    ps_mesh.add_scalar_quantity(
        "height",
        vertex_y,
        enabled=True,
        cmap='viridis',
        vminmax=(vertex_y.min(), vertex_y.max())
    )
    
    # Remove slice plane as it's not needed for this demo
    # ps_plane = ps.add_scene_slice_plane()
    # ps_plane.set_draw_plane(True)
    # ps_plane.set_draw_widget(True)
    
    # Animation state
    bc_frac = 0.0
    bc_dir = 0.03
    animate = False  # New flag to control animation
    
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
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                deformed = harmonic_deformation(
                    vertices, faces, boundary_vertices, current_boundary, k=1
                )
                
                # print(deformed)
                
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
                
                # Clear GPU memory
                del deformed
                torch.cuda.empty_cache()
    
    # Set the callback
    ps.set_user_callback(callback)
    
    # Show the window
    ps.show()

if __name__ == "__main__":
    demo()