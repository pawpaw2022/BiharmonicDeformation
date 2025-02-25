import torch
import argparse  # Added for command-line argument parsing

def compute_angles(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    """
    Compute angles at v1 for triangles defined by vertices v1, v2, v3.
    Returns cosine of angles for stable computation.
    """
    e1 = v2 - v1  # edge from v1 to v2
    e2 = v3 - v1  # edge from v1 to v3
    
    # Normalize edges
    e1_norm = torch.norm(e1, dim=1, keepdim=True)
    e2_norm = torch.norm(e2, dim=1, keepdim=True)
    
    # Compute cosine using dot product of normalized vectors
    cos_angle = torch.sum(e1 * e2, dim=1) / (e1_norm.squeeze() * e2_norm.squeeze())
    
    # Clamp for numerical stability
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    
    return cos_angle

def compute_cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the cotangent Laplacian matrix using sparse construction for better performance.
    """
    V = vertices.shape[0]    
    
    # Get vertices of triangles
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # Compute angles at each vertex of triangles
    cos_1 = compute_angles(v1, v2, v3)
    cos_2 = compute_angles(v2, v3, v1)
    cos_3 = compute_angles(v3, v1, v2)
    
    # Convert cosine to cotangent using: cot(x) = cos(x)/sin(x)
    sin_1 = torch.sqrt(1 - cos_1 * cos_1) # sin(x) = sqrt(1 - cos(x)^2)
    sin_2 = torch.sqrt(1 - cos_2 * cos_2)
    sin_3 = torch.sqrt(1 - cos_3 * cos_3)
    
    cot_1 = cos_1 / sin_1   # cot(x) = cos(x)/sin(x)
    cot_2 = cos_2 / sin_2
    cot_3 = cos_3 / sin_3
    
    # Prepare indices and values for sparse matrix construction
    i = faces[:, [1, 2, 2, 0, 0, 1]].reshape(-1)    
    j = faces[:, [2, 1, 0, 2, 1, 0]].reshape(-1)
    
    # Each edge gets contribution from both adjacent triangles
    values = torch.cat([
        0.5 * cot_1,
        0.5 * cot_2,
        0.5 * cot_3,
        0.5 * cot_1,
        0.5 * cot_2,
        0.5 * cot_3
    ])
    
    # Create sparse matrix in COO format
    indices = torch.stack([i, j])
    L_sparse = torch.sparse_coo_tensor(indices, -values, (V, V))
    
    # Convert to dense and make symmetric
    L = L_sparse.to_dense()
    L = L + L.T
    
    # Set diagonal elements to negative sum of off-diagonal elements
    L.diagonal().copy_(-torch.sum(L, dim=1))
    
    return L

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
        return torch.linalg.solve(A, b)
    except RuntimeError:
        try:
            # QR decomposition
            Q, R = torch.linalg.qr(A)
            y = torch.matmul(Q.T, b)
            return torch.triangular_solve(y, R, upper=True)[0]
        except RuntimeError:
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
    Compute k-harmonic deformation of a mesh using dense tensors for better compatibility.
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
    b = -torch.mm(L, solution) # b = -L^2x
    
    # Separate system into known and unknown parts
    A_ii = L[interior_mask][:, interior_mask] # A_ii is the submatrix of L corresponding to the interior vertices
    A_ib = L[interior_mask][:, ~interior_mask] # A_ib is the submatrix of L corresponding to the boundary vertices
    b_i = b[interior_mask] # b_i is the right-hand side vector corresponding to the interior vertices
    
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
    
    return solution

def demo(mps: bool = False) -> None:
    """
    Interactive demo showing a deforming cube using Polyscope visualization.
    
    Parameters:
        mps: bool, default False
            If True, uses MPS (Metal Performance Shaders) for Mac devices
            If False, tries to use CUDA if available, otherwise falls back to CPU
    """
    import polyscope as ps
    
    # Initialize polyscope
    ps.init()
    ps.set_program_name("Cube Deformation Demo")
    ps.set_up_dir("y_up")
    
    # Device selection logic
    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple cube mesh
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
    
    # Animation state
    bc_frac = 0.0
    bc_dir = 0.03
    animate = False  # New flag to control animation
    
    # Add boundary vertices as points for better visibility
    boundary_points = vertices[boundary_vertices].cpu().numpy()
    ps_points = ps.register_point_cloud("boundary_points", boundary_points)
    ps_points.set_color([1.0, 0.0, 0.0])  # Bright red
    ps_points.set_radius(0.02)  # Make points larger
    
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
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    deformed = harmonic_deformation(
                        vertices, faces, boundary_vertices, current_boundary, k=1
                    )
                    # Update mesh and visualization
                    vertices_np = deformed.cpu().numpy()
                    update_visualization(ps_mesh, vertices_np)
                    # Clear GPU memory
                    del deformed
                    torch.cuda.empty_cache()
            else:
                # For CPU and MPS devices
                deformed = harmonic_deformation(
                    vertices, faces, boundary_vertices, current_boundary, k=1
                )
                # Update mesh and visualization
                vertices_np = deformed.cpu().numpy()
                update_visualization(ps_mesh, vertices_np)
                # Clear memory
                del deformed

    def update_visualization(ps_mesh, vertices_np):
        """Helper function to update mesh visualization"""
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
    
    # Set the callback
    ps.set_user_callback(callback)
    
    # Show the window
    ps.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Harmonic Deformation Demo")
    parser.add_argument("--mps", action="store_true", help="Use MPS (Metal Performance Shaders) for Mac devices")
    args = parser.parse_args()
    
    # Run demo with MPS flag if specified
    demo(mps=args.mps)