import torch
import argparse

def compute_angles(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor) -> torch.Tensor:
    """
    Compute angles at v1 for triangles defined by vertices v1, v2, v3.
    Returns cosine of angles for stable computation.
    """
    e1 = v2 - v1 
    e2 = v3 - v1 
    
    e1_norm = torch.norm(e1, dim=1, keepdim=True)
    e2_norm = torch.norm(e2, dim=1, keepdim=True)
    
    cos_angle = torch.sum(e1 * e2, dim=1) / (e1_norm.squeeze() * e2_norm.squeeze())
    
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    
    return cos_angle

def compute_cotangent_laplacian(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the cotangent Laplacian matrix using sparse construction for better performance.
    """
    V = vertices.shape[0]    
    
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
        
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
    
    i = faces[:, [1, 2, 2, 0, 0, 1]].reshape(-1)    
    j = faces[:, [2, 1, 0, 2, 1, 0]].reshape(-1)
    
    values = torch.cat([
        0.5 * cot_1,
        0.5 * cot_2,
        0.5 * cot_3,
        0.5 * cot_1,
        0.5 * cot_2,
        0.5 * cot_3
    ])
    
    indices = torch.stack([i, j])
    L_sparse = torch.sparse_coo_tensor(indices, -values, (V, V))
    
    L = L_sparse.to_dense()
    L = L + L.T
    
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

def compute_mass_matrix(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute the diagonal mass matrix M for a given mesh.
    Uses Voronoi area (1/3 of each face's area is assigned to its vertices).
    """
    V = vertices.shape[0]
    
    v1, v2, v3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    
    face_areas = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1, dim=1), dim=1)
    
    # Distribute 1/3 of each face's area to its vertices
    M = torch.zeros(V, device=vertices.device)
    for i in range(3):
        M.scatter_add_(0, faces[:, i], face_areas / 3)
    
    M = torch.diag(M)
    
    return M

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
        M = compute_mass_matrix(vertices, faces)    
        M_inv = torch.linalg.solve(M, torch.eye(V, device=device))  # M⁻¹
        L = torch.mm(M_inv, L)  # M⁻¹L
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
    
    del L
    
    eps = 1e-8
    A_ii = A_ii + eps * torch.eye(A_ii.shape[0], device=device)
    
    rhs = b_i - torch.mm(A_ib, solution[boundary_vertices])
    del A_ib, b_i
    
    x_i = torch.zeros_like(rhs, device=device)
    for i in range(3):
        x_i[:, i] = solve_linear_system(A_ii, rhs[:, i])
    
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
    
    ps.init()
    ps.set_program_name("Cube Deformation Demo")
    ps.set_up_dir("y_up")
    
    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

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
    
    boundary_vertices = torch.tensor([0,1,4,5, 2,3,6,7], device=device)
    
    V_bc = vertices[boundary_vertices].clone()
    U_bc = V_bc.clone()
    
    U_bc[4:, 1] += 0.3 
    
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

    vertex_colors = torch.zeros(len(vertices), 3, device=device)
    vertex_colors[boundary_vertices] = torch.tensor([1.0, 0.2, 0.2], device=device) 
    ps_mesh.add_color_quantity("boundary", vertex_colors.cpu().numpy(), enabled=True)
    
    vertex_y = vertices.cpu().numpy()[:, 1] 
    ps_mesh.add_scalar_quantity(
        "height",
        vertex_y,
        enabled=True,
        cmap='viridis',
        vminmax=(vertex_y.min(), vertex_y.max())
    )
    
    bc_frac = 0.0
    bc_dir = 0.03
    animate = False 
    
    boundary_points = vertices[boundary_vertices].cpu().numpy()
    ps_points = ps.register_point_cloud("boundary_points", boundary_points)
    ps_points.set_color([1.0, 0.0, 0.0]) 
    ps_points.set_radius(0.02) 
    
    def callback():
        nonlocal bc_frac, bc_dir, animate
        
        if ps.imgui.Button("Toggle Animation"):
            animate = not animate
        
        ps.imgui.Text(f"Animation {'Running' if animate else 'Paused'}")
        ps.imgui.Text(f"Progress: {bc_frac:.2f}")
        
        if animate:
            bc_frac += bc_dir
            if bc_frac >= 1.0 or bc_frac <= 0.0:
                bc_dir *= -1
            
            current_boundary = V_bc + bc_frac * (U_bc - V_bc)
            
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    deformed = harmonic_deformation(
                        vertices, faces, boundary_vertices, current_boundary, k=2
                    )
                    vertices_np = deformed.cpu().numpy()
                    update_visualization(ps_mesh, vertices_np)
                    del deformed
                    torch.cuda.empty_cache()
            else:
                deformed = harmonic_deformation(
                    vertices, faces, boundary_vertices, current_boundary, k=1
                )
                vertices_np = deformed.cpu().numpy()
                update_visualization(ps_mesh, vertices_np)
                del deformed

    def update_visualization(ps_mesh, vertices_np):
        """Helper function to update mesh visualization"""
        ps_mesh.update_vertex_positions(vertices_np)
        
        ps_mesh.set_material("clay")
        ps_mesh.set_color([0.3, 0.5, 1.0])
        ps_mesh.set_edge_color([0.0, 0.0, 0.0])
        ps_mesh.set_smooth_shade(True)
        
        vertex_y = vertices_np[:, 1] 
        ps_mesh.add_scalar_quantity(
            "height",
            vertex_y,
            enabled=True,
            cmap='viridis',
            vminmax=(vertex_y.min(), vertex_y.max())
        )
    
    ps.set_user_callback(callback)
    
    ps.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonic Deformation Demo")
    parser.add_argument("--mps", action="store_true", help="Use MPS (Metal Performance Shaders) for Mac devices")
    args = parser.parse_args()
    
    demo(mps=args.mps)