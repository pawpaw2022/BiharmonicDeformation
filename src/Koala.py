import polyscope as ps
import numpy as np
import torch
import pymeshlab
from HarmonicDeformation import harmonic_deformation

def load_and_simplify_mesh(mesh_path: str, target_vertices: int) -> pymeshlab.MeshSet:
    """
    Load and simplify a mesh using pymeshlab
    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_path)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_vertices*2)
    return ms

def setup_mesh_data(mesh: pymeshlab.MeshSet, device: torch.device) -> tuple:
    """
    Convert mesh data to torch tensors and setup boundary conditions
    """
    verts = mesh.current_mesh().vertex_matrix()
    faces = mesh.current_mesh().face_matrix()
    
    vertices = torch.tensor(verts, dtype=torch.float32, device=device)
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # Select all vertices as boundary vertices
    boundary_vertices = torch.arange(len(vertices), device=device)
    V_bc = vertices[boundary_vertices].clone()
    
    # Find head region vertices
    front_vertices_mask = V_bc[:, 2] > torch.quantile(V_bc[:, 2], 0.55)
    height_range_mask = (V_bc[:, 1] > torch.quantile(V_bc[:, 1], 0.4)) & (V_bc[:, 1] < torch.quantile(V_bc[:, 1], 1.0))
    head_vertices_mask = front_vertices_mask & height_range_mask
    moving_indices = torch.where(head_vertices_mask)[0]
    
    # Setup target positions
    U_bc = V_bc.clone()
    U_bc[moving_indices, 2] += 0.5  # Forward movement
    U_bc[moving_indices, 1] += 0.5  # Upward movement
    
    return vertices, faces, boundary_vertices, V_bc, U_bc, moving_indices

def create_polyscope_mesh(vertices: torch.Tensor, faces: torch.Tensor, moving_indices: torch.Tensor) -> ps.SurfaceMesh:
    """
    Create and setup the polyscope mesh visualization
    """
    ps_mesh = ps.register_surface_mesh(
        "koala", 
        vertices.cpu().numpy(), 
        faces.cpu().numpy(),
        smooth_shade=True,
        material="clay",
        color=[0.8, 0.8, 1.0],
        edge_width=1.0
    )
    
    # Highlight moving vertices
    vertex_colors = np.zeros((len(vertices), 3))
    vertex_colors[moving_indices.cpu().numpy()] = [1.0, 0.0, 0.0]
    ps_mesh.add_color_quantity("boundary", vertex_colors, enabled=True)
    
    return ps_mesh

def main(mesh_path: str = "koala/koala.obj", target_vertices: int = 1000, mps: bool = False) -> None:
    """
    Main function to run the Koala mesh deformation demo
    
    Parameters:
        mesh_path: str, path to the mesh file
        target_vertices: int, target number of vertices after simplification
        mps: bool, whether to use MPS (Metal Performance Shaders) instead of CUDA
    """
    # Initialize polyscope
    ps.init()
    
    # Setup device
    if mps:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process mesh
    ms = load_and_simplify_mesh(mesh_path, target_vertices)
    vertices, faces, boundary_vertices, V_bc, U_bc, moving_indices = setup_mesh_data(ms, device)
    ps_mesh = create_polyscope_mesh(vertices, faces, moving_indices)
    
    # Animation state
    animation_state = {
        'bc_frac': 0.0,
        'bc_dir': 0.03,
        'animate': False
    }
    
    def callback():
        if ps.imgui.Button("Toggle Animation"):
            animation_state['animate'] = not animation_state['animate']
        
        ps.imgui.Text(f"Animation {'Running' if animation_state['animate'] else 'Paused'}")
        ps.imgui.Text(f"Progress: {animation_state['bc_frac']:.2f}")
        
        if animation_state['animate']:
            # Update animation parameter
            animation_state['bc_frac'] += animation_state['bc_dir']
            if animation_state['bc_frac'] >= 1.0 or animation_state['bc_frac'] <= 0.0:
                animation_state['bc_dir'] *= -1
            
            # Interpolate boundary conditions
            current_boundary = V_bc + animation_state['bc_frac'] * (U_bc - V_bc)
            
            # Compute deformation
            deformed = harmonic_deformation(
                vertices, faces, boundary_vertices, current_boundary, k=2
            )
            
            # Update mesh
            vertices_np = deformed.cpu().numpy()
            ps_mesh.update_vertex_positions(vertices_np)
            
            # Update visualization
            vertex_y = vertices_np[:, 1]
            ps_mesh.add_scalar_quantity(
                "height",
                vertex_y,
                enabled=True,
                cmap='viridis'
            )
    
    # Set the callback
    ps.set_user_callback(callback)
    
    # Show the mesh in the 3D UI
    ps.show()

if __name__ == "__main__":
    main()