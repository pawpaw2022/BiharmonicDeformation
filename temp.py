import polyscope as ps
import numpy as np
import trimesh
import torch
import pymeshlab
from HarmonicDeformationMps import harmonic_deformation

# Initialize polyscope
ps.init()

# Load and simplify the koala mesh using pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh("koala/koala.obj")
# Simplify to target number of vertices 
# Fix the OOM error by reducing the number of faces per vertex !!! 

target_vertices = 1000
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_vertices*2)  # roughly 2 faces per vertex

# Get the simplified mesh
mesh_simplified = ms.current_mesh()
verts = mesh_simplified.vertex_matrix()
faces = mesh_simplified.face_matrix()

# Convert to torch tensors and move to MPS if available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
vertices = torch.tensor(verts, dtype=torch.float32, device=device)
faces = torch.tensor(faces, dtype=torch.int64, device=device)

# Select all vertices as boundary vertices
boundary_vertices = torch.arange(len(vertices), device=device)

# Original positions for all vertices
V_bc = vertices[boundary_vertices].clone()

# Find indices of vertices in the head region
# Assuming the koala is facing the positive z direction
front_vertices_mask = V_bc[:, 2] > torch.quantile(V_bc[:, 2], 0.6)  # Extended to front 40% to include all of ears
height_range_mask = (V_bc[:, 1] > torch.quantile(V_bc[:, 1], 0.4)) & (V_bc[:, 1] < torch.quantile(V_bc[:, 1], 1.0))  # Keep same height range
head_vertices_mask = front_vertices_mask & height_range_mask
moving_indices = torch.where(head_vertices_mask)[0]

# Target positions are same as original, except for head vertices
U_bc = V_bc.clone()
U_bc[moving_indices, 2] += 0.5  # Keep same forward movement
U_bc[moving_indices, 1] += 0.4  # Keep same upward movement

# Register the initial mesh with polyscope
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
vertex_colors[moving_indices.cpu().numpy()] = [1.0, 0.0, 0.0]  # Red for moving vertices
ps_mesh.add_color_quantity("boundary", vertex_colors, enabled=True)

# Animation state
bc_frac = 0.0
bc_dir = 0.03
animate = False

def callback():
    global bc_frac, bc_dir, animate
    
    if ps.imgui.Button("Toggle Animation"):
        animate = not animate
    
    ps.imgui.Text(f"Animation {'Running' if animate else 'Paused'}")
    ps.imgui.Text(f"Progress: {bc_frac:.2f}")
    
    if animate:
        # Update animation parameter
        bc_frac += bc_dir
        if bc_frac >= 1.0 or bc_frac <= 0.0:
            bc_dir *= -1
        
        # Interpolate boundary conditions (using V_bc and U_bc naming)
        current_boundary = V_bc + bc_frac * (U_bc - V_bc)
        
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