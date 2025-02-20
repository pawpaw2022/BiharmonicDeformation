import polyscope as ps
import numpy as np
import trimesh

# Initialize polyscope
ps.init()

# Load the koala mesh using trimesh
mesh = trimesh.load("koala/koala.obj")

# Extract vertices and faces
verts = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# Register the mesh with polyscope
ps_mesh = ps.register_surface_mesh(
    "koala", 
    verts, 
    faces,
    smooth_shade=True,
    material="clay",
    color=[0.8, 0.8, 1.0],
    edge_width=1.0
)

# Add some visualization features
# Height-based coloring
vertex_y = verts[:, 1]  # Y coordinates
ps_mesh.add_scalar_quantity(
    "height",
    vertex_y,
    enabled=True,
    cmap='viridis'
)

# Show the mesh in the 3D UI
ps.show()