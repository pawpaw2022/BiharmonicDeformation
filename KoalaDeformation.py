import polyscope as ps
import numpy as np
import igl
import torch
from HarmonicDeformation import harmonic_deformation
import sys
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

class KoalaAnimator:
    def __init__(self):
        try:
            # Check CUDA availability first
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            if self.device.type == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name()}")
                # Set preferred backend
                torch.backends.cuda.preferred_linalg_library('magma')
            
            # Initialize polyscope with error checking
            ps.init()
            # Set better default options
            ps.set_program_name("Koala Deformation")
            ps.set_verbosity(0)
            ps.set_up_dir("y_up")
            ps.set_ground_plane_mode("none")
            ps.set_navigation_style("turntable")
            
            # Load the koala mesh
            V, F = igl.read_triangle_mesh("./koala/koala.obj")
            self.vertices = torch.tensor(V, dtype=torch.float32, device=self.device)
            self.faces = torch.tensor(F, dtype=torch.int64, device=self.device)
            
            # Select vertices for the head (do computation on GPU)
            head_y_threshold = torch.tensor(np.max(V[:, 1]) - 0.1, device=self.device)
            head_indices = torch.nonzero(self.vertices[:, 1] > head_y_threshold).squeeze()
            self.boundary_vertices = head_indices
            
            # Set up boundary conditions
            self.V_bc = self.vertices[self.boundary_vertices].clone()
            self.U_bc = self.V_bc.clone()
            self.U_bc[:, 1] += 0.3  # Move head up by 0.3 units
            
            # Animation state
            self.is_animating = False
            self.bc_frac = 0.0
            self.bc_dir = 0.03
            
            # Register initial mesh
            self.register_current_mesh()
            
            # Set callback functions
            ps.set_user_callback(self.update)
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            sys.exit(1)
        
    def register_current_mesh(self):
        try:
            # Convert current vertices to numpy for polyscope
            vertices_np = self.vertices.cpu().numpy()
            faces_np = self.faces.cpu().numpy()
            
            # Register mesh with polyscope
            ps.register_surface_mesh(
                "koala", 
                vertices_np, 
                faces_np,
                enabled=True,
                smooth_shade=True
            )
            
            # Color the head vertices
            vertex_colors = np.zeros(len(vertices_np))  # 1D array
            vertex_colors[self.boundary_vertices.cpu().numpy()] = 1.0
            ps.get_surface_mesh("koala").add_scalar_quantity(  # Changed to add_scalar_quantity
                "head", 
                vertex_colors,
                enabled=True,
                cmap='reds'
            )
            
        except Exception as e:
            print(f"Error registering mesh: {str(e)}")
            sys.exit(1)

    def update(self):
        try:
            if self.is_animating:
                # Clear CUDA cache before computation
                torch.cuda.empty_cache()
                
                # Update animation parameter
                self.bc_frac += self.bc_dir
                if self.bc_frac >= 1.0 or self.bc_frac <= 0.0:
                    self.bc_dir *= -1
                
                # Show computation status
                ps.imgui.Text("Computing deformation...")
                
                # Interpolate boundary conditions
                current_boundary = self.V_bc + self.bc_frac * (self.U_bc - self.V_bc)
                
                # Compute deformation
                with torch.cuda.device(self.device):
                    deformed = harmonic_deformation(
                        self.vertices, self.faces, self.boundary_vertices, current_boundary, k=2
                    )
                    
                    # Move to CPU and update mesh
                    vertices_np = deformed.cpu().numpy()
                    ps.get_surface_mesh("koala").update_vertex_positions(vertices_np)
                    
                    # Clear memory
                    del deformed, vertices_np
                    torch.cuda.empty_cache()
            
            # Add UI elements
            if ps.imgui.Button("Toggle Animation [d]"):
                self.is_animating = not self.is_animating
                
            ps.imgui.Text(f"Animation Progress: {self.bc_frac:.2f}")
            
        except Exception as e:
            print(f"Error in update: {str(e)}")
            return False

    def show(self):
        try:
            ps.show()
        except Exception as e:
            print(f"Error showing viewer: {str(e)}")
            sys.exit(1)

    # Rename to callback_key_down - this is the correct Polyscope method name
    def callback_key_down(self, key, mods):
        if key == 'd':
            self.is_animating = not self.is_animating
            return True
        return False

if __name__ == "__main__":
    try:
        animator = KoalaAnimator()
        animator.show()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 