#include <igl/colon.h>
#include <igl/harmonic.h>
#include <igl/readOBJ.h>
#include <igl/readDMAT.h>
#include <igl/opengl/glfw/Viewer.h>
#include <algorithm>
#include <iostream>

double bc_frac = 1.0;
double bc_dir = -0.03;
bool deformation_field = false;
Eigen::MatrixXd V,U,V_bc,U_bc;
Eigen::VectorXd Z;
Eigen::MatrixXi F;
Eigen::VectorXi b;

bool pre_draw(igl::opengl::glfw::Viewer & viewer)
{
  using namespace Eigen;
  // Handle animation of boundary conditions
  if(viewer.core().is_animating)
  {
    bc_frac += bc_dir;  // Update animation parameter
    // Reverse direction if we hit the limits
    bc_dir *= (bc_frac>=1.0 || bc_frac<=0.0?-1.0:1.0);
  }

  // Interpolate boundary conditions based on animation parameter
  const MatrixXd U_bc_anim = V_bc+bc_frac*(U_bc-V_bc);
  
  if(deformation_field)
  {
    // Compute harmonic deformation field
    MatrixXd D;
    MatrixXd D_bc = U_bc_anim - V_bc;  // Boundary displacements
    igl::harmonic(V,F,b,D_bc,2,D);     // Solve for interior displacements
    U = V+D;                            // Apply displacements
  }else
  {
    // Compute harmonic deformation directly on positions
    igl::harmonic(V,F,b,U_bc_anim,2.,U);
  }
  // Update mesh vertices and recompute normals
  viewer.data().set_vertices(U);
  viewer.data().compute_normals();
  return false;
}

bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int mods)
{
  switch(key)
  {
    case ' ':  // Space toggles animation
      viewer.core().is_animating = !viewer.core().is_animating;
      return true;
    case 'D':  // D/d toggles deformation mode
    case 'd':
      deformation_field = !deformation_field;
      return true;
  }
  return false;
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load mesh
  igl::readOBJ(TUTORIAL_SHARED_PATH "/decimated-max.obj",V,F);
  U=V;  // Initialize deformed vertices to original positions

  // Load selection data (handle assignments for vertices)
  VectorXi S;
  igl::readDMAT(TUTORIAL_SHARED_PATH "/decimated-max-selection.dmat",S);

  // Create list of boundary vertices (those with handle assignments)
  igl::colon<int>(0,V.rows()-1,b);
  b.conservativeResize(stable_partition( b.data(), b.data()+b.size(),
   [&S](int i)->bool{return S(i)>=0;})-b.data());

  // Set boundary conditions for each handle
  U_bc.resize(b.size(),V.cols());
  V_bc.resize(b.size(),V.cols());
  for(int bi = 0;bi<b.size();bi++)
  {
    V_bc.row(bi) = V.row(b(bi));  // Original position
    switch(S(b(bi)))
    {
      case 0:  // Handle 0: No movement
        U_bc.row(bi) = V.row(b(bi));
        break;
      case 1:  // Handle 1: Move down 50 units
        U_bc.row(bi) = V.row(b(bi)) + RowVector3d(0,-50,0);
        break;
      case 2:  // Handle 2 and others: Move forward 25 units
      default:
        U_bc.row(bi) = V.row(b(bi)) + RowVector3d(0,0,-25);
        break;
    }
  }

  // Color faces based on whether they're in a handle
  MatrixXd C(F.rows(),3);
  RowVector3d purple(80.0/255.0,64.0/255.0,255.0/255.0);
  RowVector3d gold(255.0/255.0,228.0/255.0,58.0/255.0);
  for(int f = 0;f<F.rows();f++)
  {
    // If all vertices of face are in a handle, color purple, else gold
    if( S(F(f,0))>=0 && S(F(f,1))>=0 && S(F(f,2))>=0)
    {
      C.row(f) = purple;
    }else
    {
      C.row(f) = gold;
    }
  }

  // Setup viewer and visualization
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(U, F);
  viewer.data().show_lines = false;
  viewer.data().set_colors(C);
  viewer.core().trackball_angle = Eigen::Quaternionf(sqrt(2.0),0,sqrt(2.0),0);
  viewer.core().trackball_angle.normalize();
  viewer.callback_pre_draw = &pre_draw;
  viewer.callback_key_down = &key_down;
  viewer.core().animation_max_fps = 30.;

  // Print usage instructions
  cout<<
    "Press [space] to toggle deformation."<<endl<<
    "Press 'd' to toggle between biharmonic surface or displacements."<<endl;
  
  // Launch viewer
  viewer.launch();
}
