#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/per_vertex_normals.h>
#include <igl/unproject_onto_mesh.h>

#include "lodepng.h"
#include "Timer.hpp"
#include "SmoothExponentialMap.hpp"

using Viewer = igl::opengl::glfw::Viewer;

void setTexture(const std::string filename, Viewer& viewer)
{
    std::vector<unsigned char> image; //the raw pixels
    unsigned w, h;
    unsigned error = lodepng::decode(image, w, h, filename);
    
    if(!error)
    {
        Eigen::Matrix<unsigned char, -1, -1> R(w, h), G(w, h), B(w, h);
        
        for(int i = 0; i < w * h; ++i)
        {
            R.data()[i] = image[4 * i];
            G.data()[i] = image[4 * i + 1];
            B.data()[i] = image[4 * i + 2];
        }
        
        viewer.data().set_texture(R, G, B);
    }
}
    
void setUV(const Eigen::MatrixXd& em, const Eigen::MatrixXi& F, Viewer& viewer)
{
    Eigen::MatrixXd em2(em.rows() + 1, 2);
    em2.topRows(em.rows()) = em;
    em2.bottomRows(1).setZero();
    const int def = (int)em.rows();
    
    Eigen::MatrixXi UV_F(F.rows(), 3);
    
    for(int i = 0; i < F.rows(); ++i)
    {
        bool flag = true;
        for(int j = 0; j < 3; ++j)
        {
            if(em(F(i, j),0) == -1)
                flag = false;
        }
        
        if(flag)
        {
            for(int j = 0; j < 3; ++j)
                UV_F(i, j) = F(i, j);
            
        } else
        {
            for(int j = 0; j < 3; ++j)
                UV_F(i, j) = def;
        }
    }
    
    viewer.data().set_uv(em2, UV_F);
}
    
Eigen::MatrixXd V, N;
Eigen::MatrixXi F;
std::unique_ptr<SmoothExponentialMap> sem;

bool mouseDown(Viewer& viewer, int key, int mod)
{
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    
    int fid = -1;
    Eigen::Vector3d bc;
    
    igl::unproject_onto_mesh(Eigen::Vector2f(x,y),
                             viewer.core().view,
                             viewer.core().proj,
                             viewer.core().viewport,
                             V, F, fid, bc);
        
    if(fid != -1)
    {
        const int vid = F(fid, std::distance(bc.data(), std::max_element(bc.data(), bc.data() + 3)));
        
        Eigen::MatrixXd em;
        sem->exponentialMap(vid, 0.25, em);
        setUV(em, F, viewer);
        viewer.data().set_face_based(false);
    }
    
    return false;
}
    
int main(int argc, const char * argv[])
{
    igl::readOFF("../data/bunny.off", V, F);
    std::vector<std::vector<int>> adj;
    igl::adjacency_list(F, adj);
    
    sem.reset(new SmoothExponentialMap(V, F));
    
    Viewer viewer;
    viewer.data().set_mesh(V, F);
    viewer.callback_mouse_down = mouseDown;
    viewer.data().compute_normals();
    viewer.data().show_texture = 1;
    Eigen::Vector3d ambient(.1,.1,.1);
    Eigen::Vector3d diffuse(.7,.7,.7);
    Eigen::Vector3d specular(.9,.9,.9);
    viewer.core().background_color.setOnes();
    viewer.data().uniform_colors(ambient, diffuse, specular);
    setTexture("../data/PolarGrid.png", viewer);
    Eigen::MatrixXd UV(V.rows(), 2);
    viewer.data().set_uv(UV.setZero());
    viewer.data().set_face_based(false);
    viewer.launch();
    
    return 0;
}
