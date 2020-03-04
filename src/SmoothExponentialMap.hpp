#pragma once

#include <Eigen/Sparse>
#include "SparseCholeskySolve.hpp"


class SmoothExponentialMap
{
    Eigen::SparseMatrix<double> G, L;
    Eigen::DiagonalMatrix<double, -1> D, M;
    SparseCholeskySolve cholL, cholA;
    std::vector<std::vector<int>> vertexVertex;
    std::vector<std::vector<int>> vertexTriangle;

    std::unique_ptr<int[]> indexBuffer;
    int* indexStartPtr = nullptr;
    int* indexEndPtr = nullptr;
    
    const Eigen::MatrixXd& V;
    
private:
    
    template<typename Derived>
    void
    rescale(Eigen::PlainObjectBase<Derived>& b)
    {
        const auto mi = b.minCoeff();
        const auto d = b.maxCoeff() - mi;
        
        if(d < 1e-12) return;
        
        for(int i = 0; i < b.size(); ++i)
            b.data()[i] = (b.data()[i] - mi) / d;
    }
    
    void
    resetNeighbors();

    std::pair<int*, int*>
    collectNeighbors(const int center, const double r);
    
    double
    gradientOperatorTriangle(const Eigen::Matrix3d& tri, Eigen::Matrix3d& g);
    
    void
    buildOperators(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    
    double
    stepsize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    
    void
    toSparse(const Eigen::VectorXd& X, Eigen::SparseMatrix<double>& Xsp, const int* rows, int rowLen);
    
    Eigen::Matrix<double, 2, 3>
    frameProjector(const Eigen::Vector3d& n);
    
    Eigen::Matrix<double, -1, 2>
    projectRing(const int src);
    
    Eigen::Matrix<double, -1, 2>
    projectRingConformal(const int src);
    
public:
    
    void
    exponentialMap(const int src, const double r, Eigen::MatrixXd& em);
    
    void
    geodesicsInHeat(const int src, const double r, Eigen::VectorXd& dist);
    
    SmoothExponentialMap(const Eigen::MatrixXd& V0, const Eigen::MatrixXi& F);
};
