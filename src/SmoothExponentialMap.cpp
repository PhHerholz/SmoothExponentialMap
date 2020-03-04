#include "SmoothExponentialMap.hpp"
#include <igl/adjacency_list.h>
#include <igl/vertex_triangle_adjacency.h>
#include <igl/cotmatrix.h>
#include <igl/doublearea.h>
#include <igl/massmatrix.h>
#include <igl/cotmatrix.h>

namespace std
{
    // enables the use of pointer pairs in range based for loops
    int* begin(std::pair<int*, int*>& p) {return p.first;}
    int* end(std::pair<int*, int*>& p) {return p.second;}
    
    const int* cbegin(const std::pair<const int*, const int*>& p) {return p.first;}
    const int* cend(const std::pair<const int*, const int*>& p) {return p.second;}
}

void
SmoothExponentialMap::resetNeighbors()
{
    indexStartPtr = nullptr;
}

std::pair<int*, int*>
SmoothExponentialMap::collectNeighbors(const int center, const double r)
{
    std::vector<char> flag(V.rows());
    
    int* queue = &indexBuffer[0];
    indexEndPtr = indexStartPtr = queue + V.rows();
    *queue++ = center;
    flag[center] = 1;
    
    const double r2 = r * r;
    Eigen::RowVector3d centerVertex = V.row(center);
    
    while(queue != &indexBuffer[0])
    {
        const int i = *--queue;
        
        if( (V.row(i) - centerVertex).squaredNorm() < r2 )
        {
            *--indexStartPtr = i;
            
            for(int j : vertexVertex[i])
            {
                if(!flag[j])
                {
                    flag[j] = 1;
                    *queue++ = j;
                }
            }
        }
    }
    
    return std::make_pair(indexStartPtr, indexEndPtr);
}

double
SmoothExponentialMap::gradientOperatorTriangle(const Eigen::Matrix3d& tri, Eigen::Matrix3d& g)
{
    g.setZero();
    Eigen::Vector3d n = (tri.col(0) - tri.col(2)).cross(tri.col(1) - tri.col(2));
    
    for(int j = 0; j < 3; ++j)
    {
        g.col(j) = n.cross( tri.col( (j+2)%3 ) - tri.col( (j+1)%3 ));
    }
    
    const double nrm2 = n.dot(n);
    g /= nrm2;
    
    return 0.5 * sqrt(nrm2);
}

void
SmoothExponentialMap::buildOperators(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    std::vector<Eigen::Triplet<double>> trip;
    Eigen::Matrix3d tri, g;
    
    M.resize(V.rows());
    M.diagonal().setZero();
    
    D.resize(3 * F.rows());
    
    for(int i = 0; i < F.rows(); ++i)
    {
        for(int j = 0; j < 3; ++j)
            tri.col(j) = V.row(F(i,j)).transpose();
        
        const double area = gradientOperatorTriangle(tri, g);
        
        for(int j = 0; j < 3; ++j)
        {
            M.diagonal()(F(i, j)) += area / 3.;
            D.diagonal()(3 * i + j) = area;
        }
        
        for(int j = 0; j < 3; ++j)
            for(int k = 0; k < 3; ++k)
                trip.emplace_back(3 * i + j, F(i, k), g(j, k));
        
    }
    
    G.resize(3 * F.rows(), V.rows());
    G.setFromTriplets(trip.begin(), trip.end());
    
    L = G.transpose() * D * G;
}

double
SmoothExponentialMap::stepsize(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    double len = .0;
    for(int i = 0; i < F.rows(); ++i)
        for(int j = 0; j < 3; ++j)
            len += (V.row(F(i, j)) - V.row(F(i,(j+1)%3))).norm();
    
    return 50 * std::pow(len / (3. * F.rows()), 2);
}

void
SmoothExponentialMap::toSparse(const Eigen::VectorXd& X, Eigen::SparseMatrix<double>& Xsp, const int* rows, int rowLen)
{
    if(!rows)
    {
        rowLen = 0;
        
        for(int i = 0; i < X.rows(); ++i)
        {
            if(X(i)) ++rowLen;
        }
    }
    
    
    Xsp.resize(X.rows(), 1);
    Xsp.resizeNonZeros(rowLen);
    Xsp.outerIndexPtr()[0] = 0;
    Xsp.outerIndexPtr()[1] = rowLen;
    
    if(rows)
    {
        for(int i = 0; i < rowLen; ++i)
        {
            Xsp.innerIndexPtr()[i] = rows[i];
            Xsp.valuePtr()[i] = X(rows[i]);
        }
    } else
    {
        int cnt = 0;
        
        for(int i = 0; i < X.rows(); ++i)
        {
            if(X(i))
            {
                Xsp.innerIndexPtr()[cnt] = i;
                Xsp.valuePtr()[cnt] = X(i);
                ++cnt;
            }
        }
    }
}


Eigen::Matrix<double, 2, 3>
SmoothExponentialMap::frameProjector(const Eigen::Vector3d& n)
{
    Eigen::Matrix<double, 2, 3> frame;
    
    if(n(0) > 1e-5)
    {
        frame(0, 0) = 0;
        frame(0, 1) = 1;
        frame(0, 2) = 0;
    } else
    {
        frame(0, 0) = 1;
        frame(0, 1) = 0;
        frame(0, 2) = 0;
    }
    
    frame.row(0) -= n.dot(frame.row(0)) * n ;
    frame.row(0).normalize();
    frame.row(1) = n.cross(frame.row(0)).normalized();
    
    return frame;
}


Eigen::Matrix<double, -1, 2>
SmoothExponentialMap::projectRing(const int src)
{
    auto nbh = vertexVertex[src];
    const int nn = (int)nbh.size();
    Eigen::Vector3d vs = V.row(src);
    
    // compute vector area
    Eigen::Vector3d n;
    n.setZero();
    
    for(int i = 0; i < nn; ++i)
    {
        Eigen::Vector3d a = V.row(nbh[i]).transpose() - vs;
        Eigen::Vector3d b = V.row(nbh[(i+1)%nn]).transpose() - vs;
        
        n += a.cross(b);
    }
    
    // build projector matrix
    auto frame = frameProjector(n);
    
    // project ring
    Eigen::Matrix<double, -1, 2> ring(nn, 2);
    
    for(int i = 0; i < nn; ++i)
        ring.row(i) = frame * (V.row(nbh[i]).transpose() - vs);
    
    return ring;
}

Eigen::Matrix<double, -1, 2>
SmoothExponentialMap::projectRingConformal(const int src)
{
    auto nbh = vertexVertex[src];
    const int nn = (int)nbh.size();
    const Eigen::RowVector3d vs = V.row(src);
    
    std::vector<double> angles(nn);
    std::vector<double> dist(nn);
    
    double total = .0;
    
    for(int i = 0; i < nn; ++i)
    {
        // compute angle
        Eigen::Vector3d e0 = V.row(nbh[i]) - vs;
        Eigen::Vector3d e1 = V.row(nbh[(i+1)%nn]) - vs;
        
        dist[i] = e0.norm();
        angles[i] = acos(e0.dot(e1) / (e1.norm() * dist[i]));
        total += angles[i];
    }
    
    
    Eigen::Matrix<double, -1, 2> ring(nn, 2);
    
    double ang = .0;
    for(int i = 0; i < nn; ++i)
    {
        ring(i, 0) = dist[i] * sin(ang);
        ring(i, 1) = dist[i] * cos(ang);
        
        ang += 2. * M_PI * angles[i] / total;
    }
    
    return ring;
}

void
SmoothExponentialMap::exponentialMap(const int src, const double r, Eigen::MatrixXd& em)
{
    ///////////////////////////////////////////////
    // map ring of vertex 'src' to tangent space
    ///////////////////////////////////////////////
    
    auto nbh = vertexVertex[src];
    const int nn = (int)nbh.size();
    auto ring = projectRingConformal(src);
    
    // build operator
    Eigen::MatrixXd A(2, nn + 1);
    A.setZero();
    
    Eigen::Matrix<double, 3, 2> tri;
    tri.row(0).setZero();
    
    for(int i = 0; i < nn; ++i)
    {
        tri.row(1) = ring.row(i);
        tri.row(2) = ring.row((i + 1) % nn);
        
        const double area = ((tri(1, 0) - tri(0,0)) * (tri(2, 1) - tri(0, 1)) - (tri(2, 0) - tri(0, 0)) * (tri(1, 1) - tri(0, 1)));
        
        A.col(0)              += area * (tri.row(2) - tri.row(1));
        A.col(1 + i)          += area * (tri.row(0) - tri.row(2));
        A.col(1 + (i+1) % nn) += area * (tri.row(1) - tri.row(0));
    }
    
    ///////////////////////////////////////////
    // compute geodesic distances
    ///////////////////////////////////////////
    
    auto nbhRange = collectNeighbors(src, r);
    
    // heat diffusion
    Eigen::SparseMatrix<double> B(cholA.cols(), nn + 1);
    B.resizeNonZeros(nn+1);
    
    B.outerIndexPtr()[0] = 0;
    B.innerIndexPtr()[0] = src;
    B.valuePtr()[0] = 1.;
    
    for(int i = 0; i < nn; ++i)
    {
        B.outerIndexPtr()[i+1] = i + 1;
        B.innerIndexPtr()[i+1] = nbh[i];
        B.valuePtr()[i+1] = 1.;
    }
    
    B.outerIndexPtr()[nn+1] = nn+1;
    const int lenNbh = (int)std::distance(indexStartPtr, indexEndPtr);
    
    Eigen::MatrixXd X;
    cholA.solveSparse(B, X, indexStartPtr, lenNbh);
    
    // sparsify: convert dense X to sparse Xsp for efficiency
    Eigen::SparseMatrix<double> Xsp;
    toSparse(X.col(0), Xsp, indexStartPtr, lenNbh);
    
    // compute gradients
    Eigen::MatrixXd grad = G * Xsp;
    
    // normalize gradients
    for(int i = 0; i < grad.size() / 3; ++i)
        Eigen::Map<Eigen::Vector3d>(grad.data() + 3 * i).normalize();
    
    // compute divergence
    Eigen::SparseMatrix<double> rhs = -(G.transpose() * (D * grad.sparseView()));
    
    // solve Poisson problem for final distances
    Eigen::Matrix<double, -1, 1> dist0, dist;
    dist0.setZero();
    cholL.solveSparse(rhs, dist0, indexStartPtr, lenNbh);
    
    // shift values in region and set rest to zero (rest is poluted during selective solve)
    dist.resizeLike(dist0);
    dist.setZero();
    
    for(int i : std::make_pair(indexStartPtr, indexEndPtr))
    {
        dist(i) = dist0(i) - dist0(src);
    }
    
    rescale(dist);
    
    ///////////////////////////////////////////
    // evaluate exponential map
    ///////////////////////////////////////////
    
    Eigen::VectorXd h(nn + 1);
    em.resize(V.rows(), 2);
    em.setConstant(-1.0);
    
    for(int i : nbhRange)
    {
        h[0] = X(i, 0);
        
        for(int j = 0; j < nn; ++j)
            h[j + 1] = X(i , j);
        
        Eigen::Vector2d polar = A * h;
        polar.normalize();
        const double a = atan2(polar[0], polar[1]);
        
        em(i, 0) = 0.5 - 0.5 * dist(i) * sin(a);
        em(i, 1) = 0.5 - 0.5 * dist(i) * cos(a);
    }
}

void
SmoothExponentialMap::geodesicsInHeat(const int src, const double r, Eigen::VectorXd& dist)
{
    // set neighbours
    auto rng = collectNeighbors(src, r);
    
    // heat diffusion
    Eigen::SparseMatrix<double> B(cholA.cols(), 1);
    B.insert(src, 0) = 1.;
    const int lenNbh = (int)std::distance(rng.first, rng.second);
    
    Eigen::VectorXd X;
    cholA.solveSparse(B, X, rng.first, lenNbh);
    
    Eigen::SparseMatrix<double> Xsp;
    toSparse(X, Xsp, rng.first, lenNbh);
    
    // compute gradients
    Eigen::MatrixXd grad = G * Xsp;
    
    // normalize gradients
    for(int i = 0; i < grad.size() / 3; ++i)
        Eigen::Map<Eigen::Vector3d>(grad.data() + 3 * i).normalize();
    
    // compute divergence
    Eigen::SparseMatrix<double> rhs = G.transpose() * (D * grad.sparseView());
    
    // solve Poisson problem for final distances
    cholL.solveSparse(rhs, dist, rng.first, lenNbh);
}

SmoothExponentialMap::SmoothExponentialMap(const Eigen::MatrixXd& V0, const Eigen::MatrixXi& F)
: indexBuffer(new int[V0.rows()]), V(V0)
{
    igl::adjacency_list(F, vertexVertex, true);
    std::vector<std::vector<int>> tmp;
    igl::vertex_triangle_adjacency((int)V.rows(), F, vertexTriangle, tmp);
    
    buildOperators(V, F);
    const double eps = stepsize(V, F);
    Eigen::SparseMatrix<double> A = L;
    
    for(int i = 0; i < A.cols(); ++i)
        for(int j = A.outerIndexPtr()[i]; j < A.outerIndexPtr()[i+1]; ++j)
            if(A.innerIndexPtr()[j] == i) A.valuePtr()[j] = M.diagonal()[i] + eps * A.valuePtr()[j];
            else A.valuePtr()[j] *= eps;
    
    cholL.analyzePattern(L);
    cholL.factorize(L);
    
    cholA.analyzePattern(A);
    cholA.factorize(A);
}
