#pragma once

#include <Eigen/Sparse>

/*************************************
 
 Extends Eigen's sparse Cholesky factorization to support selective solving.
 
************************************/

class SparseCholeskySolve : public Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>
{
    typedef typename Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Base;
    
    Eigen::VectorXi indices;
    Eigen::Matrix<int, -1, 1> flag;
    int mark = 1;
    
    // store all nodes that depend on rows indexed in 'ids' or present in 'B' in indices[ret],....,indices[indices.size()-1]
    // where 'ret' is returned by the functions
    
    int dependantNodes(const int* ids, const size_t idlen);
    
    int dependantNodes(const Eigen::SparseMatrix<double>& B);
    
    int dependantNodes(const Eigen::MatrixXd& X);
    
public:
    
    // calls analyzePattern of 'Base' and initializes 'indices' and 'flag'.
    void analyzePattern(const Eigen::SparseMatrix<double>& m);
    
    // solve A X = B, if rows != nullptr only rows rows[0],..., rows[rowsLen-1] will be guaranteed to be correct in X
    template<class BMatrixType, int M>
    void solveSparse(const BMatrixType& B, Eigen::Matrix<double, -1, M>& X, const int* rows = nullptr, const int rowsLen = 0);
    
    // if M is set to a positive number, specialized code (using intrinsics) for a specific size of ids can be generated
    template<int M = -1>
    void solveSparseUnit(const std::vector<int>& ids, Eigen::Matrix<double, -1, M>& X, const std::vector<int>* rows = nullptr);
};



template<class BMatrixType, int M>
void SparseCholeskySolve::solveSparse(const BMatrixType& B, Eigen::Matrix<double, -1, M>& X, const int* rows, const int rowsLen)
{
    const auto n = cols();
    const auto m = B.cols();
    
    assert(M == -1 || m == M);
    if(M != -1 && m != M) return;
    
    X = B;
    
    const Eigen::VectorXi& permInv = m_Pinv.indices();
    
    const int* col = matrixL().nestedExpression().outerIndexPtr();
    const int* row = matrixL().nestedExpression().innerIndexPtr();
    const double* val = matrixL().nestedExpression().valuePtr();
    
    Eigen::Matrix<double, M, 1> diag(M != -1 ? M : m);
    
    // Forward substitution
    int start = dependantNodes(B);
    
    for(int i = start; i < n; ++i)
    {
        const int k = start ? indices[i] : i;
        const int r = permInv[k];
        
        for(int j = 0; j < (M != -1 ? M : m); ++j)
        {
            diag(j) = X(r, j);
        }
        
        for(int l = col[k]; l < col[k+1]; ++l)
        {
            const int r = permInv[row[l]];
            
            for(int j = 0; j < (M != -1 ? M : m); ++j)
            {
                X(r, j) -= diag(j) * val[l];
            }
        }
    }
    
    // Multiply D^-1
    for(int i = 0; i < n; ++i)
    {
        const double d = vectorD()[i];
        const int r = permInv[i];
        
        for(int j = 0; j < (M != -1 ? M : m); ++j)
            X(r, j) /= d;
    }
    
    // Backward substitution
    start = rows ? dependantNodes(rows, rowsLen) : 0;
    
    for(int i = (int)n - 1; i >= start; --i)
    {
        const int k =  start ? indices[i] : i;
        const int r0 = permInv[k];
        
        for(int l = col[k]; l < col[k+1]; ++l)
        {
            const int r = permInv[row[l]];
            for(int j = 0; j < (M != -1 ? M : m); ++j)
            {
                X(r0, j) -= X(r, j) * val[l];
            }
        }
    }
}


template<int M>
void SparseCholeskySolve::solveSparseUnit(const std::vector<int>& ids, Eigen::Matrix<double, -1, M>& X, const std::vector<int>* rows)
{
    const auto n = cols();
    const auto m = ids.size();
    
    assert(M == -1 || m == M);
    if(M != -1 && m != M) return;
    
    X.resize(n, (M != -1 ? M : m));
    X.setZero();
    
    const Eigen::VectorXi& permInv = m_Pinv.indices();
    
    for(int i = 0; i < (M != -1 ? M : m); ++i)
    {
        X(ids[i], i) = 1.;
    }
    
    const int* col = matrixL().nestedExpression().outerIndexPtr();
    const int* row = matrixL().nestedExpression().innerIndexPtr();
    const double* val = matrixL().nestedExpression().valuePtr();
    
    Eigen::Matrix<double, M, 1> diag(M != -1 ? M : m);
    
    // Forward substitution
    int start = dependantNodes(ids.data(), ids.size());
    
    for(int i = start; i < n; ++i)
    {
        const int k = indices[i];
        const int r = permInv[k];
        
        for(int j = 0; j < (M != -1 ? M : m); ++j)
        {
            diag(j) = X(r, j);
        }
        
        for(int l = col[k]; l < col[k+1]; ++l)
        {
            const int r = permInv[row[l]];
            
            for(int j = 0; j < (M != -1 ? M : m); ++j)
            {
                X(r, j) -= diag(j) * val[l];
            }
        }
    }
    
    // Multiply D^-1
    for(int i = 0; i < n; ++i)
    {
        const double d = vectorD()[i];
        const int r = permInv[i];
        
        for(int j = 0; j < (M != -1 ? M : m); ++j)
            X(r, j) /= d;
    }
    
    // Backward substitution
    start = rows ? dependantNodes(rows->data(), rows->size()) : 0;
    
    for(int i = (int)n - 1; i >= start; --i)
    {
        const int k =  start ? indices[i] : i;
        const int r0 = permInv[k];
        
        for(int l = col[k]; l < col[k+1]; ++l)
        {
            const int r = permInv[row[l]];
            for(int j = 0; j < (M != -1 ? M : m); ++j)
            {
                X(r0, j) -= X(r, j) * val[l];
            }
        }
    }
}
