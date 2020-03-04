#include "SparseCholeskySolve.hpp"

int SparseCholeskySolve::dependantNodes(const int* ids, const size_t idlen)
{
    const int n = (int)cols();
    int pos = n;
    ++mark;
    
    const Eigen::VectorXi& perm = m_P.indices();
    
    for(int k = 0; k < idlen; ++k)
    {
        const int i = ids[k];
        
        int len = 0;
        for(int j = perm[i]; j != -1 && flag[j] != mark; j = m_parent[j])
        {
            flag[j] = mark;
            indices[len++] = j;
        }
        
        while(len > 0)
        {
            indices[--pos] = indices[--len];
        }
    }
    
    return pos;
}

int SparseCholeskySolve::dependantNodes(const Eigen::SparseMatrix<double>& B)
{
    return dependantNodes(B.innerIndexPtr(), B.outerIndexPtr()[B.cols()]);
}

int SparseCholeskySolve::dependantNodes(const Eigen::MatrixXd& X)
{
    return 0;
}


void SparseCholeskySolve::analyzePattern(const Eigen::SparseMatrix<double>& m)
{
    Base::analyzePattern(m);
    indices.resize(m.cols());
    
    flag.resize(m.cols());
    flag.setZero();
}
