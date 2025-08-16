#ifndef HAMILTONIANMATRIX_H
#define HAMILTONIANMATRIX_H

#include "linalg.h"
#include <Eigen/Sparse>

template <typename dataType,typename vectorType>
class HamiltonianMatrix
{
public:
    struct excOp
    {
        // uint8_t a; //create
        // uint8_t b; //create
        // uint8_t c; //destroy
        // uint8_t d; //destroy
        uint32_t create;
        uint32_t destroy;
        uint32_t signBitMask;
    };
private:

    static constexpr size_t maxSizeForFullConstruction = 1000000000; // Doesnt fully construct if there are more qubits than this, Currently 1GB
    bool m_isFullyConstructed = false;
    Eigen::SparseMatrix<dataType, Eigen::ColMajor> m_fullyConstructedMatrix;

    bool m_isSecQuantConstructed = false;
    std::vector<excOp> m_operators;
    std::vector<dataType> m_vals;
    void constructFromSecQuant();

    std::shared_ptr<compressor> m_compressor = nullptr;
    bool m_isCompressed = false;
    size_t m_linearSize;
    HamiltonianMatrix(const HamiltonianMatrix& other);
public:
    HamiltonianMatrix(const std::vector<dataType>& value, const std::vector<int>& iIndex, const std::vector<int>& jIndex);
    HamiltonianMatrix(const std::string &filePath,size_t numberOfQubits,std::shared_ptr<compressor> comp);
    //Expects Row Vectors and makes row vectors. We choose to accept map types to be general. Optionally a compression can be applied first. E.g. for derivatives.
    //Compressor is necessary since the Eigen types have lost this information
    void apply(const Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> &src,
               Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> &dest) const;

    void apply(const Eigen::Map<const Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> &src,
               Eigen::Map<Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> &dest,
               const Eigen::SparseMatrix<realNumType, Eigen::RowMajor>* compress = nullptr) const;

    //utility
    vector<vectorType> apply(const vector<vectorType> &src) const;
    vector<vectorType>& apply(const vector<vectorType> &src, vector<vectorType>& dest) const;
    bool ok()const {return m_isFullyConstructed || m_isSecQuantConstructed;}
    size_t rows()const {return m_linearSize;}
    size_t cols()const {return m_linearSize;}
    bool canGetSparse(){return m_isFullyConstructed;}
    Eigen::SparseMatrix<dataType, Eigen::ColMajor>& getSparse(){return m_fullyConstructedMatrix;}


};

#ifdef useComplex
extern template class HamiltonianMatrix<realNumType,realNumType>;
extern template class HamiltonianMatrix<realNumType,numType>;
extern template class HamiltonianMatrix<numType,numType>;
#else
extern template class HamiltonianMatrix<numType,numType>;
#endif

#endif // HAMILTONIANMATRIX_H
