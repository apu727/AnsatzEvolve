/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef HAMILTONIANMATRIX_H
#define HAMILTONIANMATRIX_H

#include "serialise.h"
#include "linalg.h"
#include <Eigen/Sparse>
struct excOp
{
    // uint8_t a; //create
    // uint8_t b; //create
    // uint8_t c; //destroy
    // uint8_t d; //destroy
    uint64_t create;
    uint64_t destroy;
    uint64_t signBitMask;
};

serialDataContainer MPICOMMAND_HamApplyToVector(char* ptr, size_t size);

template <typename dataType,typename vectorType>
class HamiltonianMatrix
{
public:

private:

    static constexpr size_t maxSizeForFullConstruction = 1000000000; // Doesnt fully construct if there are more qubits than this, Currently 1GB
    bool m_isFullyConstructed = false;
    Eigen::SparseMatrix<dataType, Eigen::ColMajor> m_fullyConstructedMatrix;

    bool m_isSecQuantConstructed = false;
    std::vector<excOp> m_operators;
    std::vector<dataType> m_vals;
    void constructFromSecQuant();
    void postProcessOperators();

    std::shared_ptr<compressor> m_compressor = nullptr;
    bool m_isCompressed = false;
    size_t m_linearSize;
    HamiltonianMatrix(const HamiltonianMatrix& other);
public:
    HamiltonianMatrix(const std::vector<dataType>& value, const std::vector<long>& iIndex, const std::vector<long>& jIndex,std::shared_ptr<compressor> comp);
    HamiltonianMatrix(const std::string &filePath,size_t numberOfQubits,std::shared_ptr<compressor> comp);
    //Expects Row Vectors and makes row vectors. We choose to accept map types to be general. Optionally a compression can be applied first. E.g. for derivatives.
    //Compressor is necessary since the Eigen types have lost this information
    void apply(const Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> &src,
               Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> &dest) const;

    void apply(const Eigen::Map<const Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> &src,
               Eigen::Map<Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> &dest) const;

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

template <typename dataType,typename vectorType>
class RDM;
//dest must be appropriately resized and zeroed
template<typename dataType, typename vectorType, bool compressed>
void opKernel(Eigen::Matrix<vectorType,-1,-1> &ret, const vectorType* src, std::vector<typename RDM<dataType, vectorType>::RDMOp> ops, std::shared_ptr<compressor> comp, size_t numQubits);

template <typename dataType,typename vectorType>
class RDM
{
public:
    struct RDMOp
    {
        excOp exc;
        std::pair<long, long> idxs; // stores where this goes. either 2rdm, 1rdm etc.
    };
private:
    //not adapted for Hermitian Sym!
    std::vector<RDMOp> m_twoRDMOps;  // a^\dagger_i a^\dagger_k a_j a_l, i > k && j < l
    std::vector<RDMOp> m_oneRDMOps;  // a^\dagger_i  a_j no restriction
    //m_numberCorr2RDMOps = n_i n_j = a^\dagger_i  a_i a^\dagger_j a_j = -a^\dagger_i a^\dagger_j a_i a_j + a^\dagger_i \delta_{ij} a_j (nosum)
    //                                                                 = a^\dagger_j a^\dagger_i a_i a_j + a^\dagger_i \delta_{ij} a_j (nosum)
    //Note that we require j>i or j<i in the two electron term. For it to be normal ordered (and therefore have the correct phase when applied to a basis state) the second swap must be performed.
    // This gives two cases: a^\dagger_2 a^\dagger_1 a_1 a_2 - Correct normal ordered
    //                       a^\dagger_1 a^\dagger_2 a_2 a_1 -- Wrong normal ordered but double error cancels.
    // Therefore there are no extra signs to take care of.
    std::vector<RDMOp> m_numberCorr2RDMOps;
    std::vector<RDMOp> m_numberCorr1RDMOps; // n_i = a^\dagger_i  a_i no restriction

    std::shared_ptr<compressor> m_comp;
    bool m_isCompressed;
    size_t m_numQubits;
public:
    RDM(size_t numberOfQubits, std::shared_ptr<compressor> comp);

    // ret_{(ji)(kl)} = <a^\dagger_i a^\dagger_j a_k a_l> at (j*m_numQubits+i,k*m_numQubits+l)
    // The (ji) ordering is such that when (kl) == (ji) there are no phases.
    Eigen::Matrix<vectorType,-1,-1> get2RDM(const vector<vectorType> &src);
    Eigen::Matrix<vectorType,-1,-1> get1RDM(const vector<vectorType> &src); // ret_{ik} = <a^\dagger_i a_k>
    Eigen::Matrix<vectorType,-1,-1> getNumberCorr2RDM(const vector<vectorType> &src); // ret_{ik} = n_i n_j = a^\dagger_i  a_i a^\dagger_j a_j
    Eigen::Matrix<vectorType, -1, 1> getNumberCorr1RDM(const vector<vectorType> &src); // ret_{ik} = n_i = a^\dagger_i  a_i
};

#ifdef useComplex
extern template class HamiltonianMatrix<realNumType,realNumType>;
extern template class HamiltonianMatrix<realNumType,numType>;
extern template class HamiltonianMatrix<numType,numType>;

extern template class RDM<realNumType,realNumType>;
extern template class RDM<realNumType,numType>;
extern template class RDM<numType,numType>;

#else
extern template class HamiltonianMatrix<numType,numType>;
extern template class RDM<numType,numType>;
#endif



#endif // HAMILTONIANMATRIX_H
