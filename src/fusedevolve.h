/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef FUSEDEVOLVE_H
#define FUSEDEVOLVE_H
#include "hamiltonianmatrix.h"
#include "operatorpool.h"
#include <vector>
// #define MakeParallelEvolveCode
constexpr bool fuseOnTheFly = true; // Makes the localVector on the fly.

class baseAnsatz;

class FusedEvolve
{
    std::vector<size_t> m_commuteBoundaries;
    std::vector<stateRotate::exc> m_excs; // the normal excs
    std::vector<stateRotate::exc> m_permExcs; // permuted excs.
    std::vector<size_t> m_excPerm; // m_excPerm[i] = j means that the jth canonical ordered excitation is at position i
    std::vector<size_t> m_excInversePerm; // m_excInversePerm[j] = i means that the jth canonical ordered excitation is at position i
    bool m_excsCached;

    static constexpr int8_t maxFuse =
#ifdef MakeParallelEvolveCode
        4;
#else
        12;
#endif
    template <uint16_t numberToFuse>
    using signMapX = std::array<bool,((1<<numberToFuse)/2)*numberToFuse> ; //true means +ve

    template <uint16_t numberToFuse>
    using localVectorMapX = std::array<uint64_t,1<<numberToFuse> ;

    template <uint16_t numberToFuse>
    using localVectorX  = std::vector<std::pair<localVectorMapX<numberToFuse>,signMapX<numberToFuse>>>;

    template <uint16_t numberToFuse>
    using fusedAnsatzX = std::vector<std::array<localVectorX<numberToFuse>,1<<numberToFuse>> ; //\Sum_{k=1}^{n} n choose k = 2^n for n >=0


    using localVectorDiagonal =  std::vector<uint64_t>;
    template<uint16_t numberToFuse>
    using fusedDiagonalAnsatzX = std::vector<std::array<localVectorDiagonal,1<<numberToFuse>>;



    std::vector<void*> m_fusedAnsatzes; // void* to avoid all the template horribleness
    std::vector<int8_t> m_fusedSizes;


    vector<numType> m_start; // this could be sparse and reduce the footprint.
    std::shared_ptr<HamiltonianMatrix<realNumType,numType>> m_Ham;
    bool m_lieIsCompressed;
    std::shared_ptr<compressor> m_compressor;

    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_compressMatrix; //Stores equivalent parameters
    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_compressMatrixPsi; //Stores equivalent parameters and also an extra row for psi to calculate hpsi easily
    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_deCompressMatrix; //Stores equivalent parameters

    void regenCache();
    void cleanup();

public:
    FusedEvolve(const vector<numType> &start, std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham,
                Eigen::SparseMatrix<realNumType, Eigen::RowMajor>compressMatrix, Eigen::SparseMatrix<realNumType, Eigen::RowMajor>deCompressMatrix);
    ~FusedEvolve();
    void updateExc(const std::vector<stateRotate::exc>& excs);
    void evolve(vector<numType>& dest, const std::vector<realNumType>& angles, vector<numType>* specifiedStart  = nullptr);
    void evolveMultiple(Matrix<numType>& dest, const Matrix<realNumType>::EigenMatrix& angles, vector<numType>* specifiedStart  = nullptr);
    //finalVector is the result of an evolve operation. Since we dont cache the angles this is up to the user to provide correctly.
    void evolveDerivative(const vector<numType>& finalVector,vector<realNumType>& deriv,const std::vector<realNumType>& angles, realNumType* Energy = nullptr, numType* projectedEnergy = nullptr);
    void evolveHessian(Eigen::MatrixXd& Hessian,vector<realNumType>& derivCompressed,const std::vector<realNumType>& angles, Eigen::Matrix<numType,-1,-1>* Ts = nullptr, realNumType* Energy = nullptr, numType* projectedEnergy = nullptr);
    realNumType getEnergy(const vector<numType> &psi);
    vector<realNumType> getEnergies(const Matrix<numType> &psi);

    void evolveDerivativeProj(const vector<numType>& finalVector,vector<realNumType>& deriv,const std::vector<realNumType>& angles, const vector<numType>& projVector, realNumType* Energy = nullptr);
    void evolveHessianProj(Eigen::MatrixXd& Hessian,vector<realNumType>& derivCompressed,const std::vector<realNumType>& angles, const vector<numType>& projVector, Eigen::Matrix<numType,-1,-1>* Ts = nullptr, realNumType* Energy = nullptr);
};

#endif // FUSEDEVOLVE_H
