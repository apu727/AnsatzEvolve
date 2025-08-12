/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef FUSEDEVOLVE_H
#define FUSEDEVOLVE_H
#include "operatorpool.h"
#include <vector>
// #define MakeParallelEvolveCode
class baseAnsatz;

class FusedEvolve
{
    std::vector<size_t> m_commuteBoundaries;
    std::vector<stateRotate::exc> m_excs;
    std::vector<size_t> m_excPerm; // m_excPerm[i] = j means that the jth canonical ordered excitation is at position i
    std::vector<size_t> m_excInversePerm; // m_excInversePerm[j] = i means that the jth canonical ordered excitation is at position i
    bool m_excsCached;
    void regenCache();
    static constexpr long maxFuse =
#ifdef MakeParallelEvolveCode
        4;
#else
        12;
#endif
    template <uint16_t numberToFuse>
    using signMapX = std::array<bool,((1<<numberToFuse)/2)*numberToFuse> ; //true means +ve

    template <uint16_t numberToFuse>
    using localVectorMapX = std::array<uint32_t,1<<numberToFuse> ;

    template <uint16_t numberToFuse>
    using localVectorX  = std::vector<std::pair<localVectorMapX<numberToFuse>,signMapX<numberToFuse>>>;

    template <uint16_t numberToFuse>
    using fusedAnsatzX = std::vector<std::array<localVectorX<numberToFuse>,1<<numberToFuse>> ; //\Sum_{k=1}^{n} n choose k = 2^n for n >=0

    std::vector<void*> m_fusedAnsatzes; // void* to avoid all the template horribleness
    std::vector<uint16_t> m_fusedSizes;
    void cleanup();

    vector<numType> m_start;
    Eigen::SparseMatrix<realNumType, Eigen::ColMajor> m_HamEm;
    bool m_lieIsCompressed;
    std::shared_ptr<compressor> m_compressor;

    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_compressMatrix; //Stores equivalent parameters
    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_deCompressMatrix; //Stores equivalent parameters

public:
    FusedEvolve(const vector<numType> &start, const sparseMatrix<realNumType,numType>& Ham,
                Eigen::SparseMatrix<realNumType, Eigen::RowMajor>compressMatrix, Eigen::SparseMatrix<realNumType, Eigen::RowMajor>deCompressMatrix);
    ~FusedEvolve();
    void updateExc(const std::vector<stateRotate::exc>& excs);
    void evolve(vector<numType>& dest, const std::vector<realNumType>& angles, vector<numType>* specifiedStart  = nullptr);
    //finalVector is the result of an evolve operation. Since we dont cache the angles this is up to the user to provide correctly.
    void evolveDerivative(const vector<numType>& finalVector,vector<realNumType>& deriv,const std::vector<realNumType>& angles);
    void evolveHessian(Eigen::MatrixXd& Hessian,vector<realNumType>& deriv,const std::vector<realNumType>& angles);
    realNumType getEnergy(const vector<numType> &psi);
};

#endif // FUSEDEVOLVE_H
