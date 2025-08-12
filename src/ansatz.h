/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef ANSATZ_H
#define ANSATZ_H

#include <vector>
//#include <cblas.h>
#include "globals.h"
#include "linalg.h"

#include "operatorpool.h"

class baseAnsatz
{
public:
    typedef std::pair<size_t,realNumType> rotationElement;
protected:
    operatorPool* m_lie;
    bool m_lieIsCompressed = false;
    std::shared_ptr<compressor> m_compressor;
    //TODO this is a hack and will cause caching problems
    Eigen::SparseMatrix<realNumType, Eigen::ColMajor> m_HamEm;
    bool m_HamEmIsSet = false;

    targetMatrix<numType,numType>* m_target;
    vector<numType> m_start;
    vector<numType> m_current;
    std::vector<rotationElement> m_rotationPath;
    //m_lieAlgebraElems = m_lie->getLieAlgebraMatrices();

    std::vector<vector<numType>> m_derivList;
    std::vector<vector<numType>> m_derivParallelList;
    std::vector<vector<numType>> m_derivConstList;
    std::vector<vector<numType>> m_derivSpaceNotEvolvedCache; // same as m_derivList but not evolved further
    std::vector<vector<numType>> m_hPsiEvolvedList;
    std::vector<vector<numType>> m_hPsiDeriv;

    //first index is index of second derivative, second index is index of first derivative, third index is vector.
    //e.g. m_secondDerivList[1][0][2] is the third element of \frac{d}{d\theta_1}\frac{d}{d\theta_0} \ket{start}
    //note first index is always >= second index. i.e. m_secondDerivList[0][1] does not exist. It is inferred from symmetry
    std::vector<std::vector<vector<numType>>> m_secondDerivList;

    std::vector<vector<numType>> m_lieTangentSpaceCache;
    std::vector<vector<numType>> m_lieParallelSpaceCache;
    std::vector<vector<numType>> m_lieConstSpaceCache;
    std::vector<realNumType> m_derivCoeffListCache; // list of D/dtheta_i


    bool m_lieTangentSpaceCacheValid = false;
    bool m_lieParallelSpaceCacheValid = false;
    bool m_lieConstSpaceCacheValid = false;
    //Calculates \frac{d^2}{d\theta_\mu d\theta_nu} \ket{\psi}
    bool calculateSecondDerivatives = false;
    bool m_calculateFirstDerivatives = true;

    //bool m_derivQFMListCacheValid = false;
    bool m_derivCoeffCacheValid = false;

    void rotateState(const matrixType &rotationGenerator, realNumType theta, size_t indexInPath);
    void resetState();

public:
    baseAnsatz(targetMatrix<numType,numType>* target,const vector<numType>& start, operatorPool* lie);
    baseAnsatz(const baseAnsatz& other) = delete;
    baseAnsatz& operator= (const baseAnsatz& other) = delete;

    const vector<numType>& getVec() const;
    void updateAngles(const std::vector<realNumType>& angles);
    void updateAnglesNoDeriv(const std::vector<realNumType>& angles, vector<numType>& dest);
    void calcRotationAlongPath(const std::vector<rotationElement>& rotationPath,vector<numType>& dest, const vector<numType>& start);

    //Note that ExpMat is cached. Call resetPath to clear the cached version.
    void getDerivativeVec(sparseMatrix<realNumType,numType> *ExpMat, vector<realNumType>& dest);
    void getHessianAndDerivative(sparseMatrix<realNumType,numType> *ExpMat, Matrix<realNumType>::EigenMatrix& dest, vector<realNumType>& deriv, Eigen::SparseMatrix<realNumType,Eigen::RowMajor>* compressMatrix = nullptr);

    void getDerivativeVecProj(const vector<numType>& projVec, vector<realNumType>& dest);
    void getHessianAndDerivativeProj(const vector<numType>& projVec, Matrix<realNumType>::EigenMatrix& dest, vector<realNumType>& deriv);
    //void removeAngles(const std::vector<int>& indexes);
    void resetPath();
    void addRotation(int rotationGeneratorIdx, realNumType angle);
    void setCalculateFirstDerivatives(bool val);
    void setCalculateSecondDerivatives(bool val);
    bool getCalculateSecondDerivatives();
    const vector<numType>& getStart();

    //void addAdjointElem(int adjointIdx,angle);
    const std::vector<rotationElement>& getRotationPath() const;



    const std::vector<vector<numType>> & getLieTangentSpace();
    const std::vector<vector<numType>> & getLieParallelSpace();
    const std::vector<vector<numType>> & getLieConstSpace();

    const std::vector<vector<numType>>& getDerivTangentSpace();
    const std::vector<vector<numType>>& getDerivParallelSpace();
    const std::vector<vector<numType>>& getDerivConstSpace();
    const std::vector<std::vector<vector<numType>>>& getSecondDerivTensor();
    const std::vector<realNumType>& getGradVector(bool spaceBased, const vector<numType>& targetVectorPauliBasis); // compute -2t_p.(p_p+c_p). I'll probably end up regretting this. We love inheritance
//    operatorPool* getLie(){return m_lie;}
    //const matrixType& getMat(size_t idx)const{return m_lieAlgebraElems->at(idx);}
    //const std::unordered_map<size_t,matrixType>* getLieMats() const {return m_lieAlgebraElems;}
    targetMatrix<numType,numType>* getTarget(){return m_target;}


};

class stateAnsatz : public baseAnsatz
//NOT Thread Safe. Cannot call multiple functions at once
{
public:

    stateAnsatz(targetMatrix<numType,numType>* target,const vector<numType>& start, stateRotate* lie);
    stateAnsatz(const stateAnsatz& other) = delete;
    stateAnsatz& operator= (const stateAnsatz& other) = delete;

    stateRotate* getLie(){return (stateRotate*)m_lie;}

};

typedef baseAnsatz ansatz;











#endif // ANSATZ_H
