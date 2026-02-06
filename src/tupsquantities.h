/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef TUPSQUANTITIES_H
#define TUPSQUANTITIES_H

#include "fusedevolve.h"
#include "globals.h"

#include "Eigen/Eigenvalues"
#include "ansatz.h"
#include "hamiltonianmatrix.h"

#include <vector>
#include <string>

struct TUPSQuantitiesOptions
{
    bool computeLowestEigenValue = true;
    int numberOfOverlapsToCompute = 1;
    bool noHess = false;
};

class TUPSQuantities
{
    void printOutputLine(std::vector<double>& toPrint, std::string name);
    void printOutputLine(std::vector<std::complex<double>>& toPrint, std::string name);
    void printOutputLine(std::vector<long double>& toPrint, std::string name);
    void printOutputHeaders(size_t numberOfPathsExHF);

    FILE* m_file = nullptr;
    // sparseMatrix<realNumType,numType> m_Ham;
    std::shared_ptr<HamiltonianMatrix<realNumType,numType>> m_Ham = nullptr;
    // sparseMatrix<realNumType,numType>::EigenSparseMatrix m_HamEm;
    realNumType m_NuclearEnergy = 0;

    int m_numberOfUniqueParameters = 0;
    std::string m_runPath;

    void calculateNumericalSecondDerivative(
        const std::vector<ansatz::rotationElement> &rp, realNumType startEnergy, const sparseMatrix<realNumType,numType> &Ham,
        const vector<numType>::EigenVector &direction, stateAnsatz* myAnsatz);
    void calculateNumericalSecondDerivative(const std::vector<realNumType> &rp, realNumType startEnergy, std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham,
                                            const vector<numType>::EigenVector &direction, std::shared_ptr<FusedEvolve> myAnsatz);
    void buildCompressionMatrices(int numberOfUniqueParameters, std::vector<std::pair<int,realNumType>> order,
                                  sparseMatrix<realNumType,numType>::EigenSparseMatrix &deCompressMatrix, sparseMatrix<realNumType,numType>::EigenSparseMatrix &compressMatrix);
    void asyncHij(const sparseMatrix<realNumType,numType> &Ham, const std::vector<std::vector<vector<numType>>>& secondDerivTensor,
                  const std::vector<vector<numType>> &derivTangentSpace, Matrix<realNumType>::EigenMatrix &Hij,
                  const vector<numType>& dest, const size_t iSize);

    void runNewtonMethod(FusedEvolve *myAnsatz,std::vector<realNumType> &angles,bool avoidNegativeHessianValues = true);
    void runNewtonMethodProjected(FusedEvolve *myAnsatz,std::vector<realNumType> &angles, const vector<numType>& psiH, const vector<numType>& prevDest);

    bool doStepsUntilHessianIsPositiveDefinite(FusedEvolve *myAnsatz,std::vector<realNumType>& angles, bool doDerivativeSteps);
    realNumType computeFrechetDistanceBetweenPaths(std::shared_ptr<stateAnsatz> myAnsatz, std::shared_ptr<FusedEvolve> FE,
                                                   const std::vector<baseAnsatz::rotationElement> &rotationPath, const std::vector<baseAnsatz::rotationElement> &rotationPath2);




public:
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_deCompressMatrix;
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_normCompressMatrix;
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_compressMatrix;
    TUPSQuantities(std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham, std::vector<std::pair<int,realNumType>> order,
                   int numberOfUniqueParameters, realNumType NuclearEnergy, std::string runPath,  FILE* logfile = nullptr);

    void writeProperties(std::shared_ptr<stateAnsatz> myAnsatz,
                         std::shared_ptr<FusedEvolve> FE,
                         std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                         TUPSQuantitiesOptions opt);
    void OptimiseTupsLBFGS(sparseMatrix<realNumType,numType> &Ham, std::vector<ansatz::rotationElement> &rotationPath,
                      stateAnsatz& myAnsatz, bool blanking = false);

    realNumType OptimiseTups(stateAnsatz &myAnsatz, std::vector<ansatz::rotationElement> &rotationPath,
                             bool avoidNegativeHessianValues = true);
    realNumType OptimiseTups(FusedEvolve& FE, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues);
    realNumType iterativeTups(stateAnsatz &myAnsatz, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues);
    realNumType iterativeTups(FusedEvolve& FE, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues);
    void doSubspaceDiagonalisation(std::shared_ptr<stateAnsatz> myAnsatz, std::shared_ptr<FusedEvolve> FE,  size_t numberOfMinima,const std::vector<std::vector<ansatz::rotationElement>>& rotationPaths);



};

#endif // TUPSQUANTITIES_H
