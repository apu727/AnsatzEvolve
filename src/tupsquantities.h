/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef TUPSQUANTITIES_H
#define TUPSQUANTITIES_H

#include "globals.h"

#include "Eigen/Eigenvalues"
#include "ansatz.h"

#include <vector>
#include <string>


class TUPSQuantities
{
    void printOutputLine(std::vector<double>& toPrint, std::string name);
    void printOutputLine(std::vector<long double>& toPrint, std::string name);
    void printOutputHeaders(size_t numberOfPathsExHF);

    FILE* m_file = nullptr;
    sparseMatrix<realNumType,numType> m_Ham;
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_HamEm;
    realNumType m_NuclearEnergy = 0;

    int m_numberOfUniqueParameters = 0;
    std::string m_runPath;

    void calculateNumericalSecondDerivative(
        const std::vector<ansatz::rotationElement> &rp, realNumType startEnergy, const sparseMatrix<realNumType,numType> &Ham,
        const vector<numType>::EigenVector &direction, stateAnsatz* myAnsatz);
    void buildCompressionMatrices(int numberOfUniqueParameters, std::vector<std::pair<int,realNumType>> order,
                                  sparseMatrix<realNumType,numType>::EigenSparseMatrix &deCompressMatrix, sparseMatrix<realNumType,numType>::EigenSparseMatrix &compressMatrix);
    void asyncHij(const sparseMatrix<realNumType,numType> &Ham, const std::vector<std::vector<vector<numType>>>& secondDerivTensor,
                  const std::vector<vector<numType>> &derivTangentSpace, Matrix<realNumType>::EigenMatrix &Hij,
                  const vector<numType>& dest, const size_t iSize);

    void runNewtonMethod(stateAnsatz *myAnsatz,bool avoidNegativeHessianValues = true);
    void runNewtonMethodProjected(stateAnsatz *myAnsatz,bool avoidNegativeHessianValues = true);

    bool doStepsUntilHessianIsPositiveDefinite(stateAnsatz *myAnsatz, bool doDerivativeSteps);
    realNumType computeFrechetDistanceBetweenPaths(stateAnsatz *myAnsatz, const std::vector<ansatz::rotationElement>& rotationPath, const std::vector<ansatz::rotationElement>& rotationPath2);




public:
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_deCompressMatrix;
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_normCompressMatrix;
    sparseMatrix<realNumType,numType>::EigenSparseMatrix m_compressMatrix;
    TUPSQuantities(sparseMatrix<realNumType,numType>& Ham, std::vector<std::pair<int,realNumType>> order,
                   int numberOfUniqueParameters, realNumType NuclearEnergy, std::string runPath,  FILE* logfile = nullptr);

    void writeProperties(std::vector<std::vector<ansatz::rotationElement>>& rotationPaths, stateAnsatz* myAnsatz);
    void OptimiseTupsLBFGS(sparseMatrix<realNumType,numType> &Ham, std::vector<ansatz::rotationElement> &rotationPath,
                      stateAnsatz& myAnsatz, bool blanking = false);

    realNumType OptimiseTups(sparseMatrix<realNumType,numType> &Ham, const std::vector<ansatz::rotationElement> &rotationPath,
                           stateAnsatz& myAnsatz,bool avoidNegativeHessianValues = true);
    void iterativeTups(sparseMatrix<realNumType,numType> &Ham, const std::vector<ansatz::rotationElement> &rotationPath,
                      stateAnsatz& myAnsatz);
    void doSubspaceDiagonalisation(stateAnsatz &myAnsatz, size_t numberOfMinima,const std::vector<std::vector<ansatz::rotationElement>>& rotationPaths);


};

#endif // TUPSQUANTITIES_H
