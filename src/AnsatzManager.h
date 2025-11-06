/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef ANSATZMANAGER_H
#define ANSATZMANAGER_H
#include "ansatz.h"
#include "tupsquantities.h"
#include "fusedevolve.h"
#include <mutex>
#ifndef USE_COMPLEX
#define USE_FUSED
#endif

#ifdef USE_FUSED
constexpr bool useFused = true;
#else
constexpr bool useFused = false;
#endif
class stateAnsatzManager
{
private:
    std::shared_ptr<stateRotate> m_lie = nullptr;
    std::shared_ptr<stateAnsatz> m_ansatz = nullptr;
    std::shared_ptr<TUPSQuantities> m_TUPSQuantities = nullptr;
    std::shared_ptr<FusedEvolve> m_FA;

    sparseMatrix<numType,numType> m_target; //dummy, todo remove
    std::vector<realNumType> m_angles; // To compare and only recalculate when it changes
    std::vector<stateAnsatz::rotationElement> m_rotationPath; // keeps track of the operators too


    bool m_isConstructed = false;
    realNumType m_nuclearEnergy = 0; // Not necessary
    int m_numberOfUniqueParameters = -1;
    std::string m_runPath; // Dictates where to save things

    bool validateToRun(); // checks if everything is setup and we can do evolutions of things

    bool construct(); //Constructs if everything is setup

    void setRotationPathFromAngles();
    void setAnglesFromRotationPath();
    void evolve();

    //Things needed to construct
    std::vector<stateRotate::exc> m_excitations;
    vector<numType> m_start; // Size != 0
    std::shared_ptr<HamiltonianMatrix<realNumType,numType>> m_Ham; // Size != 0
    int m_numberOfParticles = -1;
    int m_spinUp = -1;
    int m_spinDown = -1;
    bool m_InitialSZSym = false;
    bool m_OperatorSZSym = false;
    bool m_particleSym = false;
    int m_numberOfQubits = -1;
    std::vector<std::pair<int,realNumType>> m_parameterDependency;


    bool setHamiltonian();
    void setOperatorSymmetry();
    std::vector<int> m_iIndexes;
    std::vector<int> m_jIndexes;
    std::vector<realNumType> m_coeffs;

    vector<numType> tempNumType; // used in various functions as a scratch space
    vector<realNumType> tempRealNumType; // used in various functions as a scratch space

    bool m_compressStateVectors = false;
    std::shared_ptr<compressor> m_compressor = nullptr;
    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_compressMatrix; //Stores equivalent parameters
    Eigen::SparseMatrix<realNumType, Eigen::RowMajor> m_deCompressMatrix; //Stores equivalent parameters
    vector<numType> m_current;

public:
    // Lock that is used by interface to external programs to guarantee thread safe operation. Generally only one of these functions should be called at a time. Lock is not used internally
    std::mutex m_interfaceLock;
    stateAnsatzManager();
    ~stateAnsatzManager();
    //Setup functions, These just store and wait until the last one is completed before setting anything up
    bool storeOperators(const std::vector<stateRotate::exc>& excs);
    bool storeInitial(int numberOfQubits, const std::vector<int>& indexes,const std::vector<numType>& coeffs);
    bool storeHamiltonian(std::vector<int>&& iIndexes, std::vector<int>&& jIndexes, std::vector<realNumType>&& Coeffs);
    bool storeNuclearEnergy(realNumType nuclearEnergy);
    //Expresses the mapping from free parameters to actual angles. parameterDependency[0].first gives the free parameter than angle 0 depends on.
    //The scale factor (parameterDependency[0].second) scales the free parameter to make the angle
    bool storeParameterDependencies(const std::vector<std::pair<int,realNumType>>& parameterDependency);
    bool storeRunPath(const std::string& runPath);

    //external Usage functions
    bool setAngles(std::vector<realNumType> angles);
    vector<realNumType>::EigenVector getAngles();

    bool getExpectationValue(realNumType& exptValue);
    bool getFinalState(vector<numType>& finalState);
    bool getGradient(vector<realNumType>& gradient); // decompressed format
    bool getGradientComp(vector<realNumType>& gradient); // compressed format
    bool getHessian(Matrix<realNumType>::EigenMatrix& hessian);
    bool getHessianComp(Matrix<realNumType>::EigenMatrix& hessian);
    //TODO get Hessian And Derivative in one

    //Allow providing the angle and update if needed
    bool getExpectationValue(const std::vector<realNumType>& angles, realNumType& exptValue);
    bool getFinalState(const std::vector<realNumType>& angles, vector<numType>& finalState);
    bool getGradient(const std::vector<realNumType>& angles, vector<realNumType>& gradient);
    bool getGradientComp(const std::vector<realNumType>& angles,vector<realNumType>& gradient); // compressed format
    bool getHessian(const std::vector<realNumType>& angles, Matrix<realNumType>::EigenMatrix& hessian);
    bool getHessianComp(const std::vector<realNumType>& angles, Matrix<realNumType>::EigenMatrix& hessian);

    //Compute on each set of angles
    bool getExpectationValues(Matrix<realNumType>::EigenMatrix& angles, vector<realNumType>::EigenVector& exptValue);

    //Calculation functions
    bool optimise();
    // bool subspaceDiag(); // Cant ATM keep track of multiple rotation paths TODO
    // bool writeProperties(); // Cant ATM keep track of multiple rotation paths TODO
    bool generatePathsForSubspace(size_t numberOfPaths);


};

#endif // ANSATZMANAGER_H
