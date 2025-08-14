/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "AnsatzManager.h"
#include "logger.h"
#include <filesystem>

bool stateAnsatzManager::validateToRun()
{
    bool success = true;
    if (!m_isConstructed)
    {
        if (!construct()) // could be done with a short circuit but this is cleaner imo
        {
            logger().log("Not yet Constructed, something is missing");
            success = false;
        }
    }
    if (!m_lie && !useFused)
    {
        success = false;
        logger().log("You tried running without setting up the operators!");
    }
    if (!m_ansatz && !useFused)
    {
        success = false;
        logger().log("Ansatz is not setup, You have either not provided the initial state or not provided the operators");
    }
    if (!m_TUPSQuantities)
    {
        success = false;
        logger().log("TUPS Properties is not setup, Make sure the Hamiltonian, Initial state, operators are all setup");
    }
    if (m_rotationPath.size() == 0 && !useFused)
    {
        success = false;
        logger().log("Rotation path has zero size. Operators have not been setup");
    }
    if (useFused && !m_FA)
    {
        success = false;
        logger().log("Fused Ansatz is not setup, You have either not provided the initial state or not provided the operators");
    }
    return success;
}

bool stateAnsatzManager::construct()
{ // Validate and construct
    if (m_isConstructed)
        return true;
    bool success = true;
    if (m_numberOfQubits == -1)
    {
        success = false;
        logger().log("Number of Qubits is not set. Make sure the initial state has been provided and has elements");
    }
    if (m_numberOfParticles == -1)
    {
        logger().log("Number of particles is not set. Make sure the initial state has been provided, has elements and these are all of the same particle number. This may result in slow calculations");
    }
    if (m_start.size() == 0)
    {
        success = false;
        logger().log("Make sure the initial state has been provided");
    }
    if (m_start.size() == 0)
    {
        success = false;
        logger().log("Initial state has no elements");
    }
    if (m_excitations.size() == 0)
    {
        success = false;
        logger().log("No excitations have been set");
    }
    if (m_parameterDependency.size() == 0 || m_numberOfUniqueParameters == -1)
    {
        success = false;
        logger().log("Parameter Dependencies, (order file) has zero size");
    }
    if (m_parameterDependency.size() != m_excitations.size())
    {
        success = false;
        logger().log("More operators than parameters dependencies given");
    }
    if (success)
    {
        if (m_particleSym)
        {
            m_compressStateVectors = true;
            if (m_SZSym)
                m_compressor = std::make_shared<SZAndnumberOperatorCompressor>(1<<m_numberOfQubits,m_spinUp,m_spinDown);
            else
                m_compressor = std::make_shared<numberOperatorCompressor>(m_numberOfParticles,1<<m_numberOfQubits);
            m_Ham.compress(m_compressor);
        }
        logger().log("SZSym:",m_SZSym);
        logger().log("particleNumSym:",m_particleSym);
        if constexpr(useFused)
        {
            if (m_compressStateVectors)
            {
                vector<numType> compStart;
                compressor::compressVector<numType>(m_start,compStart,m_compressor);
                static_cast<Matrix<numType>&>(m_start) = std::move(compStart);
            }
            m_TUPSQuantities = std::make_shared<TUPSQuantities>(m_Ham,m_parameterDependency,m_numberOfUniqueParameters, m_nuclearEnergy,m_runPath); //TODO allow for a output file path
            m_compressMatrix = m_TUPSQuantities->m_compressMatrix;
            m_deCompressMatrix = m_TUPSQuantities->m_deCompressMatrix;
            m_FA = std::make_shared<FusedEvolve>(m_start,m_Ham,m_compressMatrix,m_deCompressMatrix);
            m_FA->updateExc(m_excitations);
            m_rotationPath.clear(); // probably empty but lets make sure
            m_rotationPath.assign(m_angles.size(),{0,0});
            setRotationPathFromAngles();
            m_angles.clear();
            m_isConstructed = true;
        }
        else
        {

            m_lie = std::make_shared<stateRotate>(m_numberOfQubits,m_compressor);
            m_rotationPath.clear(); // probably empty but lets make sure
            m_angles.clear();
            for (auto& e : m_excitations)
            {
                m_lie->getLieAlgebraMatrix(e);
                m_rotationPath.push_back({m_lie->convertDataToIdx(&e),0});
                m_angles.push_back(0);
            }
            m_ansatz = std::make_shared<stateAnsatz>(&m_target,m_start,m_lie.get());
            m_ansatz->setCalculateFirstDerivatives(false);
            m_ansatz->setCalculateSecondDerivatives(false);
            m_ansatz->resetPath();
            for (auto& rp : m_rotationPath )
            {
                m_ansatz->addRotation(rp.first,rp.second);
            }

            m_TUPSQuantities = std::make_shared<TUPSQuantities>(m_Ham,m_parameterDependency,m_numberOfUniqueParameters, m_nuclearEnergy,m_runPath); //TODO allow for a output file path
            m_isConstructed = true;
        }
    }
    return success;
}

void stateAnsatzManager::setRotationPathFromAngles()
{
    for (size_t i = 0; i < m_rotationPath.size(); i++)
    {
        m_rotationPath[i].second = m_angles[i];
    }
}

void stateAnsatzManager::setAnglesFromRotationPath()
{
    for (size_t i = 0; i < m_angles.size(); i++)
    {
        m_angles[i] = m_rotationPath[i].second;
    }
}

stateAnsatzManager::stateAnsatzManager(): m_target({1},{1},{1},1)
{
    m_runPath = std::filesystem::current_path();
}

stateAnsatzManager::~stateAnsatzManager()
{
    // need to call this destructor first since m_ansatz has a reference to m_lie.get() TODO
    m_ansatz = nullptr;
    m_lie = nullptr;
}

bool stateAnsatzManager::storeOperators(const std::vector<stateRotate::exc> &excs)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (excs.size() == 0)
    {
        success = false;
        logger().log("Excitations given has zero size");
    }
    if (success)
    {
        m_excitations = excs;
    }
    return success;
}

bool stateAnsatzManager::storeInitial(int numberOfQubits, const std::vector<int>& indexes,const std::vector<numType>& coeffs)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (numberOfQubits < 1)
    {
        success = false;
        logger().log("Number of qubits < 1", numberOfQubits);
    }
    if (indexes.size() == 0 || coeffs.size() == 0)
    {
        success = false;
        logger().log("Initial state has no elements");
    }
    if (coeffs.size() != indexes.size())
    {
        success = false;
        logger().log("Initial state indexes and coeffs are a different size");
    }
    if (success)
    {
        m_start.resize(1<<numberOfQubits,false,nullptr);
        for (size_t i = 0; i < indexes.size(); i++)
        {
            m_start[indexes[i]] = coeffs[i];
        }
        realNumType mag = std::sqrt(m_start.dot(m_start));
        if (abs(mag-1) > 1e-14)
        {
            logger().log("Warning: Initial vector is not normalised, the error is", abs(mag-1));
        }
        m_numberOfQubits = numberOfQubits;

        //Determine number of particles
        uint32_t ones = -1;
        int numberOfParticles = -1;
        bool allSameParticleNumber = true;
        bool SZSym = true;
        int spinUp = -1;
        int spinDown = -1;

        if (numberOfQubits %2 != 0)
            SZSym = false;

        for (size_t i = 0; i < indexes.size(); i++)
        {
            if (coeffs[i] == 0.)
            {
                continue;
            }
            char currNumberOfParticles = bitwiseDot(indexes[i],ones,32);

            if (numberOfParticles == -1 && currNumberOfParticles != -1)
            {
                numberOfParticles = currNumberOfParticles;
            }
            if (currNumberOfParticles != numberOfParticles)
            {
                allSameParticleNumber = false;
                break;
            }
            int currSpinUp = bitwiseDot(indexes[i]>>(numberOfQubits/2),ones,32);
            int currSpinDown = bitwiseDot(indexes[i]>>(numberOfQubits/2),ones,numberOfQubits/2);
            if (spinUp == -1 && currSpinUp != -1)
                spinUp = currSpinUp;
            if (spinDown == -1 && currSpinDown != -1)
                spinDown = currSpinDown;
            if (currSpinUp != spinUp)
                SZSym = false;
            if (currSpinDown != spinDown)
                SZSym = false;
        }
        if (numberOfParticles == -1)
        {
            logger().log("Warning: Could not determine particle number");
        }
        else if (!allSameParticleNumber)
        {
            logger().log("Warning: not all the same particle number");
        }
        m_numberOfParticles = -1;
        m_spinUp = -1;
        m_spinDown = -1;
        m_SZSym = false;
        m_particleSym = false;

        if (allSameParticleNumber && numberOfParticles != -1)
        {
            m_numberOfParticles = numberOfParticles;
            m_particleSym = true;
            if (SZSym && spinUp != -1 && spinDown != -1)
            {
                m_SZSym = true;
                m_spinUp = spinUp;
                m_spinDown = spinDown;
            }
        }
    }
    return success;
}

bool stateAnsatzManager::storeHamiltonian(const std::vector<int> &iIndexes, const std::vector<int> &jIndexes, const std::vector<realNumType> &Coeffs)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (iIndexes.size() == 0 || jIndexes.size() == 0 || Coeffs.size() == 0)
    {
        success = false;
        logger().log("Hamiltonian Matrix has no elements");
    }
    if (iIndexes.size() != jIndexes.size() || iIndexes.size() != Coeffs.size() || jIndexes.size() != Coeffs.size())
    {
        success = false;
        logger().log("Hamiltonian Matrix indexes and coeffs are a different size");
    }
    if (success)
        m_Ham = sparseMatrix<realNumType,numType>(Coeffs,iIndexes,jIndexes,bool());
    return success;
}

bool stateAnsatzManager::storeNuclearEnergy(realNumType nuclearEnergy)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (success)
    {
        m_nuclearEnergy = nuclearEnergy;
    }
    return success;
}

bool stateAnsatzManager::storeParameterDependencies(const std::vector<std::pair<int, realNumType> > &parameterDependency)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (parameterDependency.size() == 0)
    {
        success = false;
        logger().log("Parameter Dependencies, (order file) has zero size");
    }
    if (success)
    {
        m_parameterDependency = parameterDependency;
        m_numberOfUniqueParameters = -1;
        for (auto& pD : m_parameterDependency)
        {
            m_numberOfUniqueParameters = std::max(pD.first+1,m_numberOfUniqueParameters); // +1 to not get the largest index but the size
        }

        if ((size_t)m_numberOfUniqueParameters > parameterDependency.size())
        {
            m_numberOfUniqueParameters = -1;
            m_parameterDependency.clear();
            success = false;
        }
    }
    return success;
}

bool stateAnsatzManager::storeRunPath(const std::string &runPath)
{
    bool success = true;
    if (m_isConstructed)
    {
        success = false;
        logger().log("Already constructed, setup cant be modified now");
    }
    if (runPath.size() == 0)
    {
        success = false;
        logger().log("Run path has zero size");
    }
    if (success)
    {
        m_runPath = runPath;
    }
    return success;
}

bool stateAnsatzManager::setAngles(std::vector<realNumType> angles)
{// These are likely compressed angles but not necessarily
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }

    if (angles.size() == m_rotationPath.size())
        ; // decompressed angles are good;
    else if ((int)(angles.size()) == m_numberOfUniqueParameters)
    {// need to decompress the angles
        vector<realNumType>::EigenVector anglesEm(angles.size());
        for (size_t i = 0; i < angles.size(); i++)
        {
            anglesEm(i) = angles[i];
        }
        anglesEm = m_TUPSQuantities->m_deCompressMatrix * anglesEm;
        angles.resize(anglesEm.size());

        for (size_t i = 0; i < angles.size(); i++)
        {
            angles[i] = anglesEm(i);
        }
    }
    else
    {
        success = false;
        logger().log("Incorrect number of angles given", angles.size());
    }

    if (success && angles != m_angles)
    {
        m_angles = angles;
        if constexpr(useFused)
        {
            m_FA->evolve(m_current,angles);
        }
        else
        {
            setRotationPathFromAngles();
            m_ansatz->updateAngles(m_angles);
        }
    }
    return success;
}

bool stateAnsatzManager::getExpectationValue(realNumType &exptValue)
{
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    if constexpr(useFused)
    {
        exptValue = m_FA->getEnergy(m_current);
    }
    else
    {
        exptValue = m_Ham.braket(m_ansatz->getVec(),m_ansatz->getVec(),&tempNumType);
    }
    return success;
}

bool stateAnsatzManager::getFinalState(vector<numType> &finalState)
{
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    const vector<numType>& dest = useFused ?  m_current: m_ansatz->getVec();
    std::shared_ptr<compressor> comp;
    if (dest.getIsCompressed(comp))
        compressor::deCompressVector<numType>(dest,finalState,comp);
    else
        finalState.copy(dest);
    return success;
}

bool stateAnsatzManager::getGradient(vector<realNumType> &gradient)
{// Only call if you dont want the Hessian, The combined function is faster... Not by too much though
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    if (useFused)
        m_FA->evolveDerivative(m_current,gradient,m_angles);
    else
        m_ansatz->getDerivativeVec(&m_Ham,gradient);

    return success;
}

bool stateAnsatzManager::getGradientComp(vector<realNumType> &gradient)
{
    bool success = true;
    success = getGradient(gradient);
    if (success)
    {
        vector<realNumType>::EigenVector gradEm = gradient;
        gradEm = m_TUPSQuantities->m_compressMatrix * gradEm;
        //Damn move semantics
        static_cast<Matrix<realNumType>&>(gradient) = vector<realNumType>(gradEm);
    }
    return success;
}

bool stateAnsatzManager::getHessian(Matrix<realNumType>::EigenMatrix &hessian)
{
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    if constexpr(useFused)
    {
        logger().log("Fused ansatz can only provide compressed Hessians");
        __builtin_trap();
    }
    else
        m_ansatz->getHessianAndDerivative(&m_Ham,hessian,tempRealNumType);
    return success;
}

bool stateAnsatzManager::getHessianComp(Matrix<realNumType>::EigenMatrix &hessian)
{
    bool success = true;
    if constexpr (useFused)
    {
        if (!validateToRun())
        {
            success = false;
            return success;
        }
        m_FA->evolveHessian(hessian,tempRealNumType,m_angles);
    }
    else
    {
        success = getHessian(hessian);
        if (success)
        {
            hessian = m_TUPSQuantities->m_compressMatrix * hessian * m_TUPSQuantities->m_compressMatrix.transpose();
        }
    }
    return success;
}

bool stateAnsatzManager::getExpectationValue(const std::vector<realNumType> &angles, realNumType &exptValue)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getExpectationValue(exptValue);
    return success;
}

bool stateAnsatzManager::getFinalState(const std::vector<realNumType> &angles, vector<numType> &finalState)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getFinalState(finalState);
    return success;
}

bool stateAnsatzManager::getGradient(const std::vector<realNumType> &angles, vector<realNumType> &gradient)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getGradient(gradient);
    return success;
}

bool stateAnsatzManager::getGradientComp(const std::vector<realNumType> &angles, vector<realNumType> &gradient)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getGradientComp(gradient);
    return success;
}

bool stateAnsatzManager::getHessian(const std::vector<realNumType> &angles, Matrix<realNumType>::EigenMatrix &hessian)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getHessian(hessian);
    return success;
}

bool stateAnsatzManager::getHessianComp(const std::vector<realNumType> &angles, Matrix<realNumType>::EigenMatrix &hessian)
{
    bool success = true;
    success = setAngles(angles);
    if (!success)
        return success;
    success = getHessianComp(hessian);
    return success;
}

bool stateAnsatzManager::optimise()
{
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    if (useFused)
        m_TUPSQuantities->OptimiseTups(*m_FA,m_rotationPath,true);
    else
        m_TUPSQuantities->OptimiseTups(*m_ansatz,m_rotationPath,true);
    if (!useFused)
        m_rotationPath = m_ansatz->getRotationPath();
    setAnglesFromRotationPath();
    return success;
}

bool stateAnsatzManager::generatePathsForSubspace(size_t numberOfPaths)
{
    bool success = true;
    if (!validateToRun())
    {
        success = false;
        return success;
    }
    if (useFused)
    {
        logger().log("Can only do subspacediagonalisation with standard ansatz build");
        //TODO
        return false;
    }
    std::srand(100);
    std::vector<realNumType> Energies;
    std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
    size_t pathsFound = 0;
    rotationPaths.push_back(m_rotationPath);

    while (pathsFound < numberOfPaths)
    {
        vector<realNumType>::EigenVector angles(m_numberOfUniqueParameters);
        for (int i = 0; i < m_numberOfUniqueParameters; i++)
        {
            angles(i) = (2*M_PI*std::rand())/(RAND_MAX);
        }
        angles = m_TUPSQuantities->m_deCompressMatrix * angles;
        for (size_t i = 0; i < rotationPaths.back().size(); i++)
        {
            rotationPaths.back()[i].second = angles(i);
        }

        realNumType Energy;
        if (useFused)
            Energy = m_TUPSQuantities->OptimiseTups(*m_FA,rotationPaths.back(),true); // this is broken
        else
            Energy = m_TUPSQuantities->OptimiseTups(*m_ansatz,rotationPaths.back(),true);
        if (std::find_if(Energies.begin(),Energies.end(), [=](realNumType E){return std::abs(E-Energy) < 1e-10;}) == Energies.end())
        {
            Energies.push_back(Energy);
            if (!useFused)
                rotationPaths.back() = m_ansatz->getRotationPath();
            if (pathsFound < numberOfPaths-1)
            {
                rotationPaths.push_back(rotationPaths[0]);
            }
            pathsFound++;
        }
    }
    logger().log("Found following Energies",Energies);

    m_TUPSQuantities->doSubspaceDiagonalisation(m_ansatz,m_FA,numberOfPaths,rotationPaths);
    return success;
}


