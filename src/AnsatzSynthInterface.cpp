/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "Generated/AnsatzSynthInterface.h"
#include "AnsatzManager.h"
#include "logger.h"
#include <stdio.h>

bool traceInterfaceCalls = false;

//----------------------------------------------------------
// Enables logging/debugging of interface function calls.
// If val =/= 0, internal code logs arguments and usage.
// Can generate large printouts — use with caution.
//----------------------------------------------------------
void setTraceInterfaceCalls(int val)
{
    traceInterfaceCalls = val != 0;
}

//----------------------------------------------------------
// Initializes a new Ansatz object.
// Returns a C pointer to the created object.
// the Returned pointer must be given to all functions taking a ctx argument. This is almost all functions
//----------------------------------------------------------
void *init()
{
    stateAnsatzManager* ptr = new stateAnsatzManager;
    if (traceInterfaceCalls)
        logger().log("Constructed object",ptr);
    return ptr;
}

//----------------------------------------------------------
// Deletes the stateAnsatzManager object and frees memory.
// If ctx is invalid or already cleaned up, returns error.
// ctx is set to nullptr after use. Passing ctx = nullptr is safe
//----------------------------------------------------------
int cleanup(void *ctx)
{
    if (traceInterfaceCalls)
        logger().log("cleanup called with void**",ctx);
    if (ctx == nullptr)
    {
        logger().log("(void**) ctx is nullptr, something has gone wrong. Potential Leak");
        return 1;
    }

    void** myCtx = (void**)ctx;
    if (traceInterfaceCalls)
        logger().log("cleanup called with void*",*myCtx);

    if (*myCtx == nullptr)
        return 0;
    delete static_cast<stateAnsatzManager*>(*myCtx);
    *myCtx = nullptr;
    return 0;
}

//----------------------------------------------------------
// Sets excitation operators and parameter ordering.
// Inputs:
// - nparams: number of excitations
// - operators: 4*nparams integers defining excitation operators:
//   E.g. [3, 7, 4, 8] means excite from 8th and 4th qubit into 7th and 3rd respectively
// - orderfile: array defining parameter ordering. E.g. 1,1,2,3 means there are 4 operators and the first two have the same parameter
//----------------------------------------------------------
int setExcitation(int nparams, const int *operators, const int *orderfile, void *ctx)
{
    return setExcitationScale(nparams,operators,orderfile,nullptr,ctx);
}

int setExcitationScale(int nparams, const int *operators, const int *orderfile, const double *scale, void *ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (nparams < 1)
    {
        logger().log("nparams < 1");
        return 2;
    }
    if (traceInterfaceCalls)
    {
        logger().log("setExcitationScale called with:");
        logger().log("nParams", nparams);
        logger().log("operators", std::vector<int>(operators,operators+nparams*4));
        logger().log("orderfile", std::vector<int>(orderfile,orderfile+nparams));
        if (scale)
            logger().log("scale", std::vector<int>(scale,scale+nparams));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);

    std::vector<stateRotate::exc> excs(nparams);
    for (int i = 0; i < nparams; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (operators[nparams*j + i]-1 > std::numeric_limits<int8_t>::max())
            {
                //This is likely if we are reading out of bounds in the array
                logger().log("Excitation out of bounds, max is 128",operators[nparams*j + i]);
                return 3;
            }
            excs[i][j] = operators[nparams*j + i]-1; // Fortran order
        }
    }
    bool succ = thisPtr->storeOperators(excs);
    if (!succ)
        return 4;
    std::vector<std::pair<int, realNumType> > parameterDependency;
    for (int i = 0; i < nparams; i++)
    {
        if (orderfile[i] > nparams)
        {
            logger().log("orderfile out of bounds",orderfile[i]);
            return 5;
        }
        if (scale)
            parameterDependency.push_back({orderfile[i]-1,scale[i]});
        else
            parameterDependency.push_back({orderfile[i]-1,1});
    }
    succ = thisPtr->storeParameterDependencies(parameterDependency);
    if (!succ)
        return 6;
    return 0;
}

//----------------------------------------------------------
// Sets the Hamiltonian matrix (sparse form) in the backend.
// Inputs:
// - N: number of non-zero entries
// - iIndexes, jIndexes: row/column indices (1-based Fortran)
// - coeffs: matrix element values
//----------------------------------------------------------
int setHamiltonian(int N, const long* iIndexes, const long* jIndexes, const double* coeffs, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (traceInterfaceCalls)
    {
        logger().log("setHamiltonian called with:");
        logger().log("N", N);
        logger().log("iIndexes", std::vector<long>(iIndexes,iIndexes+N));
        logger().log("jIndexes", std::vector<long>(jIndexes,jIndexes+N));
        logger().log("coeffs", std::vector<long>(coeffs,coeffs+N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    if (N < 1)
    {
        logger().log("N < 1");
        return 2;
    }
    std::vector<long> iIndexesV(iIndexes,iIndexes+N);
    std::vector<long> jIndexesV(jIndexes,jIndexes+N);
    for (long& iIndex : iIndexesV)
        iIndex--;
    for (long& jIndex : jIndexesV)
        jIndex--;
    std::vector<realNumType> CoeffsV(coeffs,coeffs+N);
    bool success = thisPtr->storeHamiltonian(std::move(iIndexesV),std::move(jIndexesV),std::move(CoeffsV));
    if (!success)
        return 3;
    return 0;
}

int setHamiltonianFile(char* filepath, int filepathLength, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (traceInterfaceCalls)
    {
        logger().log("setHamiltonianFile called with:");
        logger().log("filepathLength", filepathLength);
        logger().log("filepath", std::string(filepath,filepathLength));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);

    std::string filepathStr(filepath,filepathLength);

    bool success = thisPtr->storeRunPath(filepathStr);
    if (!success)
        return 2;
    return 0;
}


//----------------------------------------------------------
// Sets the initial quantum state vector (sparse form).
// Inputs:
// - numQubits: total number of qubits
// - N: number of non-zero components
// - iIndexes: basis indices. the |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
// - coeffs: corresponding amplitudes
//----------------------------------------------------------
int setInitialState(int numQubits, long N, const long* iIndexes, const double* coeffs, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (traceInterfaceCalls)
    {
        logger().log("setInitialState called with:");
        logger().log("numQubits", numQubits);
        logger().log("N", N);
        logger().log("iIndexes", std::vector<long>(iIndexes,iIndexes+N));
        logger().log("coeffs", std::vector<double>(coeffs,coeffs+N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    if (N < 1)
    {
        logger().log("N < 1");
        return 2;
    }
    if (numQubits < 1)
    {
        logger().log("numQubits < 1");
        return 2;
    }
    std::vector<long> iIndexesV(iIndexes,iIndexes+N);
    for (long& iIndex : iIndexesV)
        iIndex--;
    std::vector<numType> CoeffsV(coeffs,coeffs+N);
    bool success = thisPtr->storeInitial(numQubits,iIndexesV,CoeffsV);
    if (!success)
        return 3;
    return 0;
}

//----------------------------------------------------------
// Sets the initial quantum state vector (sparse form).
// Inputs:
// - numQubits: total number of qubits
// - N: number of non-zero components
// - iIndexes: basis indices. the |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
// - coeffs: corresponding complex amplitudes
//----------------------------------------------------------
#ifdef useComplex
int setInitialStateComplex (int numQubits, long N, const long *iIndexes, const __GFORTRAN_DOUBLE_COMPLEX *coeffs, void *ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (traceInterfaceCalls)
    {
        logger().log("setInitialState called with:");
        logger().log("numQubits", numQubits);
        logger().log("N", N);
        logger().log("iIndexes", std::vector<int>(iIndexes,iIndexes+N));
        logger().log("coeffs", std::vector<std::complex<double>>(coeffs,coeffs+N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    if (N < 1)
    {
        logger().log("N < 1");
        return 2;
    }
    if (numQubits < 1)
    {
        logger().log("numQubits < 1");
        return 2;
    }
    std::vector<int> iIndexesV(iIndexes,iIndexes+N);
    for (int& iIndex : iIndexesV)
        iIndex--;
    std::vector<std::complex<double>> CoeffsV(coeffs,coeffs+N);
    bool success = thisPtr->storeInitial(numQubits,iIndexesV,CoeffsV);
    if (!success)
        return 3;
    return 0;
}
#else
int setInitialStateComplex (int, long, const long *, const __GFORTRAN_DOUBLE_COMPLEX *, void *)
{
    logger().log("Cannot set a complex initial state without building in complex mode");
    return 4;
}
#endif

//----------------------------------------------------------
// Computes the energy <psi|H|psi> for a given angle parameterization.
// Inputs:
// - NAngles: number of parameters
// - angles: array of real values
// Output:
// - energy: result of the expectation value
//----------------------------------------------------------
int getEnergy(int NAngles, const double* angles, double* energy, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (angles == nullptr)
    {
        logger().log("nullptr angles passed");
        return 2;
    }
    if (NAngles < 1)
    {
        logger().log("NAngles < 1");
        return 3;
    }

    if (traceInterfaceCalls)
    {
        logger().log("get_UCC_EXPT called with:");
        logger().log("NAngles", NAngles);
        logger().log("(double*)angles", (void*)angles);
        logger().log("(double*)exptvalue", energy);
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    bool success = thisPtr->getExpectationValue(std::vector<double>(angles,angles+NAngles),*energy);
    if (!success)
        return 3;
    return 0;
}
//----------------------------------------------------------
// Returns the full final quantum state vector after applying
// parameterised ansatz gates.
// Inputs:
// - NAngles: number of parameters
// - angles: gate parameters
// - NBasisVectors: expected output size
// Output:
// - finalState: output state vector in as a NBasisVectors = 2^numQubits array of double. Can be indexed by: |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
//----------------------------------------------------------
int getFinalState (int NAngles, const double* angles, int NBasisVectors, double* finalState, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (angles == nullptr)
    {
        logger().log("nullptr angles passed");
        return 2;
    }
    if (finalState == nullptr)
    {
        logger().log("nullptr finalState passed");
        return 3;
    }
    if (NBasisVectors < 1)
    {
        logger().log("NBasisVectors < 1");
        return 4;
    }
    if (NAngles < 1)
    {
        logger().log("NAngles < 1");
        return 5;
    }
    if (traceInterfaceCalls)
    {
        logger().log("get_UCC_finalState called with:");
        logger().log("NAngles", NAngles);
        logger().logAccurate("angles", std::vector<double>(angles,angles+NAngles));
        logger().log("NBasisVectors", NBasisVectors);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    vector<double> state;

#ifdef useComplex
    vector<numType> complexState;
    logger().log("Warning casting complex values to real discards imaginary part, GetFinalState");
    bool success = thisPtr->getFinalState(std::vector<double>(angles,angles+NAngles),complexState);
    static_cast<Matrix<double>&>(state) = std::move(complexState.real());
#else

    bool success = thisPtr->getFinalState(std::vector<double>(angles,angles+NAngles),state);
#endif
    if (!success)
        return 6;

    if (state.size() != (size_t)NBasisVectors)
    {
        logger().log("Final state does not have the write number allocated, NBasisVectors should be", state.size());
        return 7;
    }
    memcpy(finalState,state.begin(),state.size()*sizeof(state[0]));
    return 0;
}


#ifdef useComplex
int getFinalStateComplex (int NAngles, const double *angles, int NBasisVectors, __GFORTRAN_DOUBLE_COMPLEX *finalState, void *ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (angles == nullptr)
    {
        logger().log("nullptr angles passed");
        return 2;
    }
    if (finalState == nullptr)
    {
        logger().log("nullptr finalState passed");
        return 3;
    }
    if (NBasisVectors < 1)
    {
        logger().log("NBasisVectors < 1");
        return 4;
    }
    if (NAngles < 1)
    {
        logger().log("NAngles < 1");
        return 5;
    }
    if (traceInterfaceCalls)
    {
        logger().log("get_UCC_finalState called with:");
        logger().log("NAngles", NAngles);
        logger().logAccurate("angles", std::vector<double>(angles,angles+NAngles));
        logger().log("NBasisVectors", NBasisVectors);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    vector<std::complex<double>> state;
    bool success = thisPtr->getFinalState(std::vector<double>(angles,angles+NAngles),state);

    if (!success)
        return 6;

    if (state.size() != (size_t)NBasisVectors)
    {
        logger().log("Final state does not have the right number allocated, NBasisVectors should be", state.size());
        return 7;
    }
    memcpy(finalState,state.begin(),state.size()*sizeof(state[0]));
    return 0;
}
#else
int getFinalStateComplex (int, const double *, int , __GFORTRAN_DOUBLE_COMPLEX *, void *)
{
    logger().log("Cannot get a complex final state without building in complex mode");
    return 4;
}
#endif


//----------------------------------------------------------
// Computes gradient of the energy with respect to parameters.
// Inputs:
// - NAngles: number of parameters
// - angles: gate parameters
// Output:
// - gradient: dE/d theta_i. Length NAngles.
//----------------------------------------------------------
int getGradient_COMP (int NAngles, const double* angles, double* gradient, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (angles == nullptr)
    {
        logger().log("nullptr angles passed");
        return 2;
    }
    if (gradient == nullptr)
    {
        logger().log("nullptr gradient passed");
        return 3;
    }
    if (NAngles < 1)
    {
        logger().log("NAngles < 1");
        return 4;
    }
    if (traceInterfaceCalls)
    {
        logger().log("get_UCC_GRADIENT_COMP called with:");
        logger().log("NAngles", NAngles);
        logger().logAccurate("angles", std::vector<double>(angles,angles+NAngles));
        logger().log("(double*)gradient",gradient);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    vector<double> gradientV;
    if (!thisPtr->getGradientComp(std::vector<double>(angles,angles+NAngles),gradientV))
        return 5;

    if ((size_t)NAngles != gradientV.size())
    {
        logger().log("Gradient is the wrong size, This is a Bug on the implementation side");
        return -6; // Using Negative to distinguish user error from program error
    }
    memcpy(gradient,gradientV.begin(),gradientV.size()*sizeof(gradientV[0]));
    return 0;
}

//----------------------------------------------------------
// Computes Hessian (second derivatives of energy).
// Inputs:
// - NAngles: number of parameters
// - angles: gate parameters
// Output:
// - hessian: Hessian matrix as double array of size NAngles x NAngles.
//   Element hessian(i,j) corresponds to dE/(dtheta_i dtheta_j)
//----------------------------------------------------------
int getHessian_COMP (int NAngles, const double* angles, double* hessian, void* ctx)
{
    if (ctx == nullptr)
    {
        logger().log("nullptr ctx passed");
        return 1;
    }
    if (angles == nullptr)
    {
        logger().log("nullptr angles passed");
        return 2;
    }
    if (hessian == nullptr)
    {
        logger().log("nullptr hessian passed");
        return 3;
    }
    if (NAngles < 1)
    {
        logger().log("NAngles < 1");
        return 4;
    }
    if (traceInterfaceCalls)
    {
        logger().log("get_UCC_Hessian_COMP called with:");
        logger().log("NAngles", NAngles);
        logger().logAccurate("angles", std::vector<double>(angles,angles+NAngles));
        logger().log("(double*)hessian",hessian);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    std::lock_guard<std::mutex>lock(thisPtr->m_interfaceLock);
    Matrix<realNumType>::EigenMatrix hessianComp;
    if (!thisPtr->getHessianComp(std::vector<double>(angles,angles+NAngles),hessianComp))
        return 5;

    // I couldnt find any `nice' way of doing this copy which didnt rely on the internals of Eigen.
    // Therefore we're stuck with this copy and maybe the compiler can optimise it
    if (NAngles != hessianComp.rows() || NAngles != hessianComp.cols())
    {
        logger().log("Hessian has wront dimensions, This is a bug on the implementation side");
        return -6;
    }
    for (int i = 0; i < NAngles; i++)
    {
        for (int j = 0; j < NAngles; j++)
        {
            hessian[NAngles*j+i] = hessianComp(i,j);
        }
    }
    return 0;
}













