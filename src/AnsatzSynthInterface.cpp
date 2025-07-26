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

void setTraceInterfaceCalls(int val)
{
    traceInterfaceCalls = val != 0;
}

void *init()
{
    stateAnsatzManager* ptr = new stateAnsatzManager;
    if (traceInterfaceCalls)
        logger().log("Constructed object",ptr);
    return ptr;
}

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

int setExcitation(int nparams, const int *operators, const int *orderfile, void *ctx)
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
        logger().log("setExcitation called with:");
        logger().log("nParams", nparams);
        logger().log("operators", std::vector<int>(operators,operators+nparams*4));
        logger().log("orderfile", std::vector<int>(orderfile,orderfile+nparams));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
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
        parameterDependency.push_back({orderfile[i]-1,1});
    }
    succ = thisPtr->storeParameterDependencies(parameterDependency);
    if (!succ)
        return 6;
    return 0;
}

int setHamiltonian(int N, const int* iIndexes, const int* jIndexes, const double* coeffs, void* ctx)
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
        logger().log("iIndexes", std::vector<int>(iIndexes,iIndexes+N));
        logger().log("jIndexes", std::vector<int>(jIndexes,jIndexes+N));
        logger().log("coeffs", std::vector<int>(coeffs,coeffs+N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
    if (N < 1)
    {
        logger().log("N < 1");
        return 2;
    }
    std::vector<int> iIndexesV(iIndexes,iIndexes+N);
    std::vector<int> jIndexesV(jIndexes,jIndexes+N);
    for (int& iIndex : iIndexesV)
        iIndex--;
    for (int& jIndex : jIndexesV)
        jIndex--;
    std::vector<realNumType> CoeffsV(coeffs,coeffs+N);
    bool success = thisPtr->storeHamiltonian(iIndexesV,jIndexesV,CoeffsV);
    if (!success)
        return 3;
    return 0;
}

int setInitialState(int numQubits, int N, const int* iIndexes, const double* coeffs, void* ctx)
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
        logger().log("coeffs", std::vector<int>(coeffs,coeffs+N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager* thisPtr = static_cast<stateAnsatzManager*>(ctx);
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
    std::vector<realNumType> CoeffsV(coeffs,coeffs+N);
    bool success = thisPtr->storeInitial(numQubits,iIndexesV,CoeffsV);
    if (!success)
        return 3;
    return 0;
}

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
    bool success = thisPtr->getExpectationValue(std::vector<double>(angles,angles+NAngles),*energy);
    if (!success)
        return 3;
    return 0;
}

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
    vector<double> state;
    bool success = thisPtr->getFinalState(std::vector<double>(angles,angles+NAngles),state);
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













