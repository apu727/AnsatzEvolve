/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "Generated/AnsatzSynthInterface.h"
#include "AnsatzManager.h"
#include "logger.h"
#include <stdio.h>

bool traceInterfaceCalls = false; //!< see setTraceInterfaceCalls(int)

/**
 * \brief Enable or disable logging/tracing of interface function calls.
 * \param val If non-zero, subsequent calls to the interface functions log their arguments and usage.
 * \note Can generate large printouts (e.g. full angle/coefficient arrays) — use with caution, particularly for large systems.
 */
void setTraceInterfaceCalls(int val)
{
    traceInterfaceCalls = val != 0;
}

/**
 * \brief Create a new stateAnsatzManager and return an opaque handle to it.
 * \return A C pointer to the created object. This must be passed as the `ctx` argument to almost every other function in this interface.
 * \note The returned pointer must eventually be passed to cleanup() to avoid leaking the object.
 */
void *init()
{
    stateAnsatzManager *ptr = new stateAnsatzManager;
    if (traceInterfaceCalls) logger().log("Constructed object", ptr);
    return ptr;
}

/**
 * \brief Destroy the stateAnsatzManager created by init() and free its memory.
 * \param ctx Address of the pointer previously returned by init(), passed as `void*` (i.e. a `void**`). Passing a pointer to `nullptr` is safe and does nothing.
 * \retval 0 Success (including the no-op case where `*ctx` was already `nullptr`).
 * \retval 1 `ctx` itself is `nullptr` — this indicates a bug on the caller's side, not a normal "already cleaned up" state.
 * \note `*ctx` is set to `nullptr` after the object is deleted, so calling cleanup() twice on the same handle is safe.
 */
int cleanup(void *ctx)
{
    if (traceInterfaceCalls) logger().log("cleanup called with void**", ctx);
    if (ctx == nullptr)
    {
        logger().log("(void**) ctx is nullptr, something has gone wrong. Potential Leak");
        return 1;
    }

    void **myCtx = (void **) ctx;
    if (traceInterfaceCalls) logger().log("cleanup called with void*", *myCtx);

    if (*myCtx == nullptr) return 0;
    delete static_cast<stateAnsatzManager *>(*myCtx);
    *myCtx = nullptr;
    return 0;
}

/**
 * \brief Set the excitation operators and parameter ordering (equivalent to stateAnsatzManager::storeOperators() + stateAnsatzManager::storeParameterDependencies(), with an implicit scale of 1 for every operator).
 * \see setExcitationScale() for the full parameter description; this is a thin wrapper that passes `scale = nullptr`.
 */
int setExcitation(int nparams, const int *operators, const int *orderfile, void *ctx)
{
    return setExcitationScale(nparams, operators, orderfile, nullptr, ctx);
}

/**
 * \brief Set the excitation operators and parameter ordering, with optional per-operator scale factors (equivalent to stateAnsatzManager::storeOperators() + stateAnsatzManager::storeParameterDependencies()).
 * \param nparams Number of excitation operators. Must be at least 1.
 * \param operators 1-indexed (Fortran-style) array of `4*nparams` integers, laid out as 4 blocks of `nparams` values (i.e. `operators[nparams*j + i]` is component `j` of operator `i`), defining `create`/`create`/`destroy`/`destroy` qubit indices for each excitation. E.g. `[3, 7, 4, 8]` (for `nparams=1`) means excite from the 8th and 4th qubit into the 7th and 3rd respectively.
 * \param orderfile 1-indexed array of `nparams` integers giving the parameter-dependency mapping (the "Order file" — see the \ref manual_page "manual"). E.g. `1,1,2,3` means there are 4 operators and the first two share the same free parameter.
 * \param scale Optional array of `nparams` scale factors applied to each operator's free parameter (see stateAnsatzManager::storeParameterDependencies()). Pass `nullptr` for a scale of 1 on every operator.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `nparams < 1`.
 * \retval 3 An entry in \p operators is out of bounds (qubit index exceeds 128).
 * \retval 4 stateAnsatzManager::storeOperators() failed — see the log for details.
 * \retval 5 An entry in \p orderfile is out of bounds (exceeds \p nparams).
 * \retval 6 stateAnsatzManager::storeParameterDependencies() failed — see the log for details.
 */
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
        logger().log("operators", std::vector<int>(operators, operators + nparams * 4));
        logger().log("orderfile", std::vector<int>(orderfile, orderfile + nparams));
        if (scale) logger().log("scale", std::vector<int>(scale, scale + nparams));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);

    std::vector<stateRotate::exc> excs(nparams);
    for (int i = 0; i < nparams; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (operators[nparams * j + i] - 1 > std::numeric_limits<int8_t>::max())
            {
                //This is likely if we are reading out of bounds in the array
                logger().log("Excitation out of bounds, max is 128", operators[nparams * j + i]);
                return 3;
            }
            excs[i][j] = operators[nparams * j + i] - 1; // Fortran order
        }
    }
    bool succ = thisPtr->storeOperators(excs);
    if (!succ) return 4;
    std::vector<std::pair<int, realNumType>> parameterDependency;
    for (int i = 0; i < nparams; i++)
    {
        if (orderfile[i] > nparams)
        {
            logger().log("orderfile out of bounds", orderfile[i]);
            return 5;
        }
        if (scale)
            parameterDependency.push_back({orderfile[i] - 1, scale[i]});
        else
            parameterDependency.push_back({orderfile[i] - 1, 1});
    }
    succ = thisPtr->storeParameterDependencies(parameterDependency);
    if (!succ) return 6;
    return 0;
}

/**
 * \brief Set the Hamiltonian as a sparse matrix (equivalent to stateAnsatzManager::storeHamiltonian()).
 * \param N Number of non-zero entries. Must be at least 1.
 * \param iIndexes 1-indexed (Fortran-style) row indices of non-zero Hamiltonian entries, length \p N.
 * \param jIndexes 1-indexed (Fortran-style) column indices of non-zero Hamiltonian entries, length \p N.
 * \param coeffs Values of the non-zero Hamiltonian entries, length \p N.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `N < 1`.
 * \retval 3 stateAnsatzManager::storeHamiltonian() failed — see the log for details.
 */
int setHamiltonian(int N, const int *iIndexes, const int *jIndexes, const double *coeffs, void *ctx)
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
        logger().log("iIndexes", std::vector<int>(iIndexes, iIndexes + N));
        logger().log("jIndexes", std::vector<int>(jIndexes, jIndexes + N));
        logger().log("coeffs", std::vector<int>(coeffs, coeffs + N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    if (N < 1)
    {
        logger().log("N < 1");
        return 2;
    }
    std::vector<int> iIndexesV(iIndexes, iIndexes + N);
    std::vector<int> jIndexesV(jIndexes, jIndexes + N);
    for (int &iIndex : iIndexesV)
        iIndex--;
    for (int &jIndex : jIndexesV)
        jIndex--;
    std::vector<realNumType> CoeffsV(coeffs, coeffs + N);
    bool success = thisPtr->storeHamiltonian(std::move(iIndexesV), std::move(jIndexesV), std::move(CoeffsV));
    if (!success) return 3;
    return 0;
}

/**
 * \brief Set the initial statevector, real amplitudes (equivalent to stateAnsatzManager::storeInitial()).
 * \param numQubits Total number of qubits. Must be at least 1.
 * \param N Number of non-zero components. Must be at least 1.
 * \param iIndexes 1-indexed basis-state indices, length \p N. The basis state \f$|XXX\rangle\f$ corresponds to the index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
 * \param coeffs Amplitudes corresponding to each entry in \p iIndexes, length \p N.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `N < 1` or `numQubits < 1`.
 * \retval 3 stateAnsatzManager::storeInitial() failed — see the log for details.
 */
int setInitialState(int numQubits, int N, const int *iIndexes, const double *coeffs, void *ctx)
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
        logger().log("iIndexes", std::vector<int>(iIndexes, iIndexes + N));
        logger().log("coeffs", std::vector<double>(coeffs, coeffs + N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
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
    std::vector<int> iIndexesV(iIndexes, iIndexes + N);
    for (int &iIndex : iIndexesV)
        iIndex--;
    std::vector<numType> CoeffsV(coeffs, coeffs + N);
    bool success = thisPtr->storeInitial(numQubits, iIndexesV, CoeffsV);
    if (!success) return 3;
    return 0;
}

/**
 * \brief Set the initial statevector, complex amplitudes (equivalent to stateAnsatzManager::storeInitial()). Only available in complex-mode builds — see setInitialState() for the real-valued equivalent.
 * \param numQubits Total number of qubits. Must be at least 1.
 * \param N Number of non-zero components. Must be at least 1.
 * \param iIndexes 1-indexed basis-state indices, length \p N. The basis state \f$|XXX\rangle\f$ corresponds to the index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
 * \param coeffs Complex amplitudes corresponding to each entry in \p iIndexes, length \p N.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `N < 1` or `numQubits < 1`.
 * \retval 3 stateAnsatzManager::storeInitial() failed — see the log for details.
 * \retval 4 The library was not built in complex mode; this function is a no-op stub in that case (see `USE_COMPLEX`/the `-DCOMPLEX_MODE` build option in the manual).
 */
#ifdef useComplex
int setInitialStateComplex(int numQubits, int N, const int *iIndexes, const __GFORTRAN_DOUBLE_COMPLEX *coeffs, void *ctx)
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
        logger().log("iIndexes", std::vector<int>(iIndexes, iIndexes + N));
        logger().log("coeffs", std::vector<std::complex<double>>(coeffs, coeffs + N));
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
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
    std::vector<int> iIndexesV(iIndexes, iIndexes + N);
    for (int &iIndex : iIndexesV)
        iIndex--;
    std::vector<std::complex<double>> CoeffsV(coeffs, coeffs + N);
    bool success = thisPtr->storeInitial(numQubits, iIndexesV, CoeffsV);
    if (!success) return 3;
    return 0;
}
#else
int setInitialStateComplex(
    int MAYBE_UNUSED numQubits,
    int MAYBE_UNUSED N,
    const int MAYBE_UNUSED *iIndexes,
    const __GFORTRAN_DOUBLE_COMPLEX MAYBE_UNUSED *coeffs,
    void MAYBE_UNUSED *ctx
)
{
    logger().log("Cannot set a complex initial state without building in complex mode");
    return 4;
}
#endif

/**
 * \brief Set the angles and compute the resulting energy, \f$\braket{\psi|H|\psi}\f$ (equivalent to stateAnsatzManager::getExpectationValue(const std::vector<realNumType>&, realNumType&)).
 * \param NAngles Number of angles given. May be either the compressed or decompressed count — see \ref stateAnsatzManager "Angle formats" in the manager documentation.
 * \param angles The angles to evolve to, length \p NAngles.
 * \param energy Output: the resulting expectation value.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `angles` is `nullptr`.
 * \retval 3 `NAngles < 1`, or the evolution/expectation-value computation failed — see the log for details.
 */
int getEnergy(int NAngles, const double *angles, double *energy, void *ctx)
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
        logger().log("(double*)angles", (void *) angles);
        logger().log("(double*)exptvalue", energy);
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    bool success = thisPtr->getExpectationValue(std::vector<double>(angles, angles + NAngles), *energy);
    if (!success) return 3;
    return 0;
}
/**
 * \brief Set the angles and return the full, dense final statevector, \f$\psi\f$ (equivalent to stateAnsatzManager::getFinalState(const std::vector<realNumType>&, vector<numType>&)).
 * \param NAngles Number of angles given. May be either the compressed or decompressed count — see \ref stateAnsatzManager "Angle formats" in the manager documentation.
 * \param angles The angles to evolve to, length \p NAngles.
 * \param NBasisVectors Expected length of \p finalState, i.e. \f$2^{numQubits}\f$.
 * \param finalState Output: the dense final statevector, length \p NBasisVectors. Indexed such that basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `angles` is `nullptr`.
 * \retval 3 `finalState` is `nullptr`.
 * \retval 4 `NBasisVectors < 1`.
 * \retval 5 `NAngles < 1`.
 * \retval 6 stateAnsatzManager::getFinalState() failed — see the log for details.
 * \retval 7 The computed state's size did not match \p NBasisVectors — this indicates a bug on the implementation side, not a user error.
 * \note If built in complex mode, this discards the imaginary part of the result and logs a warning — use getFinalStateComplex() to retrieve the full complex state.
 */
int getFinalState(int NAngles, const double *angles, int NBasisVectors, double *finalState, void *ctx)
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
        logger().logAccurate("angles", std::vector<double>(angles, angles + NAngles));
        logger().log("NBasisVectors", NBasisVectors);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    vector<double> state;

#ifdef useComplex
    vector<numType> complexState;
    logger().log("Warning casting complex values to real discards imaginary part, GetFinalState");
    bool success = thisPtr->getFinalState(std::vector<double>(angles, angles + NAngles), complexState);
    static_cast<Matrix<double> &>(state) = std::move(complexState.real());
#else

    bool success = thisPtr->getFinalState(std::vector<double>(angles, angles + NAngles), state);
#endif
    if (!success) return 6;

    if (state.size() != (size_t) NBasisVectors)
    {
        logger().log("Final state does not have the write number allocated, NBasisVectors should be", state.size());
        return 7;
    }
    memcpy(finalState, state.begin(), state.size() * sizeof(state[0]));
    return 0;
}

/**
 * \brief Set the angles and return the full, dense, complex final statevector (equivalent to stateAnsatzManager::getFinalState(const std::vector<realNumType>&, vector<numType>&)). Only available in complex-mode builds — see getFinalState() for the real-valued equivalent.
 * \param NAngles Number of angles given. May be either the compressed or decompressed count — see \ref stateAnsatzManager "Angle formats" in the manager documentation.
 * \param angles The angles to evolve to, length \p NAngles.
 * \param NBasisVectors Expected length of \p finalState, i.e. \f$2^{numQubits}\f$.
 * \param finalState Output: the dense complex final statevector, length \p NBasisVectors. Indexed such that basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `angles` is `nullptr`.
 * \retval 3 `finalState` is `nullptr`.
 * \retval 4 `NBasisVectors < 1`.
 * \retval 5 `NAngles < 1`.
 * \retval 6 stateAnsatzManager::getFinalState() failed — see the log for details.
 * \retval 7 The computed state's size did not match \p NBasisVectors — this indicates a bug on the implementation side, not a user error.
 * \note If the library was not built in complex mode, this function is instead a no-op stub that logs an error and returns `4`.
 */
#ifdef useComplex
int getFinalStateComplex(int NAngles, const double *angles, int NBasisVectors, __GFORTRAN_DOUBLE_COMPLEX *finalState, void *ctx)
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
        logger().logAccurate("angles", std::vector<double>(angles, angles + NAngles));
        logger().log("NBasisVectors", NBasisVectors);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    vector<std::complex<double>> state;
    bool success = thisPtr->getFinalState(std::vector<double>(angles, angles + NAngles), state);

    if (!success) return 6;

    if (state.size() != (size_t) NBasisVectors)
    {
        logger().log("Final state does not have the right number allocated, NBasisVectors should be", state.size());
        return 7;
    }
    memcpy(finalState, state.begin(), state.size() * sizeof(state[0]));
    return 0;
}
#else
int getFinalStateComplex(
    int MAYBE_UNUSED NAngles,
    const double MAYBE_UNUSED *angles,
    int MAYBE_UNUSED NBasisVectors,
    __GFORTRAN_DOUBLE_COMPLEX MAYBE_UNUSED *finalState,
    void MAYBE_UNUSED *ctx
)
{
    logger().log("Cannot get a complex final state without building in complex mode");
    return 4;
}
#endif

/**
 * \brief Set the angles and compute the compressed-format gradient of the energy, \f$\frac{\partial \braket{\psi|H|\psi}}{\partial \theta_i}\f$ (equivalent to stateAnsatzManager::getGradientComp(const std::vector<realNumType>&, vector<realNumType>&)).
 * \param NAngles Number of angles given and expected length of \p gradient. Must be the compressed count (one per unique parameter) — see \ref stateAnsatzManager "Angle formats" in the manager documentation.
 * \param angles The angles to evolve to, length \p NAngles.
 * \param gradient Output: the gradient, one entry per unique parameter, length \p NAngles.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `angles` is `nullptr`.
 * \retval 3 `gradient` is `nullptr`.
 * \retval 4 `NAngles < 1`.
 * \retval 5 stateAnsatzManager::getGradientComp() failed — see the log for details.
 * \retval -6 The computed gradient's size did not match \p NAngles — this indicates a bug on the implementation side, not a user error.
 */
int getGradient_COMP(int NAngles, const double *angles, double *gradient, void *ctx)
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
        logger().logAccurate("angles", std::vector<double>(angles, angles + NAngles));
        logger().log("(double*)gradient", gradient);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    vector<double> gradientV;
    if (!thisPtr->getGradientComp(std::vector<double>(angles, angles + NAngles), gradientV)) return 5;

    if ((size_t) NAngles != gradientV.size())
    {
        logger().log("Gradient is the wrong size, This is a Bug on the implementation side");
        return -6; // Using Negative to distinguish user error from program error
    }
    memcpy(gradient, gradientV.begin(), gradientV.size() * sizeof(gradientV[0]));
    return 0;
}

/**
 * \brief Set the angles and compute the compressed-format Hessian of the energy, \f$\frac{\partial^2 \braket{\psi|H|\psi}}{\partial \theta_i \partial \theta_j}\f$ (equivalent to stateAnsatzManager::getHessianComp(const std::vector<realNumType>&, Matrix<realNumType>::EigenMatrix&)).
 * \param NAngles Number of angles given, and both dimensions of \p hessian. Must be the compressed count (one per unique parameter) — see \ref stateAnsatzManager "Angle formats" in the manager documentation.
 * \param angles The angles to evolve to, length \p NAngles.
 * \param hessian Output: the Hessian matrix, stored column-major as a flat `NAngles * NAngles` array, where element `hessian[NAngles*j + i]` corresponds to \f$\partial^2 E/\partial\theta_i\partial\theta_j\f$.
 * \param ctx Handle returned by init().
 * \retval 0 Success.
 * \retval 1 `ctx` is `nullptr`.
 * \retval 2 `angles` is `nullptr`.
 * \retval 3 `hessian` is `nullptr`.
 * \retval 4 `NAngles < 1`.
 * \retval 5 stateAnsatzManager::getHessianComp() failed — see the log for details.
 * \retval -6 The computed Hessian's dimensions did not match \p NAngles — this indicates a bug on the implementation side, not a user error.
 */
int getHessian_COMP(int NAngles, const double *angles, double *hessian, void *ctx)
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
        logger().logAccurate("angles", std::vector<double>(angles, angles + NAngles));
        logger().log("(double*)hessian", hessian);
        //Not logging final state because its massive!
        logger().log("ctx", ctx);
    }
    stateAnsatzManager *thisPtr = static_cast<stateAnsatzManager *>(ctx);
    std::lock_guard<std::mutex> lock(thisPtr->m_interfaceLock);
    Matrix<realNumType>::EigenMatrix hessianComp;
    if (!thisPtr->getHessianComp(std::vector<double>(angles, angles + NAngles), hessianComp)) return 5;

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
            hessian[NAngles * j + i] = hessianComp(i, j);
        }
    }
    return 0;
}
