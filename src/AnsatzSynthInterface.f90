!  Copyright (C) 2025 Bence Csakany
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

 module AnsatzSynthInterface
    interface
        !> \brief Enable or disable logging/tracing of interface function calls.
        !> \see setTraceInterfaceCalls() in AnsatzSynthInterface.cpp for the full description.
        !> \param val If non-zero, subsequent calls to the interface functions log their arguments and usage.
        subroutine setTraceInterfaceCalls(val)  bind(C, name="setTraceInterfaceCalls")
            use iso_c_binding, only: c_int
            integer(c_int),value :: val
        end subroutine setTraceInterfaceCalls

        !> \brief Create a new stateAnsatzManager and return an opaque handle to it.
        !> \see init() in AnsatzSynthInterface.cpp for the full description.
        !> \return `ctx`, a C pointer to the created object. Must be passed as the `ctx` argument to almost every other function in this interface, and must eventually be passed to cleanup() to avoid leaking the object.
        function init() result(ctx) bind(C, name="init")
            use iso_c_binding, only: c_ptr
            type(c_ptr) :: ctx
        end function init

        !> \brief Destroy the stateAnsatzManager created by init() and free its memory.
        !> \see cleanup() in AnsatzSynthInterface.cpp for the full description.
        !> \param ctx The pointer previously returned by init(). Set to a null pointer after use. Passing an already-null `ctx` is safe.
        !> \return `status`: 0 on success, 1 if `ctx` was null.
        function cleanup(ctx) result(status) bind(C, name="cleanup")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), intent(inout):: ctx
            integer(c_int) :: status
        end function cleanup

        !> \brief Set the excitation operators and parameter ordering, with an implicit scale of 1 for every operator.
        !> \see setExcitation() in AnsatzSynthInterface.cpp for the full description.
        !> \param nparams Number of excitation operators. Must be at least 1.
        !> \param operators 1-indexed array of `4*nparams` integers defining excitation operators. E.g. `[3, 7, 4, 8]` means excite from the 8th and 4th qubit into the 7th and 3rd respectively.
        !> \param orderfile 1-indexed array of `nparams` integers giving the parameter-dependency mapping (the "Order file"). E.g. `1,1,2,3` means there are 4 operators and the first two share the same free parameter.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see setExcitation() in AnsatzSynthInterface.cpp for the full list of error codes.
        function setExcitation(nparams, operators, orderfile, ctx) result(status) bind(C, name="setExcitation")
            use iso_c_binding, only: c_int, c_ptr
            integer(c_int), value :: nparams
            integer(c_int), intent(in) :: operators(*)
            integer(c_int), intent(in) :: orderfile(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setExcitation

        !> \brief Set the excitation operators and parameter ordering, with optional per-operator scale factors.
        !> \see setExcitationScale() in AnsatzSynthInterface.cpp for the full description.
        !> \param nparams Number of excitation operators. Must be at least 1.
        !> \param operators 1-indexed array of `4*nparams` integers defining excitation operators. E.g. `[3, 7, 4, 8]` means excite from the 8th and 4th qubit into the 7th and 3rd respectively.
        !> \param orderfile 1-indexed array of `nparams` integers giving the parameter-dependency mapping (the "Order file"). E.g. `1,1,2,3` means there are 4 operators and the first two share the same free parameter.
        !> \param scaleFactor Array of `nparams` scale factors applied to each operator's free parameter.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see setExcitationScale() in AnsatzSynthInterface.cpp for the full list of error codes.
        function setExcitationScale(nparams, operators, orderfile, scaleFactor, ctx) result(status)&
        bind(C, name="setExcitationScale")
            use iso_c_binding, only: c_int, c_ptr, c_double
            integer(c_int), value :: nparams
            integer(c_int), intent(in) :: operators(*)
            integer(c_int), intent(in) :: orderfile(*)
            real(c_double), intent(in) :: scaleFactor(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setExcitationScale

        !> \brief Set the Hamiltonian as a sparse matrix given by parallel index/coefficient arrays.
        !> \see setHamiltonian() in AnsatzSynthInterface.cpp for the full description.
        !> \param N Number of non-zero entries. Must be at least 1.
        !> \param iIndexes 1-indexed row indices of non-zero Hamiltonian entries, length `N`.
        !> \param jIndexes 1-indexed column indices of non-zero Hamiltonian entries, length `N`.
        !> \param coeffs Values of the non-zero Hamiltonian entries, length `N`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see setHamiltonian() in AnsatzSynthInterface.cpp for the full list of error codes.
        function setHamiltonian(N, iIndexes, jIndexes, coeffs, ctx) result(status) bind(C, name="setHamiltonian")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*), jIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setHamiltonian

        !> \brief Set the initial statevector, real amplitudes.
        !> \see setInitialState() in AnsatzSynthInterface.cpp for the full description.
        !> \param numQubits Total number of qubits. Must be at least 1.
        !> \param N Number of non-zero components. Must be at least 1.
        !> \param iIndexes 1-indexed basis-state indices, length `N`. Basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
        !> \param coeffs Amplitudes corresponding to each entry in `iIndexes`, length `N`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see setInitialState() in AnsatzSynthInterface.cpp for the full list of error codes.
        function setInitialState(numQubits, N, iIndexes, coeffs, ctx) result(status) bind(C, name="setInitialState")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: numQubits
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setInitialState

        !> \brief Set the initial statevector, complex amplitudes. Only functional in complex-mode builds.
        !> \see setInitialStateComplex() in AnsatzSynthInterface.cpp for the full description.
        !> \param numQubits Total number of qubits. Must be at least 1.
        !> \param N Number of non-zero components. Must be at least 1.
        !> \param iIndexes 1-indexed basis-state indices, length `N`. Basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
        !> \param coeffs Complex amplitudes corresponding to each entry in `iIndexes`, length `N`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see setInitialStateComplex() in AnsatzSynthInterface.cpp for the full list of error codes, including the non-complex-mode stub behavior.
        function setInitialStateComplex(numQubits, N, iIndexes, coeffs, ctx) result(status) bind(C, name="setInitialStateComplex")
            use iso_c_binding, only: c_int, c_double_complex, c_ptr
            integer(c_int), value :: numQubits
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*)
            complex(c_double_complex), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setInitialStateComplex

        !> \brief Set the angles and compute the resulting energy, \f$\braket{\psi|H|\psi}\f$.
        !> \see getEnergy() in AnsatzSynthInterface.cpp for the full description.
        !> \param NAngles Number of angles given. May be either the compressed or decompressed count.
        !> \param angles The angles to evolve to, length `NAngles`.
        !> \param energy Output: the resulting expectation value.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see getEnergy() in AnsatzSynthInterface.cpp for the full list of error codes.
        function getEnergy(NAngles, angles, energy, ctx) result(status) bind(C, name="getEnergy")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: energy
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getEnergy

        !> \brief Set the angles and return the full, dense final statevector, \f$\psi\f$.
        !> \see getFinalState() in AnsatzSynthInterface.cpp for the full description.
        !> \param NAngles Number of angles given. May be either the compressed or decompressed count.
        !> \param angles The angles to evolve to, length `NAngles`.
        !> \param NBasisVectors Expected length of `finalState`, i.e. \f$2^{numQubits}\f$.
        !> \param finalState Output: the dense final statevector, length `NBasisVectors`. Indexed such that basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see getFinalState() in AnsatzSynthInterface.cpp for the full list of error codes.
        !> \note If built in complex mode, this discards the imaginary part of the result and logs a warning — use getFinalStateComplex() to retrieve the full complex state.
        function getFinalState(NAngles, angles, NBasisVectors, finalState, ctx) result(status) bind(C, name="getFinalState")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            integer(c_int), value :: NBasisVectors
            real(c_double), intent(out) :: finalState(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getFinalState

        !> \brief Set the angles and return the full, dense, complex final statevector. Only functional in complex-mode builds.
        !> \see getFinalStateComplex() in AnsatzSynthInterface.cpp for the full description.
        !> \param NAngles Number of angles given. May be either the compressed or decompressed count.
        !> \param angles The angles to evolve to, length `NAngles`.
        !> \param NBasisVectors Expected length of `finalState`, i.e. \f$2^{numQubits}\f$.
        !> \param finalState Output: the dense complex final statevector, length `NBasisVectors`. Indexed such that basis state \f$|XXX\rangle\f$ corresponds to index `0bXXX + 1`, e.g. \f$|100\rangle\f$ corresponds to `9`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see getFinalStateComplex() in AnsatzSynthInterface.cpp for the full list of error codes, including the non-complex-mode stub behavior.
        function getFinalStateComplex(NAngles, angles, NBasisVectors, finalState, ctx) result(status)&
        bind(C, name="getFinalStateComplex")
            use iso_c_binding, only: c_double, c_double_complex, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            integer(c_int), value :: NBasisVectors
            complex(c_double_complex), intent(out) :: finalState(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getFinalStateComplex

        !> \brief Set the angles and compute the compressed-format gradient of the energy, \f$\frac{\partial \braket{\psi|H|\psi}}{\partial \theta_i}\f$.
        !> \see getGradient_COMP() in AnsatzSynthInterface.cpp for the full description.
        !> \param NAngles Number of angles given and expected length of `gradient`. Must be the compressed count (one per unique parameter).
        !> \param angles The angles to evolve to, length `NAngles`.
        !> \param gradient Output: the gradient, one entry per unique parameter, length `NAngles`.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see getGradient_COMP() in AnsatzSynthInterface.cpp for the full list of error codes.
        function getGradient_COMP(NAngles, angles, gradient, ctx) result(status) bind(C, name="getGradient_COMP")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: gradient(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getGradient_COMP

        !> \brief Set the angles and compute the compressed-format Hessian of the energy, \f$\frac{\partial^2 \braket{\psi|H|\psi}}{\partial \theta_i \partial \theta_j}\f$.
        !> \see getHessian_COMP() in AnsatzSynthInterface.cpp for the full description.
        !> \param NAngles Number of angles given, and both dimensions of `hessian`. Must be the compressed count (one per unique parameter).
        !> \param angles The angles to evolve to, length `NAngles`.
        !> \param hessian Output: the Hessian matrix, stored column-major as a flat `NAngles * NAngles` array, where element `hessian(NAngles*j + i)` corresponds to \f$\partial^2 E/\partial\theta_i\partial\theta_j\f$.
        !> \param ctx Handle returned by init().
        !> \return `status`: 0 on success; see getHessian_COMP() in AnsatzSynthInterface.cpp for the full list of error codes.
        function getHessian_COMP(NAngles, angles, hessian, ctx) result(status) bind(C, name="getHessian_COMP")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: hessian(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getHessian_COMP
    end interface

end module AnsatzSynthInterface
