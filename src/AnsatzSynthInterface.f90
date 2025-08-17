!  Copyright (C) 2025 Bence Csakany
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

 module AnsatzSynthInterface
    interface
        !----------------------------------------------------------
        ! Enables logging/debugging of interface function calls.
        ! If val =/= 0, internal code logs arguments and usage.
        ! Can generate large printouts — use with caution.
        !----------------------------------------------------------
        subroutine setTraceInterfaceCalls(val)  bind(C, name="setTraceInterfaceCalls")
            use iso_c_binding, only: c_int
            integer(c_int),value :: val
        end subroutine setTraceInterfaceCalls

        !----------------------------------------------------------
        ! Initializes a new Ansatz object.
        ! Returns a C pointer to the created object.
        ! the Returned pointer must be given to all functions taking a ctx argument. This is almost all functions
        !----------------------------------------------------------
        function init() result(ctx) bind(C, name="init")
            use iso_c_binding, only: c_ptr
            type(c_ptr) :: ctx
        end function init

        !----------------------------------------------------------
        ! Deletes the stateAnsatzManager object and frees memory.
        ! If ctx is invalid or already cleaned up, returns error.
        ! ctx is set to nullptr after use. Passing ctx = nullptr is safe
        !----------------------------------------------------------
        function cleanup(ctx) result(status) bind(C, name="cleanup")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), intent(inout):: ctx
            integer(c_int) :: status
        end function cleanup

        !----------------------------------------------------------
        ! Sets excitation operators and parameter ordering.
        ! Inputs:
        ! - nparams: number of excitations
        ! - operators: 4*nparams integers defining excitation operators:
        !   E.g. [3, 7, 4, 8] means excite from 8th and 4th qubit into 7th and 3rd respectively
        ! - orderfile: array defining parameter ordering. E.g. 1,1,2,3 means there are 4 operators and the first two have the same parameter
        !----------------------------------------------------------
        function setExcitation(nparams, operators, orderfile, ctx) result(status) bind(C, name="setExcitation")
            use iso_c_binding, only: c_int, c_ptr
            integer(c_int), value :: nparams
            integer(c_int), intent(in) :: operators(*)
            integer(c_int), intent(in) :: orderfile(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setExcitation

        !----------------------------------------------------------
        ! Sets the Hamiltonian matrix (sparse form) in the backend.
        ! Inputs:
        ! - N: number of non-zero entries
        ! - iIndexes, jIndexes: row/column indices (1-based Fortran)
        ! - coeffs: matrix element values
        !----------------------------------------------------------
        function setHamiltonian(N, iIndexes, jIndexes, coeffs, ctx) result(status) bind(C, name="setHamiltonian")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*), jIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setHamiltonian

        !----------------------------------------------------------
        ! Sets the initial quantum state vector (sparse form).
        ! Inputs:
        ! - numQubits: total number of qubits
        ! - N: number of non-zero components
        ! - iIndexes: basis indices. the |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
        ! - coeffs: corresponding amplitudes
        !----------------------------------------------------------
        function setInitialState(numQubits, N, iIndexes, coeffs, ctx) result(status) bind(C, name="setInitialState")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: numQubits
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setInitialState

        !----------------------------------------------------------
        ! Sets the initial quantum state vector (sparse form).
        ! Inputs:
        ! - numQubits: total number of qubits
        ! - N: number of non-zero components
        ! - iIndexes: basis indices. the |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
        ! - coeffs: corresponding complex amplitudes
        !----------------------------------------------------------
        function setInitialStateComplex(numQubits, N, iIndexes, coeffs, ctx) result(status) bind(C, name="setInitialStateComplex")
            use iso_c_binding, only: c_int, c_double_complex, c_ptr
            integer(c_int), value :: numQubits
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*)
            complex(c_double_complex), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setInitialStateComplex

        !----------------------------------------------------------
        ! Computes the energy <psi|H|psi> for a given angle parameterization.
        ! Inputs:
        ! - NAngles: number of parameters
        ! - angles: array of real values
        ! Output:
        ! - energy: result of the expectation value
        !----------------------------------------------------------
        function getEnergy(NAngles, angles, energy, ctx) result(status) bind(C, name="getEnergy")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: energy
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getEnergy

        !----------------------------------------------------------
        ! Returns the full final quantum state vector after applying
        ! parameterised ansatz gates.
        ! Inputs:
        ! - NAngles: number of parameters
        ! - angles: gate parameters
        ! - NBasisVectors: expected output size
        ! Output:
        ! - finalState: output state vector in as a NBasisVectors = 2^numQubits array of double. Can be indexed by: |XXX> qubit corresponds to the number 0bXXX + 1. E.g |100> => 9
        !----------------------------------------------------------
        function getFinalState(NAngles, angles, NBasisVectors, finalState, ctx) result(status) bind(C, name="getFinalState")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            integer(c_int), value :: NBasisVectors
            real(c_double), intent(out) :: finalState(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getFinalState

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

        !----------------------------------------------------------
        ! Computes gradient of the energy with respect to parameters.
        ! Inputs:
        ! - NAngles: number of parameters
        ! - angles: gate parameters
        ! Output:
        ! - gradient: dE/d theta_i. Length NAngles.
        !----------------------------------------------------------
        function getGradient_COMP(NAngles, angles, gradient, ctx) result(status) bind(C, name="getGradient_COMP")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: gradient(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getGradient_COMP

        !----------------------------------------------------------
        ! Computes Hessian (second derivatives of energy).
        ! Inputs:
        ! - NAngles: number of parameters
        ! - angles: gate parameters
        ! Output:
        ! - hessian: Hessian matrix as double array of size NAngles x NAngles.
        !   Element hessian(i,j) corresponds to dE/(dtheta_i dtheta_j)
        !----------------------------------------------------------
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

