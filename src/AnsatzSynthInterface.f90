!  Copyright (C) 2025 Bence Csakany
!  This Source Code Form is subject to the terms of the Mozilla Public
!  License, v. 2.0. If a copy of the MPL was not distributed with this
!  file, You can obtain one at https://mozilla.org/MPL/2.0/.

 module AnsatzSynthInterface
    interface
        subroutine setTraceInterfaceCalls(val)  bind(C, name="setTraceInterfaceCalls")
            use iso_c_binding, only: c_int
            integer(c_int),value :: val
        end subroutine setTraceInterfaceCalls

        function init() result(ctx) bind(C, name="init")
            use iso_c_binding, only: c_ptr
            type(c_ptr) :: ctx
        end function init

        function cleanup(ctx) result(status) bind(C, name="cleanup")
            use iso_c_binding, only: c_ptr, c_int
            type(c_ptr), intent(inout):: ctx
            integer(c_int) :: status
        end function cleanup

        function setExcitation(nparams, operators, orderfile, ctx) result(status) bind(C, name="setExcitation")
            use iso_c_binding, only: c_int, c_ptr
            integer(c_int), value :: nparams
            integer(c_int), intent(in) :: operators(*)
            integer(c_int), intent(in) :: orderfile(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setExcitation

        function setHamiltonian(N, iIndexes, jIndexes, coeffs, ctx) result(status) bind(C, name="setHamiltonian")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*), jIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setHamiltonian



        function setInitialState(numQubits, N, iIndexes, coeffs, ctx) result(status) bind(C, name="setInitialState")
            use iso_c_binding, only: c_int, c_double, c_ptr
            integer(c_int), value :: numQubits
            integer(c_int), value :: N
            integer(c_int), intent(in) :: iIndexes(*)
            real(c_double), intent(in) :: coeffs(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function setInitialState




        function getEnergy(NAngles, angles, energy, ctx) result(status) bind(C, name="getEnergy")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: energy
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getEnergy

        function getFinalState(NAngles, angles, NBasisVectors, finalState, ctx) result(status) bind(C, name="getFinalState")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            integer(c_int), value :: NBasisVectors
            real(c_double), intent(out) :: finalState(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getFinalState

        function getGradient_COMP(NAngles, angles, gradient, ctx) result(status) bind(C, name="getGradient_COMP")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: angles(*)
            real(c_double), intent(out) :: gradient(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getGradient_COMP

        function getHessian_COMP(NAngles, params, hessian, ctx) result(status) bind(C, name="getHessian_COMP")
            use iso_c_binding, only: c_double, c_ptr, c_int
            integer(c_int), value :: NAngles
            real(c_double), intent(in) :: params(*)
            real(c_double), intent(out) :: hessian(*)
            type(c_ptr), value :: ctx
            integer(c_int) :: status
        end function getHessian_COMP
    end interface

end module AnsatzSynthInterface

