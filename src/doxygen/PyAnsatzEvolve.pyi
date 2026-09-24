"""!
\brief Python bindings for AnsatzEvolve, exposing stateAnsatzManager and a handful of file-loading helpers via pybind11.

\note Unlike the C interface (see AnsatzSynthInterface.cpp), most `get*`-family methods on
stateAnsatzManager discard the underlying C++ success flag and always return a value, even on
failure — see the \warning on each affected method. Failures are only visible via the logged
error message (see setTraceInterfaceCalls-equivalent behaviour is not exposed to Python; errors
are always logged internally regardless of tracing).

\note Index conventions are **not uniform** across this module. Some methods are thin lambdas
that adjust for Python/Fortran-style 1-indexing; others are bound directly to the C++ methods
and expect the same 0-indexed convention used internally. Each method below states which
convention it uses — check carefully when porting code from the Fortran or C interface.
"""

from typing import overload
import numpy


class stateAnsatzManager:
    """!
    \brief Manages the full lifecycle of a single ansatz: state setup, evolution, and property evaluation.

    ## Usage
    A `stateAnsatzManager` is configured by calling the `store*` methods (in any order) to
    provide the initial state, operators (excitations), Hamiltonian, parameter dependencies,
    nuclear energy, and run path. None of these trigger any computation on their own — they
    just record the data.

    The manager is lazily constructed: the first call to any evolution-triggering method
    (`setAngles`, any `get*` method, `optimise`) checks whether setup is complete and, if so,
    builds the ansatz automatically. If required data is missing or inconsistent, an error is
    logged; see the per-method notes below for how (or whether) this surfaces in the return value.

    Once construction has succeeded, calling any `store*` method again will fail (setup cannot
    be modified after construction) — construct a new `stateAnsatzManager` instead.

    ## Angle formats
    Angles can be provided/retrieved in two equivalent representations:
    - **Decompressed**: one angle per operator/excitation.
    - **Compressed**: one angle per *unique* free parameter, as defined by
      storeParameterDependencies().

    Methods taking a plain list of angles accept either length and auto-detect which was given.
    Methods with a `Comp` suffix always work in the compressed representation.
    """

    def __init__(self) -> None:
        """!
        \brief Construct an empty, unconfigured manager. The run path defaults to the current working directory (see storeRunPath()).
        """
        ...

    def storeOperators(self, operators: list[list[int]]) -> None:
        """!
        \brief Store the excitation operators that define the ansatz.
        \param operators A list of 4-element `[create, create, destroy, destroy]` qubit-index lists, one per excitation. Must be non-empty.
        \warning Indices here are **1-indexed** (matching the Fortran/C convention) — internally 1 is subtracted from each before storing.
        \warning Unlike its `store*` siblings, this method returns `None` regardless of success or failure — the underlying `bool` result is discarded. A failure (e.g. an operator index too large, or an empty list) is only visible via the logged error message.
        """
        ...

    def storeInitial(self, numberOfQubits: int, indexes: list[int], coeffs: list[float]) -> bool:
        """!
        \brief Store the initial statevector as a sparse list of (index, coefficient) pairs.
        \param numberOfQubits Number of qubits; must be at least 1. Determines the full statevector length (\f$2^{numberOfQubits}\f$).
        \param indexes Basis-state indices with non-zero amplitude.
        \param coeffs Amplitudes corresponding to each entry in `indexes`. Must be the same length as `indexes`. In complex-mode builds, this is a list of complex numbers instead.
        \return `True` on success. `False` (with a logged error) if setup is already constructed, `numberOfQubits` is invalid, or `indexes`/`coeffs` are empty or mismatched in length.
        \warning Indices here are **0-indexed**, bound directly to the C++ method with no adjustment — unlike storeOperators(), there is no +1/-1 conversion. `indexes[i]` is the raw basis-state integer (e.g. \f$|100\rangle\f$ is index `4`, not `9` as in the C interface's 1-indexed-plus-1 convention).
        \note If the resulting state is not normalised (within \f$10^{-14}\f$), a warning is logged but the call still succeeds.
        """
        ...

    def storeHamiltonian(self, iIndexes: list[int], jIndexes: list[int], Coeffs: list[float]) -> bool:
        """!
        \brief Store the Hamiltonian as a sparse matrix given by parallel index/coefficient arrays.
        \param iIndexes Row indices of non-zero Hamiltonian entries.
        \param jIndexes Column indices of non-zero Hamiltonian entries. Must match `iIndexes` in length.
        \param Coeffs Values of the non-zero Hamiltonian entries. Must match `iIndexes` in length.
        \return `True` on success. `False` (with a logged error) if setup is already constructed, or if the arrays are empty or mismatched in length.
        \warning Indices here are **0-indexed**, bound directly to the C++ method with no adjustment — unlike the C/Fortran interface, which subtracts 1 from Fortran-style 1-indexed input before calling into C++, this Python binding expects raw 0-indexed values directly.
        \note If this is not called, the Hamiltonian is instead loaded from the binary integral files at the run path (see storeRunPath()) during construction.
        """
        ...

    def storeNuclearEnergy(self, nuclearEnergy: float) -> bool:
        """!
        \brief Store the nuclear repulsion energy to be added to computed expectation values.
        \param nuclearEnergy The nuclear energy. Defaults to 0 if never called.
        \return `True` on success. `False` (with a logged error) if setup is already constructed.
        """
        ...

    def storeParameterDependencies(self, parameterDependency: list[tuple[int, float]]) -> bool:
        """!
        \brief Store the mapping from free (unique) parameters to per-operator angles.

        `parameterDependency[i][0]` gives the free parameter that operator `i`'s angle depends
        on, and `parameterDependency[i][1]` is the scale factor applied to that free parameter
        to obtain operator `i`'s angle. This is the same information as the "Order file" — see
        the \ref manual_page "manual" for the file-format description.

        \param parameterDependency One entry per operator, in the same order as the excitations given to storeOperators(). Must be non-empty.
        \return `True` on success. `False` (with a logged error) if setup is already constructed, `parameterDependency` is empty, or the implied number of unique parameters exceeds the number of entries given.
        """
        ...

    def storeRunPath(self, runPath: str) -> bool:
        """!
        \brief Store the path used to locate/save resources (e.g. Hamiltonian integral files, output files).
        \param runPath Path prefix, in the same style as the `filepath` command-line option of `cppAnsatzSynth` (see the \ref manual_page "manual").
        \return `True` on success. `False` (with a logged error) if setup is already constructed, or if `runPath` is empty.
        """
        ...

    def setAngles(self, angles: list[float]) -> bool:
        """!
        \brief Set the current angles for the evolution of the ansatz.
        \param angles The angles to set. Either compressed or decompressed format (auto-detected by length).
        \return `True` on success. `False` (with a logged error) if something went wrong.
        \note This causes an evolution of the ansatz — use it only when needed. Calling it again with the same angles is a safe no-op.
        """
        ...

    def getAngles(self) -> numpy.ndarray:
        """!
        \brief Get the current angles used for the latest evolution of the ansatz, always in **compressed** format.
        \return The angles in compressed format, or an empty array if something went wrong.
        """
        ...

    @overload
    def getExpectationValue(self) -> float:
        """!
        \brief Get the current energy using the latest evolution of the ansatz, \f$\braket{\psi|H|\psi}\f$.
        \return The expectation value.
        \warning The underlying success flag is discarded — this always returns a value, which may be meaningless if the manager was not correctly set up. Check the log for errors.
        """
        ...

    @overload
    def getExpectationValue(self, angles: list[float]) -> float:
        """!
        \brief Set the angles and get the current energy in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The expectation value.
        \warning The underlying success flag is discarded — this always returns a value, which may be meaningless if the call failed (e.g. wrong angle count). Check the log for errors.
        """
        ...

    @overload
    def getFinalState(self) -> numpy.ndarray:
        """!
        \brief Get the current evolved statevector, \f$\psi\f$, in dense format.
        \return The dense statevector.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getFinalState(self, angles: list[float]) -> numpy.ndarray:
        """!
        \brief Set the angles and get the dense final statevector in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The dense statevector.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getGradient(self) -> numpy.ndarray:
        """!
        \brief Get the gradient of the current energy, \f$\frac{\partial \braket{\psi|H|\psi}}{\partial \theta_i}\f$, one entry per **operator** (decompressed format).
        \return The gradient vector.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getGradient(self, angles: list[float]) -> numpy.ndarray:
        """!
        \brief Set the angles and get the decompressed-format gradient in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The gradient vector, one entry per operator (decompressed format).
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getGradientComp(self) -> numpy.ndarray:
        """!
        \brief Get the gradient of the current energy, one entry per **unique parameter** (compressed format).
        \return The gradient vector.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getGradientComp(self, angles: list[float]) -> numpy.ndarray:
        """!
        \brief Set the angles and get the compressed-format gradient in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The gradient vector, one entry per unique parameter (compressed format).
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getHessian(self) -> numpy.ndarray:
        """!
        \brief <strong>Do not use</strong> use getHessianComp() instead. Get the Hessian of the current energy, \f$\frac{\partial^2 \braket{\psi|H|\psi}}{\partial \theta_i \partial \theta_j}\f$, one entry per **operator** (decompressed format).
        \return The Hessian matrix.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        \note This will crash unless the library was compiled without `USE_FUSED`.
        """
        ...

    @overload
    def getHessian(self, angles: list[float]) -> numpy.ndarray:
        """!
        \brief <strong>Do not use</strong> use getHessianComp() instead. Set the angles and get the decompressed-format Hessian in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The Hessian matrix, one entry per operator (decompressed format).
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        \note This will crash unless the library was compiled without `USE_FUSED`.
        """
        ...

    @overload
    def getHessianComp(self) -> numpy.ndarray:
        """!
        \brief Get the Hessian of the current energy, one entry per **unique parameter** (compressed format).
        \return The Hessian matrix.
        \warning The underlying success flag is discarded - this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    @overload
    def getHessianComp(self, angles: list[float]) -> numpy.ndarray:
        """!
        \brief Set the angles and get the compressed-format Hessian in one call.
        \param angles The angles to evolve to. Either compressed or decompressed format (auto-detected by length).
        \return The Hessian matrix, one entry per unique parameter (compressed format).
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed. Check the log for errors.
        """
        ...

    def getExpectationValues(self, angles: numpy.ndarray) -> numpy.ndarray:
        """!
        \brief Compute the energy at multiple sets of angles.
        \param angles A 2D array where each *row* is one set of angles.
        \return The energies at each set of angles, one entry per row of `angles`.
        \warning The underlying success flag is discarded — this always returns a value, which may be empty/meaningless if the call failed (e.g. wrong number of angle columns). Check the log for errors.
        """
        ...

    def optimise(self) -> bool:
        """!
        \brief Optimise the configured ansatz to a minimum, using the Newton-Raphson method with backtracking.
        \return `True` on success. `False` (with a logged error) if something went wrong.
        \note Results can be read out afterwards using getExpectationValue() and getAngles().
        """
        ...


def loadParameters(
    orderFilePath: str,
    parameterFilepath: str,
    templatePath: list[tuple[int, float]],
) -> tuple[list[list[tuple[int, float]]], list[tuple[int, float]], int]:
    """!
    \brief Load parameter dependencies and one or more rotation paths from a parameter/order file pair.
    \param orderFilePath Path to the "Order file" (see the \ref manual_page "manual").
    \param parameterFilepath Path to the "Parameter file" (see the \ref manual_page "manual").
    \param templatePath A rotation path providing the operator ordering/identity (e.g. as returned by loadPath()).
    \return A tuple of `(rotationPaths, order, numberOfUniqueParameters)`.
    """
    ...


def loadPath(filepath: str) -> list[tuple[int, float]]:
    """!
    \brief Load a single rotation path (operator index / angle pairs) from an "Operators file".
    \param filepath Path to the operators file (see the \ref manual_page "manual").
    \return The loaded rotation path.
    """
    ...


def loadNuclearEnergy(filepath: str) -> float:
    """!
    \brief Load the nuclear repulsion energy from a "Nuclear Energy file", if present.
    \param filepath Run-path prefix (see the \ref manual_page "manual"); the `_Nuclear_Energy.dat` suffix is appended internally.
    \return The nuclear energy, or `0` if the file is not present.
    """
    ...


def readCsvState(filepath: str) -> list[float]:
    """!
    \brief Load a dense statevector from an "Initial State file".
    \param filepath Path to the initial-state CSV file (see the \ref manual_page "manual").
    \return The loaded coefficients.
    """
    ...


def loadOperators(filepath: str) -> list[list[int]]:
    """!
    \brief Load excitation operators from an "Operators file".
    \param filepath Path to the operators file (see the \ref manual_page "manual").
    \return A list of 4-element `[create, create, destroy, destroy]` qubit-index lists, one per excitation.
    \note Indices here are **1-indexed** on return (matching the Fortran/C convention) — 1 is added back to each internally stored index before returning to Python. This can be ...ed directly to storeOperators()
    """
    ...
