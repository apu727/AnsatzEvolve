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

#define USE_FUSED

#ifdef USE_FUSED
constexpr bool useFused = true;
#else
constexpr bool useFused = false;
#endif
/**
 * \brief Manages the full lifecycle of a single ansatz: state setup, evolution, and property evaluation.
 *
 * ## Usage
 * A `stateAnsatzManager` is configured by calling the `store*` functions (in any order) to
 * provide the initial state, operators (excitations), Hamiltonian, parameter dependencies,
 * nuclear energy, and run path. None of these trigger any computation on their own — they
 * just record the data.
 *
 * The manager is lazily constructed: the first call to any evolution-triggering function
 * (`setAngles`, any `get*` function, `optimise`, `generatePathsForSubspace`) checks whether
 * setup is complete and, if so, builds the ansatz automatically. If required data is missing
 * or inconsistent (e.g. mismatched sizes between operators and parameter dependencies), that
 * call returns `false` and an error describing the problem is written via `logger()`. No
 * exceptions are thrown.
 *
 * Once construction has succeeded, calling any `store*` function again will fail (setup
 * cannot be modified after construction) — build a new `stateAnsatzManager` instead if the
 * problem needs to change.
 *
 * ## Angle formats
 * Angles can be provided/retrieved in two equivalent representations:
 * - **Decompressed**: one angle per operator/excitation (length equal to the number of
 *   operators provided via storeOperators()).
 * - **Compressed**: one angle per *unique* free parameter (length equal to the number of
 *   unique parameters implied by storeParameterDependencies()), where multiple operators may
 *   share or be scaled from the same free parameter.
 *
 * Functions taking a plain `std::vector<realNumType>` of angles (e.g. setAngles(), the
 * angle-taking overloads of getExpectationValue()/getFinalState()/getGradient()/getHessian())
 * accept either length and auto-detect which was given. Functions with a `Comp` suffix
 * (getGradientComp(), getHessianComp()) always return in the compressed representation.
 *
 * ## Thread safety
 * m_interfaceLock is provided for external interfaces (e.g. the C/Fortran bindings) to
 * serialize access from multiple callers; it is not acquired internally by this class, so
 * callers sharing a `stateAnsatzManager` across threads are responsible for locking around
 * calls themselves. This is done automatically by the C interface. 
 */
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
    /**
     * \brief Lock for external interfaces (e.g. the C/Fortran bindings) to guarantee thread-safe
     * operation. Generally only one member function should be in progress at a time.
     * \note Not acquired internally by this class — it exists purely as a convention for external
     * callers to use around their own calls into this manager.
     */
    std::mutex m_interfaceLock;

    /// Constructs an empty, unconfigured manager. The run path defaults to the current working directory (see storeRunPath()).
    stateAnsatzManager();
    ~stateAnsatzManager();

    /**
     * \brief Store the excitation operators that define the ansatz.
     * \param excs The list of excitations (single/double) to use. Must be non-empty.
     * \return `true` on success. `false` (with a logged error) if something went wrong
     */
    bool storeOperators(const std::vector<stateRotate::exc> &excs);

    /**
     * \brief Store the initial statevector as a sparse list of (index, coefficient) pairs.
     * \param numberOfQubits Number of qubits; must be at least 1. Determines the full statevector length (\f$2^{numberOfQubits}\f$).
     * \param indexes Basis-state indexes with non-zero amplitude.
     * \param coeffs Amplitudes corresponding to each entry in \p indexes. Must be the same size as \p indexes.
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     * \note If the resulting state is not normalised (within \f$10^{-14}\f$), a warning is logged but the call still succeeds.
     */
    bool storeInitial(int numberOfQubits, const std::vector<int>& indexes,const std::vector<numType>& coeffs);
    /**
     * \brief Store the Hamiltonian as a sparse matrix given by parallel index/coefficient arrays.
     * \param iIndexes Zero indexed row indexes of non-zero Hamiltonian entries.
     * \param jIndexes Zero indexed column indexes of non-zero Hamiltonian entries. Must match \p iIndexes in size.
     * \param Coeffs Values of the non-zero Hamiltonian entries. Must match \p iIndexes in size.
     * \return `true` on success. `false` (with a logged error) if something went wrong. 
     * \note If this is not called, the Hamiltonian is instead loaded from the binary integral files at the run path (see storeRunPath()) during construction.
     */
    bool storeHamiltonian(std::vector<int>&& iIndexes, std::vector<int>&& jIndexes, std::vector<realNumType>&& Coeffs);
    /**
     * \brief Store the nuclear repulsion energy to be added to computed expectation values.
     * \param nuclearEnergy The nuclear energy. Defaults to 0 if never called.
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     */
    bool storeNuclearEnergy(realNumType nuclearEnergy);
    /**
     * \brief Store the mapping from free (unique) parameters to per-operator angles.
     *
     * `parameterDependency[i].first` gives the free parameter that operator `i`'s angle depends
     * on, and `parameterDependency[i].second` is the scale factor applied to that free parameter
     * to obtain operator `i`'s angle. This is the same information as the "Order file" — see the
     * \subpage manual_page "manual" for the file-format description.
     *
     * \param parameterDependency One entry per operator, in the same order as the excitations given to storeOperators(). Must be non-empty.
     * \return `true` on success. `false` (with a logged error) if something went wrong. 
     */
    //Expresses the mapping from free parameters to actual angles. parameterDependency[0].first gives the free parameter than angle 0 depends on.
    //The scale factor (parameterDependency[0].second) scales the free parameter to make the angle
    bool storeParameterDependencies(const std::vector<std::pair<int,realNumType>>& parameterDependency);
    /**
     * \brief Store the path used to locate/save resources (e.g. Hamiltonian integral files, output files).
     * \param runPath Path prefix, in the same style as the `filepath` command-line option of `cppAnsatzSynth` (see the \subpage manual_page "manual").
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     */
    bool storeRunPath(const std::string& runPath);

    //external Usage functions
    /**
     * \brief Set the current angles for the evolution of the ansatz. 
     * These can either be in 'compressed' one angle per unique parameter or 'decompressed' one angle per operator format. 
     * \param angles the angles to set to. if `angles.size == numberOfDecompressedAngles` it is assumed to be decompressed 
     * otherwise it is assumed to be compressed.  
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     * 
     * \note This function causes an evolution of the ansatz, use it only when you need it.
     * \note Calling this function multiple times with the same exact angles will not cause another evolution and is safe.  
     */
    bool setAngles(std::vector<realNumType> angles);
    /**
     * \brief Get the current angles used for the latest evolution of the ansatz. 
     * These are always returned in the 'compressed' format, one angle per unique parameter.
     * 
     * \return The angles in compressed format, or a default constructed `Eigen::vector` if something went wrong. 
     * \note If you have provided decompressed angles that are inconsistent with the compression function, this will not return your original angles. 
     */
    vector<realNumType>::EigenVector getAngles();

    /**
     * \brief Get the current energy using the latest evolution of the ansatz with the supplied Hamiltonian. 
     * i.e. \f$\braket{\psi|H|\psi}\f$
     * \param exptValue the resultant expectation value. 
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     * \note If you have provided decompressed angles that are inconsistent with the compression function, this will not return your original angles. 
     */
    bool getExpectationValue(realNumType& exptValue);
    /**
     * \brief Get the current evolved statevector. This is in a dense format and so can be very large. Internally it is compressed.
     * i.e. \f$ \psi \f$
     * \param finalState the place to store the final state
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     */
    bool getFinalState(vector<numType>& finalState);
    /**
     * \brief Get the derivative of the current energy using the latest evolution of the ansatz with the supplied Hamiltonian. 
     * i.e. \f$\frac{\partial \braket{\psi|H|\psi}}{\partial \theta_i}\f$
     * \param gradient The resultant gradient.  There is one entry for each **operator** (decompressed format) in the ansatz
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     */
    bool getGradient(vector<realNumType>& gradient); // decompressed format
    /**
     * \brief Get the derivative of the current energy using the latest evolution of the ansatz with the supplied Hamiltonian. 
     * i.e. \f$\frac{\partial \braket{\psi|H|\psi}}{\partial \theta_i}\f$
     * \param gradient The resultant gradient. There is one entry for each **unique parameter** (compressed format) in the ansatz
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     */
    bool getGradientComp(vector<realNumType>& gradient); // compressed format
    /**
     * \brief **DO NOT USE, use getHessianComp()**
     *  Get the second derivative of the current energy using the latest evolution of the ansatz with the supplied Hamiltonian. 
     * i.e. \f$\frac{\partial^2 \braket{\psi|H|\psi}}{\partial \theta_i \partial \theta_j}\f$
     * \param hessian The resultant hessian.  There is one entry for each **operator** (decompressed format) in the ansatz
     * \return `true` on success. `false` (with a logged error) if something went wrong.
     * \note This function will crash unless compiled without USE_FUSED. This is currently untested. 
     */
    bool getHessian(Matrix<realNumType>::EigenMatrix& hessian);
    /**
     * \brief **DO NOT USE** Get the second derivative of the current energy using the latest evolution of the ansatz with the supplied Hamiltonian. 
     * i.e. \f$\frac{\partial^2 \braket{\psi|H|\psi}}{\partial \theta_i \partial \theta_j}\f$
     * \param hessian The resultant hessian.  There is one entry for each **unique parameter** (compressed format) in the ansatz
     * \return `true` on success. `false` (with a logged error) if something went wrong. 
     */
    bool getHessianComp(Matrix<realNumType>::EigenMatrix& hessian);
    //TODO get Hessian And Derivative in one

    //Allow providing the angle and update if needed
    /**
     * \brief Overload of getExpectationValue(). Allows setting the angles and getting the expectation value in one go.
     */
    bool getExpectationValue(const std::vector<realNumType>& angles, realNumType& exptValue);
    /**
     * \brief Overload of getFinalState(). Allows setting the angles and computing the final state in one go.
     */
    bool getFinalState(const std::vector<realNumType>& angles, vector<numType>& finalState);
    /**
     * \brief Overload of getGradient(). Allows setting the angles and getting the gradient in one go.
     */
    bool getGradient(const std::vector<realNumType>& angles, vector<realNumType>& gradient);
    /**
     * \brief Overload of getGradientComp(). Allows setting the angles and getting the gradient in one go.
     */
    bool getGradientComp(const std::vector<realNumType>& angles,vector<realNumType>& gradient); // compressed format
    /**
     * \brief Overload of getHessian(). Allows setting the angles and getting the hessian in one go.
     */
    bool getHessian(const std::vector<realNumType>& angles, Matrix<realNumType>::EigenMatrix& hessian);
    /**
     * \brief Overload of getHessian(). Allows setting the angles and getting the hessian in one go.
     */
    bool getHessianComp(const std::vector<realNumType>& angles, Matrix<realNumType>::EigenMatrix& hessian);

    //Compute on each set of angles
    /**
     * \brief compute expectation values at lots of sets of angles. Each *row* of the matrix is a set of angles. 
     */
    bool getExpectationValues(Matrix<realNumType>::EigenMatrix& angles, vector<realNumType>::EigenVector& exptValue);

    //Calculation functions
    /**
     * \brief Optimise the setup ansatz to a minimum. This used newton-raphson method with backtracking.
     *  Results can be read out using getExpectationValue() and getAngles()
     */
    bool optimise();
    // bool subspaceDiag(); // Cant ATM keep track of multiple rotation paths TODO
    // bool writeProperties(); // Cant ATM keep track of multiple rotation paths TODO
    /**
     * \brief UnTested. 
     */
    bool generatePathsForSubspace(size_t numberOfPaths);


};

#endif // ANSATZMANAGER_H
