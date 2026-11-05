/*
 * Test the interface of the fusedEvolve class. There are two methods for computing gradients there, that can be compared against one another.
 * Test that it can evolve a wavefunction correctly.
 * Test that it does the same thing in complex mode. An easy cheat is to multiply the wavefunction by i at the start and then divide it at the end.
 * Test that the fusing of gates works correctly, need to produce the same results with fused and non-fused evolutions.
 * Check that gradients + Hessians are computed correctly. Can just compute the answer once and regression check that it doesnt change.
 *
 *  Specifically the following functions should be tested
 *  void updateExc(const std::vector<stateRotate::exc>& excs);
 *  void evolve(vector<numType>& dest, const std::vector<realNumType>& angles, vector<numType>* specifiedStart  = nullptr);
 *  void evolveMultiple(Matrix<numType>& dest, const Matrix<realNumType>::EigenMatrix& angles, vector<numType>* specifiedStart  = nullptr);
 *  //finalVector is the result of an evolve operation. Since we dont cache the angles this is up to the user to provide correctly.
 *  void evolveDerivative(const vector<numType>& finalVector,vector<realNumType>& deriv,const std::vector<realNumType>& angles, realNumType* Energy = nullptr);
 *  void evolveHessian(Eigen::MatrixXd& Hessian,vector<realNumType>& derivCompressed,const std::vector<realNumType>& angles, Eigen::Matrix<numType,-1,-1>* Ts = nullptr, realNumType* Energy = nullptr);
 *  realNumType getEnergy(const vector<numType> &psi);
 *  vector<realNumType> getEnergies(const Matrix<numType> &psi);
 *  const vector<numType> &getStart() const { return m_start; }
 *
 * The ansatz should be small. A couple difficult rotations is enough:
 * E.g. |01> Initial state and E_0^2-E^0_2 and E_1^2-E^1_2 This checks fermionic signs
 * Should also check two commuting rotations produce the right answer:
 * E.g. |02> Initial state and E_0^1-E^0_1 and E_2^3-E^2_3
 * These are simple enough that for nice angles analytic derivs and hessians are possible.
 *
 * Make sure the tests are deterministic
 */
#include "TUPSLoadingUtils.h"
#include "TestUtil.hpp"
#include "fusedevolve.h"
#include "logger.h"

#include <cmath>
#include <limits>

static const std::string H4L1Path = ANSATZEVOLVE_RESOURCE_DIR "/H4/L1/H4";
static const std::string H4L5Path = ANSATZEVOLVE_RESOURCE_DIR "/H4/L5/H4";
constexpr int numberOfQubits = 8;
constexpr uint32_t H4HFState = 85; // 01010101, two spin up and two spin down electrons
constexpr realNumType tol = 1e-10;

typedef std::vector<numType> refState;
typedef Eigen::SparseMatrix<realNumType, Eigen::RowMajor> parameterMatrix;
typedef HamiltonianMatrix<realNumType, numType> hamiltonian;

/*
 * Reference implementation: dense state vectors, textbook Jordan-Wigner signs and one gate at a time, i.e. no fusing.
 * Each excitation is the generator G = A - A^dagger with
 *   single {p,q}:     A = a^dagger_p a_q
 *   double {i,j,k,l}: A = a^dagger_i a^dagger_j a_l a_k
 * applied in order as exp(-theta G) = 1 - sin(theta) G + (1 - cos(theta)) G^2, which holds since G^3 = -G.
 * Derivatives of a gate are then exact: -cos(theta) G + sin(theta) G^2 and sin(theta) G + cos(theta) G^2.
 */
struct ladderOp
{
    int orbital;
    bool create;
};

// Applies a ladder operator to a basis state. Returns false if the state is destroyed
static bool applyLadder(uint32_t &state, int &sign, ladderOp op)
{
    uint32_t bit = 1u << op.orbital;
    if (bool(state & bit) == op.create)
        return false;
    if (__builtin_popcount(state & (bit - 1)) & 1)
        sign = -sign;
    state ^= bit;
    return true;
}

static refState applyGenerator(const stateRotate::exc &e, const refState &psi)
{
    // Written left to right, applied right to left
    std::vector<ladderOp> A;
    if (e.isSingleExc())
        A = {{e[0], true}, {e[1], false}};
    else
        A = {{e[0], true}, {e[1], true}, {e[3], false}, {e[2], false}};
    std::vector<ladderOp> ADagger;
    for (auto op = A.rbegin(); op != A.rend(); op++)
        ADagger.push_back({op->orbital, !op->create});

    refState ret(psi.size(), 0);
    for (uint32_t basisState = 0; basisState < psi.size(); basisState++)
    {
        for (auto [product, sign] : {std::make_pair(&A, 1), std::make_pair(&ADagger, -1)})
        {
            uint32_t state = basisState;
            bool alive = true;
            for (auto op = product->rbegin(); op != product->rend() && alive; op++)
                alive = applyLadder(state, sign, *op);
            if (alive)
                ret[state] += realNumType(sign) * psi[basisState];
        }
    }
    return ret;
}

// The derivative'th derivative of exp(-theta G) applied to psi
static refState applyRotation(const stateRotate::exc &e, realNumType theta, const refState &psi, int derivative)
{
    refState G = applyGenerator(e, psi);
    refState GG = applyGenerator(e, G);
    realNumType s = std::sin(theta);
    realNumType c = std::cos(theta);
    const realNumType coeffs[3][3] = {{1, -s, 1 - c}, {0, -c, s}, {0, s, c}};
    const realNumType *coeff = coeffs[derivative];
    refState ret(psi.size());
    for (size_t i = 0; i < psi.size(); i++)
        ret[i] = coeff[0] * psi[i] + coeff[1] * G[i] + coeff[2] * GG[i];
    return ret;
}

// Evolves start through every gate, differentiating with respect to angles derivA and derivB if given
static refState refEvolve(const std::vector<stateRotate::exc> &excs, const std::vector<realNumType> &angles, const refState &start,
                          long derivA = -1, long derivB = -1)
{
    refState psi = start;
    for (long i = 0; i < (long)excs.size(); i++)
        psi = applyRotation(excs[i], angles[i], psi, (i == derivA) + (i == derivB));
    return psi;
}

// Re <bra|H|ket>
static realNumType expectation(const hamiltonian &H, const refState &bra, const refState &ket)
{
    vector<numType> ketVec(ket);
    vector<numType> HKet;
    H.apply(ketVec, HKet);
    realNumType ret = 0;
    for (size_t i = 0; i < bra.size(); i++)
        ret += std::real(std::conj(bra[i]) * HKet[i]);
    return ret;
}

struct refResult
{
    refState psi;
    std::vector<refState> tangents; // d psi / d angle_i
    realNumType energy;
    Eigen::VectorXd gradient; // With respect to the compressed parameters, as is the Hessian
    Eigen::MatrixXd hessian;
};

static refResult reference(const hamiltonian &H, const std::vector<stateRotate::exc> &excs, const std::vector<realNumType> &angles,
                           const refState &start, const parameterMatrix &compress, bool calcHessian)
{
    long n = excs.size();
    refResult ret;
    ret.psi = refEvolve(excs, angles, start);
    ret.energy = expectation(H, ret.psi, ret.psi);
    Eigen::VectorXd gradient(n);
    for (long i = 0; i < n; i++)
    {
        ret.tangents.push_back(refEvolve(excs, angles, start, i));
        gradient[i] = 2 * expectation(H, ret.psi, ret.tangents[i]);
    }
    ret.gradient = compress * gradient;
    if (calcHessian)
    {
        Eigen::MatrixXd hessian(n, n);
        for (long i = 0; i < n; i++)
            for (long j = i; j < n; j++)
                hessian(i, j) = hessian(j, i) =
                    2 * (expectation(H, ret.tangents[i], ret.tangents[j]) + expectation(H, ret.psi, refEvolve(excs, angles, start, i, j)));
        ret.hessian = compress * hessian * compress.transpose();
    }
    return ret;
}

// Copies a state out of FusedEvolve, decompressing it if needed
static refState toRefState(const numType *data, size_t size, std::shared_ptr<compressor> comp)
{
    if (!comp)
        return refState(data, data + size);
    refState ret(1 << numberOfQubits, 0);
    for (uint32_t i = 0; i < size; i++)
    {
        uint32_t fullIdx;
        comp->deCompressIndex(i, fullIdx);
        ret[fullIdx] = data[i];
    }
    return ret;
}

static realNumType maxDifference(const refState &a, const refState &b)
{
    if (a.size() != b.size())
        return std::numeric_limits<realNumType>::infinity();
    realNumType ret = 0;
    for (size_t i = 0; i < a.size(); i++)
        ret = std::max<realNumType>(ret, std::abs(a[i] - b[i]));
    return ret;
}

static realNumType maxDifference(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
        return std::numeric_limits<realNumType>::infinity();
    return (a - b).cwiseAbs().maxCoeff();
}

static stateRotate::exc makeExc(int8_t a, int8_t b, int8_t c = -1, int8_t d = -1)
{
    int8_t arr[4] = {a, b, c, d};
    return stateRotate::exc(arr);
}

static parameterMatrix identityParameters(long n)
{
    parameterMatrix ret(n, n);
    ret.setIdentity();
    return ret;
}

struct testSystem
{
    std::string name;
    uint32_t startState;
    std::vector<stateRotate::exc> excs;
    parameterMatrix deCompress; // D_{i mu} = d angle_i / d parameter_mu
    std::vector<std::vector<realNumType>> angleSets; // One angle per excitation
    bool checkHessian; // The reference Hessian is too slow for large ansatze
};

static bool loadH4System(const std::string &path, const std::string &name, bool checkHessian, testSystem &s)
{
    s.name = name;
    s.startState = H4HFState;
    s.checkHessian = checkHessian;
    std::vector<ansatz::rotationElement> rotationPath;
    std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
    std::vector<std::pair<int, realNumType>> orders;
    int numberOfUniqueParameters;
    if (!stateRotate::loadOperators(path + "_Operators.dat", s.excs) || !loadPath(nullptr, path + "_Operators.dat", rotationPath) ||
        !loadParameters(path + "_Order.dat", path + "_Parameters.dat", rotationPath, rotationPaths, orders, numberOfUniqueParameters))
        return false;

    int numberOfParameters = 0;
    for (auto &o : orders)
        numberOfParameters = std::max(numberOfParameters, o.first + 1);
    s.deCompress.resize(orders.size(), numberOfParameters);
    for (size_t i = 0; i < orders.size(); i++)
        s.deCompress.insert(i, orders[i].first) = orders[i].second;

    // Generic angles, tied together by the order file
    for (int set = 0; set < 2; set++)
    {
        Eigen::VectorXd parameters(numberOfParameters);
        for (int mu = 0; mu < numberOfParameters; mu++)
            parameters[mu] = (set + 1) * std::sin(1.0 + mu + 2.3 * set);
        Eigen::VectorXd angles = s.deCompress * parameters;
        s.angleSets.emplace_back(angles.data(), angles.data() + angles.size());
    }
    return true;
}

// The two circuits from the description above, worked out by hand. Bit i of a basis state is orbital i.
static int AnalyticStateTest(std::shared_ptr<hamiltonian> H)
{
    int ret = 0;
    realNumType t1 = 0.3;
    realNumType t2 = 0.7;
    realNumType c1 = std::cos(t1), s1 = std::sin(t1), c2 = std::cos(t2), s2 = std::sin(t2);
    struct analyticCase
    {
        std::string name;
        uint32_t start;
        std::vector<stateRotate::exc> excs;
        std::vector<std::pair<uint32_t, realNumType>> amplitudes;
    };
    std::vector<analyticCase> cases = {
        // |01> -> |12> hops over the occupied orbital 1 so picks up a fermionic sign, giving |02> and |12> opposite signs.
        // The two gates share orbital 2 so do not commute
        {"Fermionic sign", 0b011, {makeExc(2, 0), makeExc(2, 1)}, {{0b011, c1 * c2}, {0b101, -c1 * s2}, {0b110, s1}}},
        // Disjoint orbitals so they commute and are fused together
        {"Commuting", 0b0101, {makeExc(1, 0), makeExc(3, 2)}, {{0b0101, c1 * c2}, {0b0110, -s1 * c2}, {0b1001, -c1 * s2}, {0b1010, s1 * s2}}},
    };
    for (auto &c : cases)
    {
        refState start(1 << numberOfQubits, 0);
        start[c.start] = 1;
        refState expected(1 << numberOfQubits, 0);
        for (auto [idx, amplitude] : c.amplitudes)
            expected[idx] = amplitude;

        vector<numType> startVec(start);
        FusedEvolve FE(startVec, H, identityParameters(2), identityParameters(2));
        FE.updateExc(c.excs);
        vector<numType> psi;
        FE.evolve(psi, {t1, t2});
        ret += expectNear(0.0, maxDifference(expected, toRefState(&psi[0], psi.size(), nullptr)), tol, __PRETTY_FUNCTION__, c.name + ": evolved state wrong");
        // Everything else is checked against the reference so make sure it agrees
        ret += expectNear(0.0, maxDifference(expected, refEvolve(c.excs, {t1, t2}, start)), tol, __PRETTY_FUNCTION__, c.name + ": reference state wrong");
    }
    return ret;
}

// Checks everything FusedEvolve calculates against the reference. The start state is multiplied by phase.
static int ReferenceTest(const testSystem &s, const hamiltonian &refH, std::shared_ptr<hamiltonian> H, std::shared_ptr<compressor> comp, numType phase)
{
    int ret = 0;
    std::string name = s.name + (comp ? " compressed" : "") + (phase != numType(1) ? " times i" : "");
    refState refStart(1 << numberOfQubits, 0);
    refStart[s.startState] = phase;
    vector<numType> fullStart(refStart);
    vector<numType> start;
    if (comp)
        compressor::compressVector<numType>(fullStart, start, comp);
    else
        start.copy(fullStart);

    parameterMatrix compress = s.deCompress.transpose();
    FusedEvolve FE(start, H, compress, s.deCompress);
    FE.updateExc(s.excs);
    ret += expectNear(0.0, maxDifference(refStart, toRefState(&FE.getStart()[0], FE.getStart().size(), comp)), tol, __PRETTY_FUNCTION__, name + ": getStart wrong");

    for (size_t set = 0; set < s.angleSets.size(); set++)
    {
        const std::vector<realNumType> &angles = s.angleSets[set];
        refResult ref = reference(refH, s.excs, angles, refStart, compress, s.checkHessian);
        std::string where = name + " angle set " + std::to_string(set);

        vector<numType> psi;
        FE.evolve(psi, angles);
        ret += expectNear(0.0, maxDifference(ref.psi, toRefState(&psi[0], psi.size(), comp)), tol, __PRETTY_FUNCTION__, where + ": evolve differs from reference");
        ret += expectNear(ref.energy, FE.getEnergy(psi), tol, __PRETTY_FUNCTION__, where + ": getEnergy wrong");

        vector<numType> psiTwice;
        FE.evolve(psiTwice, angles, &psi);
        ret += expectNear(0.0, maxDifference(refEvolve(s.excs, angles, ref.psi), toRefState(&psiTwice[0], psiTwice.size(), comp)), tol,
                          __PRETTY_FUNCTION__, where + ": evolve from a given start differs from reference");

        vector<realNumType> deriv;
        realNumType energy = 0;
        FE.evolveDerivative(psi, deriv, angles, &energy);
        ret += expectNear(ref.energy, energy, tol, __PRETTY_FUNCTION__, where + ": evolveDerivative energy wrong");
        ret += expectNear(0.0, maxDifference(ref.gradient, Eigen::VectorXd(deriv)), tol, __PRETTY_FUNCTION__, where + ": evolveDerivative differs from reference");

        Eigen::MatrixXd hessian;
        vector<realNumType> hessianDeriv;
        Eigen::Matrix<numType, -1, -1> tangents;
        energy = 0;
        FE.evolveHessian(hessian, hessianDeriv, angles, &tangents, &energy);
        ret += expectNear(ref.energy, energy, tol, __PRETTY_FUNCTION__, where + ": evolveHessian energy wrong");
        ret += expectNear(0.0, maxDifference(ref.gradient, Eigen::VectorXd(hessianDeriv)), tol, __PRETTY_FUNCTION__, where + ": evolveHessian derivative differs from reference");
        if (s.checkHessian)
            ret += expectNear(0.0, maxDifference(ref.hessian, hessian), tol, __PRETTY_FUNCTION__, where + ": evolveHessian differs from reference");
        else
            ret += expectNear(0.0, maxDifference(hessian, hessian.transpose()), tol, __PRETTY_FUNCTION__, where + ": evolveHessian not symmetric");

        ret += expectEqual(s.excs.size(), size_t(tangents.cols()), __PRETTY_FUNCTION__, where + ": wrong number of tangents");
        realNumType tangentDifference = 0;
        for (long i = 0; i < std::min<long>(tangents.cols(), s.excs.size()); i++)
            tangentDifference = std::max(tangentDifference, maxDifference(ref.tangents[i], toRefState(&tangents(0, i), tangents.rows(), comp)));
        ret += expectNear(0.0, tangentDifference, tol, __PRETTY_FUNCTION__, where + ": evolveHessian tangents differ from reference");
    }

    // All angle sets at once
    Matrix<realNumType>::EigenMatrix anglesMatrix(s.angleSets.size(), s.excs.size());
    for (size_t set = 0; set < s.angleSets.size(); set++)
        for (size_t i = 0; i < s.excs.size(); i++)
            anglesMatrix(set, i) = s.angleSets[set][i];
    Matrix<numType> psis;
    FE.evolveMultiple(psis, anglesMatrix);
    vector<realNumType> energies = FE.getEnergies(psis);
    for (size_t set = 0; set < s.angleSets.size(); set++)
    {
        std::string where = name + " angle set " + std::to_string(set);
        refState expected = refEvolve(s.excs, s.angleSets[set], refStart);
        ret += expectNear(0.0, maxDifference(expected, toRefState(&psis.at(set, 0), psis.m_jSize, comp)), tol, __PRETTY_FUNCTION__, where + ": evolveMultiple differs from reference");
        ret += expectNear(expectation(refH, expected, expected), energies[set], tol, __PRETTY_FUNCTION__, where + ": getEnergies wrong");
    }
    return ret;
}

// Changing the excitations after construction must rebuild the fused gates
static int UpdateExcTest(const testSystem &s, std::shared_ptr<hamiltonian> H)
{
    int ret = 0;
    refState start(1 << numberOfQubits, 0);
    start[s.startState] = 1;
    vector<numType> startVec(start);
    FusedEvolve FE(startVec, H, s.deCompress.transpose(), s.deCompress);

    std::vector<stateRotate::exc> reversedExcs(s.excs.rbegin(), s.excs.rend());
    std::vector<realNumType> reversedAngles(s.angleSets[0].rbegin(), s.angleSets[0].rend());
    std::vector<std::pair<std::vector<stateRotate::exc>, std::vector<realNumType>>> steps = {
        {s.excs, s.angleSets[0]}, {reversedExcs, reversedAngles}, {s.excs, s.angleSets[0]}};
    for (size_t step = 0; step < steps.size(); step++)
    {
        auto &[excs, angles] = steps[step];
        FE.updateExc(excs);
        vector<numType> psi;
        FE.evolve(psi, angles);
        ret += expectNear(0.0, maxDifference(refEvolve(excs, angles, start), toRefState(&psi[0], psi.size(), nullptr)), tol, __PRETTY_FUNCTION__,
                          s.name + ": state wrong after updateExc step " + std::to_string(step));
    }
    return ret;
}

int FusedEvolveTests(int argc, char **argv)
{
    int ret = 0;
    auto H = std::make_shared<hamiltonian>(H4L1Path, numberOfQubits, nullptr);
    std::shared_ptr<compressor> comp = std::make_shared<SZAndnumberOperatorCompressor>(1 << numberOfQubits, 2, 2);
    auto compressedH = std::make_shared<hamiltonian>(H4L1Path, numberOfQubits, comp);

    std::vector<testSystem> toySystems = {
        {"Fermionic sign", 0b011, {makeExc(2, 0), makeExc(2, 1)}, identityParameters(2), {{0.3, 0.7}, {-1.2, 2.5}}, true},
        {"Commuting", 0b0101, {makeExc(1, 0), makeExc(3, 2)}, identityParameters(2), {{0.3, 0.7}, {-1.2, 2.5}}, true},
    };
    testSystem H4L1;
    testSystem H4L5;
    if (!loadH4System(H4L1Path, "H4 L1", true, H4L1) || !loadH4System(H4L5Path, "H4 L5", false, H4L5))
        return -1;

    std::vector<numType> phases = {1};
#ifdef useComplex
    phases.push_back(iu);
#endif

    ret += AnalyticStateTest(H);
    for (numType phase : phases)
    {
        for (auto &s : toySystems)
            ret += ReferenceTest(s, *H, H, nullptr, phase);
        for (auto *s : {&H4L1, &H4L5})
        {
            ret += ReferenceTest(*s, *H, H, nullptr, phase);
            ret += ReferenceTest(*s, *H, compressedH, comp, phase);
        }
    }
    ret += UpdateExcTest(H4L1, H);
    return ret > 0 ? -1 : 0;
}
