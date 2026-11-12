/*
 * Test loading of files and make sure their internal representations are correct.
 * This is not a great test as it requires knowledge of the internal representation
 *
 * The following functions need to be tested:
 * void LoadNuclearEnergy(realNumType &NuclearEnergy, std::string filePath);
 *
 * bool loadPath(std::shared_ptr<stateRotate> sr, std::string filePath,
 *            std::vector<ansatz::rotationElement>& rotationPath);

 * bool loadParameters(std::string filePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &order,
                    int& numberOfUniqueParameters);

 * bool loadParameters(const std::string& orderFilePath, const std::string& parameterFilePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &orders // the fixed relation between parameters
                    ,int& numberOfUniqueParameters);
 * stateRotate::loadOperators(std::string filePath, std::vector<stateRotate::exc>& excs);
 *
 * Example data is already present and used in EndtoEndTests.cpp - use that.
 * Inspect the state of NuclearEnergy, rotationPath, rotationPaths, orders, excs, and make sure they were loaded correctly
 */
#include "TUPSLoadingUtils.h"
#include "TestUtil.hpp"
#include "logger.h"

static const std::string H4Path = ANSATZEVOLVE_RESOURCE_DIR "/H4/L1/H4";
static const std::string MissingPath = ANSATZEVOLVE_RESOURCE_DIR "/H4/L1/DoesNotExist";

// H4_Operators.dat, 0 indexed. -1 marks an unused slot of a single excitation
static const std::vector<std::vector<int>> H4Excs = {
    {0, 1, -1, -1}, {4, 5, -1, -1}, {0, 4, 1, 5},   {0, 1, -1, -1}, {4, 5, -1, -1}, {2, 3, -1, -1}, {6, 7, -1, -1}, {2, 6, 3, 7},
    {2, 3, -1, -1}, {6, 7, -1, -1}, {1, 2, -1, -1}, {5, 6, -1, -1}, {1, 5, 2, 6},   {1, 2, -1, -1}, {5, 6, -1, -1},
};

// H4_Order.dat, as {0 indexed parameter, scale factor}
static const std::vector<std::pair<int, realNumType>> H4Orders = {
    {0, 1}, {0, 1}, {1, 1}, {0, -1}, {0, -1}, {2, 1}, {2, 1}, {3, 1}, {2, -1}, {2, -1}, {4, 1}, {4, 1}, {5, 1}, {6, 1}, {6, 1},
};

// The angle of each operator for the zero path followed by the two minima in H4_Parameters.dat
static const std::vector<std::vector<realNumType>> H4Angles = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0.1, 0.1, 0.2, -0.1, -0.1, 0.3, 0.3, 0.4, -0.3, -0.3, 0.5, 0.5, 0.6, 0.7, 0.7},
    {-3.14159265358979, -3.14159265358979, 0.296442558183625, 3.14159265358979, 3.14159265358979, -7.31564874620643e-20, -7.31564874620643e-20,
     0.536266956045058, 7.31564874620643e-20, 7.31564874620643e-20, 0.907191125929521, 0.907191125929521, 3.43173936258855, 0.602580862545367,
     0.602580862545367},
};

static std::vector<int> toVector(const stateRotate::exc &e)
{
    return {e[0], e[1], e[2], e[3]};
}

static int LoadNuclearEnergyTest()
{
    int ret = 0;
    realNumType NuclearEnergy = 0;
    LoadNuclearEnergy(NuclearEnergy, H4Path);
    ret += expectNear(1.14655062366, NuclearEnergy, __PRETTY_FUNCTION__, "H4 nuclear energy wrong");

    // A missing file is reported and gives zero
    NuclearEnergy = 1;
    LoadNuclearEnergy(NuclearEnergy, MissingPath);
    ret += expectNear(0.0, NuclearEnergy, __PRETTY_FUNCTION__, "Missing nuclear energy file should give zero");
    return ret;
}

static int LoadOperatorsTest()
{
    int ret = 0;
    std::vector<stateRotate::exc> excs;
    ret += expectEqual(true, stateRotate::loadOperators(H4Path + "_Operators.dat", excs), __PRETTY_FUNCTION__, "Could not load operators");
    ret += expectEqual(H4Excs.size(), excs.size(), __PRETTY_FUNCTION__, "Wrong number of operators");
    for (size_t i = 0; i < std::min(H4Excs.size(), excs.size()); i++)
        ret += expectEqual(H4Excs[i], toVector(excs[i]), __PRETTY_FUNCTION__, "Operator " + std::to_string(i) + " wrong");

    ret += expectEqual(false, stateRotate::loadOperators(MissingPath, excs), __PRETTY_FUNCTION__, "Missing operator file should fail");
    return ret;
}

static int LoadPathTest()
{
    int ret = 0;
    // Without a stateRotate only the length of the path is known
    std::vector<ansatz::rotationElement> rotationPath;
    ret += expectEqual(true, loadPath(nullptr, H4Path + "_Operators.dat", rotationPath), __PRETTY_FUNCTION__, "Could not load path");
    ret += expectEqual(H4Excs.size(), rotationPath.size(), __PRETTY_FUNCTION__, "Wrong path length");
    for (size_t i = 0; i < rotationPath.size(); i++)
    {
        ret += expectEqual(size_t(0), rotationPath[i].first, __PRETTY_FUNCTION__, "Operator index should be zero without a stateRotate");
        ret += expectNear(0.0, rotationPath[i].second, __PRETTY_FUNCTION__, "Angles should start at zero");
    }

    // With a stateRotate each operator is stored as its lie algebra index. Repeated operators share an index
    auto sr = std::make_shared<stateRotate>(8);
    sr->loadOperators(H4Path + "_Operators.dat");
    ret += expectEqual(9, sr->m_lieOpDim, __PRETTY_FUNCTION__, "Wrong number of distinct operators");
    rotationPath.clear();
    ret += expectEqual(true, loadPath(sr, H4Path + "_Operators.dat", rotationPath), __PRETTY_FUNCTION__, "Could not load path");
    ret += expectEqual(H4Excs.size(), rotationPath.size(), __PRETTY_FUNCTION__, "Wrong path length");
    for (size_t i = 0; i < std::min(H4Excs.size(), rotationPath.size()); i++)
    {
        stateRotate::exc e;
        sr->convertIdxToExc(rotationPath[i].first, e);
        ret += expectEqual(H4Excs[i], toVector(e), __PRETTY_FUNCTION__, "Path element " + std::to_string(i) + " has the wrong operator");
    }

    ret += expectEqual(false, loadPath(nullptr, MissingPath, rotationPath), __PRETTY_FUNCTION__, "Missing operator file should fail");
    return ret;
}

static int checkParameters(
    const std::vector<ansatz::rotationElement> &rotationPath,
    const std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
    const std::vector<std::pair<int, realNumType>> &orders,
    int numberOfUniqueParameters
)
{
    int ret = 0;
    ret += expectEqual(H4Orders.size(), orders.size(), __PRETTY_FUNCTION__, "Wrong number of orders");
    for (size_t i = 0; i < std::min(H4Orders.size(), orders.size()); i++)
    {
        ret += expectEqual(H4Orders[i].first, orders[i].first, __PRETTY_FUNCTION__, "Order " + std::to_string(i) + " has the wrong parameter");
        ret += expectNear(H4Orders[i].second, orders[i].second, __PRETTY_FUNCTION__, "Order " + std::to_string(i) + " has the wrong scale factor");
    }
    // The parameter file has 9 parameters per minimum even though only 7 are used by the order file
    ret += expectEqual(9, numberOfUniqueParameters, __PRETTY_FUNCTION__, "Wrong number of unique parameters");

    ret += expectEqual(H4Angles.size(), rotationPaths.size(), __PRETTY_FUNCTION__, "Wrong number of paths");
    for (size_t p = 0; p < std::min(H4Angles.size(), rotationPaths.size()); p++)
    {
        ret += expectEqual(rotationPath.size(), rotationPaths[p].size(), __PRETTY_FUNCTION__, "Path " + std::to_string(p) + " has the wrong length");
        for (size_t i = 0; i < std::min(rotationPath.size(), rotationPaths[p].size()); i++)
        {
            std::string where = "Path " + std::to_string(p) + " element " + std::to_string(i);
            ret += expectEqual(rotationPath[i].first, rotationPaths[p][i].first, __PRETTY_FUNCTION__, where + " has the wrong operator");
            ret += expectNear(H4Angles[p][i], rotationPaths[p][i].second, __PRETTY_FUNCTION__, where + " has the wrong angle");
        }
    }
    return ret;
}

static int LoadParametersTest()
{
    int ret = 0;
    std::vector<ansatz::rotationElement> rotationPath;
    if (!loadPath(nullptr, H4Path + "_Operators.dat", rotationPath)) return 1;

    {
        std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
        std::vector<std::pair<int, realNumType>> orders;
        int numberOfUniqueParameters = -1;
        bool success = loadParameters(H4Path + "_Order.dat", H4Path + "_Parameters.dat", rotationPath, rotationPaths, orders, numberOfUniqueParameters);
        ret += expectEqual(true, success, __PRETTY_FUNCTION__, "Could not load parameters");
        ret += checkParameters(rotationPath, rotationPaths, orders, numberOfUniqueParameters);
    }
    {
        // Same files found from the shared prefix
        std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
        std::vector<std::pair<int, realNumType>> orders;
        int numberOfUniqueParameters = -1;
        bool success = loadParameters(H4Path, rotationPath, rotationPaths, orders, numberOfUniqueParameters);
        ret += expectEqual(true, success, __PRETTY_FUNCTION__, "Could not load parameters from prefix");
        ret += checkParameters(rotationPath, rotationPaths, orders, numberOfUniqueParameters);
    }
    {
        // Without a parameter file the orders still load and only the zero path is made
        std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
        std::vector<std::pair<int, realNumType>> orders;
        int numberOfUniqueParameters = -1;
        bool success = loadParameters(H4Path + "_Order.dat", MissingPath, rotationPath, rotationPaths, orders, numberOfUniqueParameters);
        ret += expectEqual(true, success, __PRETTY_FUNCTION__, "Missing parameter file should not fail");
        ret += expectEqual(H4Orders.size(), orders.size(), __PRETTY_FUNCTION__, "Wrong number of orders without parameter file");
        ret += expectEqual(7, numberOfUniqueParameters, __PRETTY_FUNCTION__, "Unique parameters should come from the order file");
        ret += expectEqual(size_t(1), rotationPaths.size(), __PRETTY_FUNCTION__, "Only the zero path should be made");
    }
    {
        std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
        std::vector<std::pair<int, realNumType>> orders;
        int numberOfUniqueParameters = -1;
        bool success = loadParameters(MissingPath, H4Path + "_Parameters.dat", rotationPath, rotationPaths, orders, numberOfUniqueParameters);
        ret += expectEqual(false, success, __PRETTY_FUNCTION__, "Missing order file should fail");
    }
    // The path must have exactly one operator per order entry
    for (size_t pathLength : {rotationPath.size() - 1, rotationPath.size() + 1})
    {
        std::vector<ansatz::rotationElement> wrongPath(pathLength, {0, 0});
        std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
        std::vector<std::pair<int, realNumType>> orders;
        int numberOfUniqueParameters = -1;
        bool success = loadParameters(H4Path, wrongPath, rotationPaths, orders, numberOfUniqueParameters);
        ret += expectEqual(false, success, __PRETTY_FUNCTION__, "Path length " + std::to_string(pathLength) + " should not match the order file");
    }
    return ret;
}

int LoadFilesTests(int argc, char **argv)
{
    int ret = 0;
    ret += LoadNuclearEnergyTest();
    ret += LoadOperatorsTest();
    ret += LoadPathTest();
    ret += LoadParametersTest();
    return ret > 0 ? -1 : 0;
}
