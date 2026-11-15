#include "AnsatzManager.h"
#include "TUPSLoadingUtils.h"
#include "TestUtil.hpp"
#include "logger.h"

constexpr double HFEnergyH4 = -1.575616476701866;
constexpr double H4MinimaEnergy = -1.69179518058558;

template<bool complexInitialState>
int H4EndToEndTest()
{
    int ret = 0;
    std::string thisPath = ANSATZEVOLVE_RESOURCE_DIR "/H4/L1/H4";
    std::vector<stateRotate::exc> excs;
    stateRotate::loadOperators(thisPath + "_Operators.dat", excs);

    std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
    std::vector<std::pair<int, realNumType>> order;
    int numberOfUniqueParameters;
    std::vector<ansatz::rotationElement> rotationPath;
    if (!loadPath(nullptr, thisPath + "_Operators.dat", rotationPath)) return 1;
    if (!loadParameters(thisPath + "_Order.dat", thisPath + "_Parameters.dat", rotationPath, rotationPaths, order, numberOfUniqueParameters))
        return 1;

    realNumType NuclearEnergy;
    LoadNuclearEnergy(NuclearEnergy, thisPath);

    stateAnsatzManager sa;
    sa.storeRunPath(thisPath);

#ifdef useComplex
    if constexpr (complexInitialState)
        sa.storeInitial(8, {85}, {iu});
    else
        sa.storeInitial(8, {85}, {1.});
#else
    if constexpr (complexInitialState)
        static_assert(val_dependent_false_v<complexInitialState>, "Complex Initial state in non complex mode");
    else
        sa.storeInitial(8, {85}, {1.});
#endif

    sa.storeOperators(excs);
    sa.storeParameterDependencies(order);
    sa.storeNuclearEnergy(NuclearEnergy);

    // HF energy
    std::vector<realNumType> angles(rotationPath.size(), 0);
    sa.setAngles(angles);
    {
        realNumType HFEnergy;
        sa.getExpectationValue(HFEnergy);
        ret += expectNear(HFEnergyH4, HFEnergy, __PRETTY_FUNCTION__, "H4 HF energy wrong");
    }

    // Minima energy
    angles = {-3.14159265358979, 0.296442558183625, 0, 0.536266956045058, 0.907191125929521, 3.43173936258855, 0.602580862545367};
    sa.setAngles(angles);
    {
        realNumType energy;
        sa.getExpectationValue(energy);
        ret += expectNear(H4MinimaEnergy, energy, __PRETTY_FUNCTION__, "H4 Minima energy wrong");
    }

    // Test Opt
    for (auto &a : angles)
    {
        a += 0.1;
    }
    sa.setAngles(angles);
    {
        realNumType energy;
        sa.getExpectationValue(energy);
        ret += expectNotNear(H4MinimaEnergy, energy, __PRETTY_FUNCTION__, "Offset angles did not change the energy");
        sa.optimise();
        sa.getExpectationValue(energy);
        ret += expectNear(H4MinimaEnergy, energy, __PRETTY_FUNCTION__, "H4 Optimised Minima energy wrong");
    }

    return ret;
}

int EndtoEndTests(int argc, char **argv)
{
    logger().log("Test Dir is", ANSATZEVOLVE_RESOURCE_DIR);
    int ret = 0;
    ret += H4EndToEndTest<false>();
#ifdef useComplex
    ret += H4EndToEndTest<true>();
#endif
    return ret > 0 ? -1 : 0;
}