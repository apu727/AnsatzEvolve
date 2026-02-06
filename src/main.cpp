/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

#include "benchmark.h"
#include "hamiltonianmatrix.h"
#include "operatorpool.h"
#include "ansatz.h"
#include "threadpool.h"

#include "TUPSLoadingUtils.h"
#include "tupsquantities.h"
#include "logger.h"


#include <vector>

bool OperatorsHaveSZSymmetry(int numberOfQubits, const std::vector<stateRotate::exc>& excs)
{   //Ugly bodge. This has been copied from AnsatzManager.cpp. Ideally we should also use that but this breaks the build structure
    bool ret = true;
    uint32_t spinDownBitMask = (1<<(numberOfQubits/2))-1;
    uint32_t spinUpBitMask = ((1<<(numberOfQubits))-1) ^ spinDownBitMask;

    for (const auto& e : excs)
    {
        uint32_t create = 0;
        uint32_t destroy = 0;
        if (e.isSingleExc())
        {
            create = (1<<e.first);
            destroy = (1<<e.second);
        }
        else if (e.isDoubleExc())
        {
            create = (1<<e.first) | (1 << e.second);
            destroy = (1<<e.third) | (1 << e.fourth);
        }
        else
        {
            char msg[100] = {};
            sprintf(msg,"exc is neither single nor double excitation %i, %i, %i, %i\n",e.first, e.second, e.third, e.fourth);
            releaseAssert(false, msg);
        }
        int spinUpCreate = popcount(create & spinUpBitMask);
        int spinDownCreate = popcount(create & spinDownBitMask);

        int spinUpDestroy = popcount(destroy & spinUpBitMask);
        int spinDownDestroy = popcount(destroy & spinDownBitMask);
        ret = ret && (spinUpCreate == spinUpDestroy) && (spinDownCreate == spinDownDestroy);
    }
    if (excs.size() == 0)
        ret = false;
    return ret;
}

struct options
{
    bool optimise = false;
    bool iterativeOptimise = false;
    bool makeLie = false;
    bool subspaceDiag = false;
    bool writeProperties = false;
    bool generatePathsForSubspace = false;
    std::string filePath = "";
    bool ok = true;
    bool benchmark = false;
    bool noCompress = false;
    bool onlyEvolve = false;
    bool noLowestEigenValue = false;
    int numberOfPathsToLoad = -1;
    int numberOfOverlapsToCompute = 1;
    bool noHess = false;
    static void printHelp()
    {
        logger().log("Help:");
        logger().log("'optimise' ----------------- Do newton raphson steps starting at the first path in the parameter file");
        logger().log("'iterativeoptimise' -------- Do iterative newton raphson steps starting at the first path in the parameter file - in development");
        logger().log("'makelie' ------------------ Use the old ansatz to perform computations. Not recommended for new codes.");
        logger().log("'subspacediag' ------------- Load parameters from the parameter file and diagonalise in the subspace spanned by the resultant wavefunctions");
        logger().log("'writeproperties' ---------- Print various properties about the solutions found in the parameter file to stdout ");
        logger().log("'generatepathsForsubspace' - Generate random angles and optimise using newton raphson. Pseudo basin hopping");
        logger().log("'filepath XX/YY' ----------- Set the file path to search for resources. filepath should be the complete prefix. E.g. for Hams/H10_Paramaters.dat supply 'filepath Hams/H10'");
        logger().log("'benchmark' ---------------- Benchmarks various operations. Used for development and subject to change");
        logger().log("'NoLowestEigenValue'-------- Don't compute the lowest eigenvalue of the Hamiltonian");
        logger().log("'loadpaths N' -------------- Only load the N lowest paths");
        logger().log("'NOverlap' ----------------- Compute the overlap of the result with the N Lowest eigenvectors");
        logger().log("'NoHess' ------------------- Don't compute the Hessian and metric");
        logger().log("'help' --------------------- Print this");
    }
    static options parse(int argc, char* argv[])
    {
        if (argc <= 1)
        {
            logger().log("Usage: /path/to/cppAnsatzSynth <options>");
            printHelp();
            std::exit(0);
        }
        options o;
        for (int count = 1; count < argc; count++)
        {
            // logger().log("Reading argv", std::string(argv[count]));
            char* arg = argv[count];
            if (!strcmp(arg,"optimise"))
            {
                o.optimise = true;
            }
            else if (!strcmp(arg,"iterativeoptimise"))
            {
                o.iterativeOptimise = true;
            }
            else if (!strcmp(arg,"makelie"))
            {
                o.makeLie = true;
            }
            else if (!strcmp(arg,"subspacediag"))
            {
                o.subspaceDiag = true;
            }
            else if (!strcmp(arg,"writeproperties"))
            {
                o.writeProperties = true;
            }
            else if (!strcmp(arg,"generatepathsForsubspace"))
            {
                o.generatePathsForSubspace = true;
            }
            else if (!strcmp(arg,"filepath"))
            {
                if (count+1 < argc)
                {
                    o.filePath = std::string(argv[count+1]);
                    count++;
                }
                else
                {
                    logger().log("`filepath' specified but not provided");
                    o.ok = false;
                }
            }
            else if (!strcmp(arg,"benchmark"))
            {
                o.benchmark = true;
            }
            else if (!strcmp(arg,"nocompress"))
            {
                o.noCompress = true;
            }
            else if (!strcmp(arg,"onlyevolve"))
            {
                o.onlyEvolve = true;
            }
            else if (!strcmp(arg,"NoLowestEigenValue"))
            {
                o.noLowestEigenValue = true;
            }
            else if (!strcmp(arg,"loadpaths"))
            {
                if (count+1 < argc)
                {
                    o.numberOfPathsToLoad = atoi(argv[count+1]);
                    if (o.numberOfPathsToLoad <= 0)
                    {
                        logger().log("Could not parse integer. Loading all paths");
                        o.numberOfPathsToLoad = -1;
                    }
                    count++;
                }
                else
                {
                    logger().log("`loadpaths' specified but integer N not provided");
                    o.ok = false;
                }
            }
            else if (!strcmp(arg, "NOverlap"))
            {
                if (count + 1 < argc)
                {
                    o.numberOfOverlapsToCompute = atoi(argv[count + 1]);
                    if (o.numberOfOverlapsToCompute <= 0)
                    {
                        logger().log("Could not parse integer. Computing 1 overlap");
                        o.numberOfOverlapsToCompute = 1;
                    }
                    count++;
                }
                else
                {
                    logger().log("`NOverlap' specified but integer N not provided");
                    o.ok = false;
                }
            }
            else if (!strcmp(arg, "NoHess"))
            {
                o.noHess = true;
            }
            else if (!strcmp(arg,"help"))
            {
                printHelp();
                std::exit(0);
            }
            else
            {
                logger().log("Unrecognised command line option",std::string(arg));
            }
        }
        return o;
    }
};

int main(int argc, char *argv[])
{
    //std::vector<uint32_t> statevectorBasis;
    std::vector<numType> statevectorCoeffs;
    options opt = options::parse(argc,argv);
    const std::string& filePath = opt.filePath;
    if (!opt.ok)
        return 1;
    if (opt.filePath.size() == 0)
    {
        logger().log("No file given");
        return 1;
    }
    threadpool::getInstance(NUM_CORES); // create the threadpool
    if (!readCsvState(statevectorCoeffs,filePath + "_Initial.dat"))
    {
        logger().log("Failed to read statevectorCoeffs from",filePath + "_Initial.dat" );
        return 1;
    }

    uint32_t ones = -1;
    char numberOfParticles = -1;
    bool allSameParticleNumber = true;
    bool SZSym = true;
    int spinUp = -1;
    int spinDown = -1;
    int numberOfQubits = 0;
    {
        size_t dummy = statevectorCoeffs.size()-1;
        while(dummy)
        {
            numberOfQubits++;
            dummy = dummy >>1;
        }
    }



    for (size_t i = 0; i < statevectorCoeffs.size(); i++)
    {
        if (statevectorCoeffs[i] == 0.)
        {
            continue;
        }
        if (numberOfQubits %2 != 0)
            SZSym = false;

        char currNumberOfParticles = bitwiseDot(i,ones,32);

        if (numberOfParticles == -1 && currNumberOfParticles != -1)
        {
            numberOfParticles = currNumberOfParticles;
        }
        if (currNumberOfParticles != numberOfParticles)
        {
            allSameParticleNumber = false;
            break;
        }

        int currSpinUp = bitwiseDot(i>>(numberOfQubits/2),ones,32);
        int currSpinDown = bitwiseDot(i,ones,numberOfQubits/2);
        if (spinUp == -1 && currSpinUp != -1)
            spinUp = currSpinUp;
        if (spinDown == -1 && currSpinDown != -1)
            spinDown = currSpinDown;
        if (currSpinUp != spinUp)
            SZSym = false;
        if (currSpinDown != spinDown)
            SZSym = false;

    }
    if (numberOfParticles == -1)
    {
        logger().log("Could not determine particle number");
        return 1;
    }
    if (spinUp == -1 || spinDown == -1)
    {
        logger().log("Could not determine spin number");
        return 1;
    }
    logger().log("Initial state SZSym:",SZSym);
    {
        std::vector<stateRotate::exc> excs;
        stateRotate::loadOperators(filePath + + "_Operators.dat",excs); // we may load it twice. Oh well.
        bool OperatorSZSym = OperatorsHaveSZSymmetry(numberOfQubits,excs);
        logger().log("Operator SZSym:",OperatorSZSym);
        SZSym = OperatorSZSym && SZSym;
    }
    logger().log("SZSym:",SZSym);
    logger().log("particleNumSym:",allSameParticleNumber);

    logger().log("NumberOfQubits",numberOfQubits);
    logger().log("NumberOfParticles",numberOfParticles);
    logger().log("SpinUp",spinUp);
    logger().log("spinDown",spinDown);
    std::shared_ptr<stateRotate> lie = nullptr;
    std::shared_ptr<compressor> comp;
    std::shared_ptr<FusedEvolve> FE;
    if (allSameParticleNumber && SZSym && !opt.noCompress)
        comp = std::make_shared<SZAndnumberOperatorCompressor>(statevectorCoeffs.size(),spinUp,spinDown);
    if (allSameParticleNumber && !SZSym && !opt.noCompress)
        comp = std::make_shared<numberOperatorCompressor>(numberOfParticles,statevectorCoeffs.size());
    logger().log("comp",comp.get());
    std::vector<stateRotate::exc> excs;
    if (opt.makeLie)
    {
        lie = std::make_shared<stateRotate>(numberOfQubits,comp);
        lie->loadOperators(filePath + "_Operators.dat");
    }
    else
    {
        stateRotate::loadOperators(filePath + + "_Operators.dat",excs);
    }

    vector<numType> start(statevectorCoeffs);


    std::shared_ptr<stateAnsatz> myAnsatz = nullptr;
    sparseMatrix<numType,numType> target({1},{1},{1},1);
    if (opt.makeLie)
        myAnsatz = std::make_shared<stateAnsatz>(&target,start,lie.get());
    else if (comp)
    {
        vector<numType> temp;
        compressor::compressVector<numType>(start,temp,comp);
        static_cast<Matrix<numType>&>(start) = std::move(temp);
    }



    std::vector<ansatz::rotationElement> rotationPath;
    loadPath(lie,filePath + "_Operators.dat",rotationPath);

    std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
    std::vector<std::pair<int,realNumType>> order;
    int numberOfUniqueParameters = 0;
    if (!loadParameters(filePath,rotationPath,rotationPaths,order,numberOfUniqueParameters))
        return 1;

    if (opt.numberOfPathsToLoad > -1)
    {
        rotationPaths.resize(opt.numberOfPathsToLoad);
    }

    // sparseMatrix<realNumType,numType> Ham;
    // if (!Ham.loadMatrix(filePath,numberOfQubits,comp))
    //     return 1;
    std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham;
    if (!opt.onlyEvolve)
    {
        Ham = std::make_shared<HamiltonianMatrix<realNumType,numType>>(filePath,numberOfQubits,comp);
        if (!Ham->ok())
            return 1;
    }
    // Ham.dumpMatrix(filePath);


    realNumType NuclearEnergy = 0;
    LoadNuclearEnergy(NuclearEnergy, filePath);

    TUPSQuantities quantityCalc(Ham,order,numberOfUniqueParameters, NuclearEnergy,filePath); // Can also optimise
    if (!opt.makeLie)
    {
        FE = std::make_shared<FusedEvolve>(start,Ham,quantityCalc.m_compressMatrix,quantityCalc.m_deCompressMatrix);
        FE->updateExc(excs);
        if (opt.onlyEvolve)
        {
            FE = std::make_shared<FusedEvolve>(start,Ham,quantityCalc.m_compressMatrix,quantityCalc.m_deCompressMatrix);
            FE->updateExc(excs);
            vector<numType> dest;
            std::vector<realNumType> angles(rotationPath.size());
            std::transform(rotationPaths[1].begin(),rotationPaths[1].end(), angles.begin(),[](baseAnsatz::rotationElement r){return r.second;});
            // logger().log("angles:",angles);
            FE->evolve(dest,angles);
            return 0;
        }
    }

    if (opt.benchmark)
    {
        //benchmark(myAnsatz.get(),rotationPaths[1], Ham,quantityCalc.m_compressMatrix, quantityCalc.m_deCompressMatrix);
        //return 0;
        FE = std::make_shared<FusedEvolve>(start,Ham,quantityCalc.m_compressMatrix,quantityCalc.m_deCompressMatrix);
        FE->updateExc(excs);
        vector<numType> dest;
        std::vector<realNumType> angles(rotationPath.size());
        std::transform(rotationPaths[1].begin(),rotationPaths[1].end(), angles.begin(),[](baseAnsatz::rotationElement r){return r.second;});
        // logger().log("angles:",angles);

        FE->evolve(dest,angles);
        vector<realNumType> deriv;
        realNumType Energy;

        auto time1 = std::chrono::high_resolution_clock::now();
        constexpr int numLoops = 10000;
        for (int i = 0; i < numLoops; i++)
            FE->evolveDerivative(dest,deriv,angles,&Energy);
        auto time2 = std::chrono::high_resolution_clock::now();
        long duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
        logger().log("Benchmark Derivative Time per derivative (us)", (double)duration1/numLoops);
        
        time1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < numLoops; i++)
            Energy = FE->getEnergy(dest);
        time2 = std::chrono::high_resolution_clock::now();
        duration1 = std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count();
        logger().log("Benchmark getEnergy (us)", (double)duration1/numLoops);

        return 0;     
    }

    //TODO command line switches
    if (opt.subspaceDiag || opt.generatePathsForSubspace)
    {
        size_t numberOfPaths = 9;
        size_t numberOfSteps = 9;
        if (opt.generatePathsForSubspace)
        {
            std::srand(100);
            std::vector<realNumType> Energies;
            rotationPaths.erase(rotationPaths.begin()+1,rotationPaths.end());
            size_t pathsFound = 0;
            size_t stepsDone = 0;
            rotationPaths.push_back(rotationPaths[0]);

            while (pathsFound < numberOfPaths && stepsDone < numberOfSteps )
            {
                stepsDone++;
                vector<realNumType>::EigenVector angles(numberOfUniqueParameters);
                for (int i = 0; i < numberOfUniqueParameters; i++)
                {
                    angles(i) = (2*M_PI*std::rand())/(RAND_MAX);
                }
                angles = quantityCalc.m_deCompressMatrix * angles;
                for (size_t i = 0; i < rotationPaths.back().size(); i++)
                {
                    rotationPaths.back()[i].second = angles(i);
                }
                realNumType Energy;
                if (opt.makeLie)
                    Energy = quantityCalc.OptimiseTups(*myAnsatz,rotationPaths.back(),true);
                else
                    Energy = quantityCalc.OptimiseTups(*FE,rotationPaths.back(),true);

                if (std::find_if(Energies.begin(),Energies.end(), [=](realNumType E){return std::abs(E-Energy) < 1e-10;}) == Energies.end())
                {
                    Energies.push_back(Energy);
                    if (pathsFound < numberOfPaths-1)
                    {
                        rotationPaths.push_back(rotationPaths[0]);
                    }
                    pathsFound++;
                }
            }
            logger().log("Found following Energies",Energies);

        }
        if (opt.subspaceDiag)
            quantityCalc.doSubspaceDiagonalisation(myAnsatz,FE,numberOfPaths,rotationPaths);
    }



    if (opt.optimise)
    {
        logger().log("Start Optimise");
        rotationPaths.push_back(rotationPaths[1]);
        if (opt.makeLie)
        {
            quantityCalc.OptimiseTups(*myAnsatz,rotationPaths.back(),true);
        }
        else
        {
            quantityCalc.OptimiseTups(*FE,rotationPaths.back(),true);
        }
    }
    if (opt.iterativeOptimise)
    {
        if (opt.makeLie)
        {
            rotationPaths.push_back(rotationPaths[1]);
            quantityCalc.iterativeTups(*myAnsatz,rotationPaths.back(),true);
            quantityCalc.OptimiseTups(*myAnsatz,rotationPaths.back(),true);
        }
        else
        {
            rotationPaths.push_back(rotationPaths[1]);
            quantityCalc.iterativeTups(*FE,rotationPaths.back(),true);
            quantityCalc.OptimiseTups(*FE,rotationPaths.back(),true);
        }
    }

    //showContinuousSymmetry(rotationPaths,myAnsatz,Ham);


    if (opt.writeProperties)
    {
        TUPSQuantitiesOptions TUPSOpts;
        TUPSOpts.computeLowestEigenValue = !opt.noLowestEigenValue;
        TUPSOpts.numberOfOverlapsToCompute = opt.numberOfOverlapsToCompute;
        TUPSOpts.noHess = opt.noHess;
        quantityCalc.writeProperties(myAnsatz, FE, rotationPaths, TUPSOpts);
    }
    return 0;
}
