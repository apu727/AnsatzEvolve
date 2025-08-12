/* Any copyright is dedicated to the Public Domain.
 * https://creativecommons.org/publicdomain/zero/1.0/ */

#include "benchmark.h"
#include "operatorpool.h"
#include "ansatz.h"
#include "threadpool.h"

#include "TUPSLoadingUtils.h"
#include "tupsquantities.h"
#include "logger.h"


#include <vector>


int main(int argc, char *argv[])
{
    //std::vector<uint32_t> statevectorBasis;
    std::vector<numType> statevectorCoeffs;

    std::string filePath = "";
    enum typeofFile
    {
        NoFile=0,
        PauliHamiltonian=1,
        Statevector=2
    };

    typeofFile fileType = NoFile;

    if (argc >=2)
    {
        if (argv[1][0] != '_')
        {//escape character to do internal H2 Ham instead
            fileType = Statevector;
            filePath = std::string(argv[1]); // dangerous but oops
        }

    }
    if (argc >=3)
    {
        sscanf(argv[2],"%lu",&NUM_CORES);
        fprintf(stderr,"specified num cores changed to:%lu\n",NUM_CORES);
    }


    threadpool::getInstance(NUM_CORES); // create the threadpool
    if (fileType != NoFile)
    {
        if (!readCsvState(statevectorCoeffs,filePath + "_Initial.dat"))
        {
            printf("Failed to read CSV");
            return 1;
        }
    }
    else
    {
        printf("No built in statevector");
        return 1;
    }
    uint32_t ones = -1;
    char numberOfParticles = -1;
    bool allSameParticleNumber = true;

    for (size_t i = 0; i < statevectorCoeffs.size(); i++)
    {
        if (statevectorCoeffs[i] == 0.)
        {
            continue;
        }
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
    }
    if (numberOfParticles == -1)
    {
        logger().log("Could not determine particle number");
        return 1;
    }

    std::shared_ptr<stateRotate> lie = nullptr;
    if (allSameParticleNumber)
    {
        lie = std::make_shared<stateRotate>(std::log2(statevectorCoeffs.size()),true,numberOfParticles);
    }
    else
        lie = std::make_shared<stateRotate>(std::log2(statevectorCoeffs.size()));

    lie->loadOperators(filePath + "_Operators.dat");

    vector<numType> start(statevectorCoeffs);


    std::shared_ptr<stateAnsatz> myAnsatz = nullptr;
    sparseMatrix<numType,numType> target({1},{1},{1},1);

    myAnsatz = std::make_shared<stateAnsatz>(&target,start,lie.get());



    std::vector<ansatz::rotationElement> rotationPath;
    loadPath(*lie,filePath + "_Operators.dat",rotationPath);

    std::vector<std::vector<ansatz::rotationElement>> rotationPaths;
    std::vector<std::pair<int,realNumType>> order;
    int numberOfUniqueParameters = 0;
    if (!loadParameters(filePath,rotationPath,rotationPaths,order,numberOfUniqueParameters))
        return 1;



    sparseMatrix<realNumType,numType> Ham;
    Ham.loadMatrix(filePath);
    if (allSameParticleNumber)
    {
        std::shared_ptr<compressor> comp;
        lie->getCompressor(comp);
        Ham.compress(comp);
    }



    realNumType NuclearEnergy = 0;
    LoadNuclearEnergy(NuclearEnergy, filePath);

    TUPSQuantities quantityCalc(Ham,order,numberOfUniqueParameters, NuclearEnergy,filePath); // Can also optimise
    // benchmark(myAnsatz.get(),rotationPaths[1], Ham,quantityCalc.m_compressMatrix, quantityCalc.m_deCompressMatrix);
    // return 0;

    //TODO command line switches
    bool optimise = true;
    bool subspaceDiag = false;
    bool writeProperties = true;
    bool generatePathsForSubspace = false;
    if (subspaceDiag)
    {
        size_t numberOfPaths = 9;

        if (generatePathsForSubspace)
        {
            std::srand(100);
            std::vector<realNumType> Energies;
            rotationPaths.erase(rotationPaths.begin()+1,rotationPaths.end());
            size_t pathsFound = 0;
            rotationPaths.push_back(rotationPaths[0]);

            while (pathsFound < numberOfPaths)
            {
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
                realNumType Energy = quantityCalc.OptimiseTups(Ham,rotationPaths.back(),*myAnsatz,true);
                if (std::find_if(Energies.begin(),Energies.end(), [=](realNumType E){return std::abs(E-Energy) < 1e-10;}) == Energies.end())
                {
                    Energies.push_back(Energy);
                    rotationPaths.back() = myAnsatz->getRotationPath();
                    if (pathsFound < numberOfPaths-1)
                    {
                        rotationPaths.push_back(rotationPaths[0]);
                    }
                    pathsFound++;
                }
            }
            logger().log("Found following Energies",Energies);

        }
        quantityCalc.doSubspaceDiagonalisation(*myAnsatz,numberOfPaths,rotationPaths);
    }



    if (optimise)
    {
        logger().log("Start Optimise");
        rotationPaths.push_back(rotationPaths[1]);
        quantityCalc.OptimiseTups(Ham,rotationPaths.back(),*myAnsatz,true);

        // quantityCalc.iterativeTups(Ham,rotationPaths[0],*myAnsatz,true);
        // rotationPaths.push_back(myAnsatz->getRotationPath());
        // quantityCalc.OptimiseTups(Ham,rotationPaths.back(),*myAnsatz,true);
        // rotationPaths.push_back(myAnsatz->getRotationPath());



        // quantityCalc.OptimiseTupsLBFGS(Ham,rotationPaths[1],*myAnsatz,false);



    }

    //showContinuousSymmetry(rotationPaths,myAnsatz,Ham);


    if (writeProperties)
        quantityCalc.writeProperties(rotationPaths,myAnsatz.get());
    return 0;
}
