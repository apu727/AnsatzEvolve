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
            fprintf(stderr,"Failed to read CSV");
            return 1;
        }
    }
    else
    {
        fprintf(stderr,"No built in statevector");
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
    logger().log("SZSym:",SZSym);
    logger().log("particleNumSym:",allSameParticleNumber);
    logger().log("NumberOfQubits",numberOfQubits);
    logger().log("NumberOfParticles",numberOfParticles);
    logger().log("SpinUp",spinUp);
    logger().log("spinDown",spinDown);
    std::shared_ptr<stateRotate> lie = nullptr;
    std::shared_ptr<compressor> comp;
    std::shared_ptr<FusedEvolve> FE;
    if (allSameParticleNumber && SZSym)
        comp = std::make_shared<SZAndnumberOperatorCompressor>(statevectorCoeffs.size(),spinUp,spinDown);
    if (allSameParticleNumber && !SZSym)
        comp = std::make_shared<numberOperatorCompressor>(statevectorCoeffs.size(),numberOfParticles);

    bool makeLie = false;
    std::vector<stateRotate::exc> excs;
    if (makeLie)
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
    if (makeLie)
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



    // sparseMatrix<realNumType,numType> Ham;
    // if (!Ham.loadMatrix(filePath,numberOfQubits,comp))
    //     return 1;
    std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham = std::make_shared<HamiltonianMatrix<realNumType,numType>>(filePath,numberOfQubits,comp);
    if (!Ham->ok())
        return 1;
    // Ham.dumpMatrix(filePath);


    realNumType NuclearEnergy = 0;
    LoadNuclearEnergy(NuclearEnergy, filePath);

    TUPSQuantities quantityCalc(Ham,order,numberOfUniqueParameters, NuclearEnergy,filePath); // Can also optimise
    if (!makeLie)
    {
        FE = std::make_shared<FusedEvolve>(start,Ham,quantityCalc.m_compressMatrix,quantityCalc.m_deCompressMatrix);
        FE->updateExc(excs);
    }
    // benchmark(myAnsatz.get(),rotationPaths[1], Ham,quantityCalc.m_compressMatrix, quantityCalc.m_deCompressMatrix);
    // return 0;

    //TODO command line switches
    bool optimise = false;
    bool subspaceDiag = false;
    bool writeProperties = true;
    bool generatePathsForSubspace = true;
    if (subspaceDiag || generatePathsForSubspace)
    {
        size_t numberOfPaths = 9;
        size_t numberOfSteps = 9;
        if (generatePathsForSubspace)
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
                if (makeLie)
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
        if (subspaceDiag)
            quantityCalc.doSubspaceDiagonalisation(myAnsatz,FE,numberOfPaths,rotationPaths);
    }



    if (optimise)
    {
        logger().log("Start Optimise");
        rotationPaths.push_back(rotationPaths[1]);
        if (makeLie)
        {
            quantityCalc.OptimiseTups(*myAnsatz,rotationPaths.back(),true);
        }
        else
        {
            quantityCalc.OptimiseTups(*FE,rotationPaths.back(),true);
        }

        // quantityCalc.iterativeTups(Ham,rotationPaths[0],*myAnsatz,true);
        // rotationPaths.push_back(myAnsatz->getRotationPath());
        // quantityCalc.OptimiseTups(Ham,rotationPaths.back(),*myAnsatz,true);
        // rotationPaths.push_back(myAnsatz->getRotationPath());



        // quantityCalc.OptimiseTupsLBFGS(Ham,rotationPaths[1],*myAnsatz,false);



    }

    //showContinuousSymmetry(rotationPaths,myAnsatz,Ham);


    if (writeProperties)
        quantityCalc.writeProperties(myAnsatz,FE,rotationPaths);
    return 0;
}
