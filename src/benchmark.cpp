/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "benchmark.h"
#include "logger.h"
#include <chrono>


void benchmarkDeriv(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, sparseMatrix<realNumType,numType> & Ham)
{

    std::vector<realNumType> angles;
    ansatz->setCalculateFirstDerivatives(false);
    ansatz->setCalculateSecondDerivatives(false);
    ansatz->resetPath();
    for (auto& rp : rotationPath)
    {
        ansatz->addRotation(rp.first,rp.second);
        angles.push_back(rp.second);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        vector<realNumType> deriv;
        ansatz->getDerivativeVec(&Ham,deriv);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("deriv Time taken:",duration);
}

void benchmarkRotate(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rp, vector<numType>& destVec)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        ansatz->calcRotationAlongPath(rp,destVec,startVec);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("Rotate Time taken:",duration);
    startVec.copy(destVec);

    std::shared_ptr<compressor> comp;
    if (ansatz->getLie()->getCompressor(comp))
    {
        compressor::deCompressVector(startVec,destVec,comp);
    }
}

void benchmarkRotate3(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec)
{

    std::vector<realNumType> angles;
    ansatz->setCalculateFirstDerivatives(false);
    ansatz->setCalculateSecondDerivatives(false);
    ansatz->resetPath();
    for (auto& rp : rotationPath)
    {
        ansatz->addRotation(rp.first,rp.second);
        angles.push_back(rp.second);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
    {
        ansatz->updateAngles(angles);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("Rotate3 Time taken:",duration);


    std::shared_ptr<compressor> comp;
    if (ansatz->getLie()->getCompressor(comp))
    {
        compressor::deCompressVector(ansatz->getVec(),destVec,comp);
    }
}


constexpr char dot(const uint32_t a, const uint32_t b, int dim)
{
    uint32_t prod = a&b;
    char ret = 0; //at most 32
    for (int i = 0; i < dim; i++)
    {
        ret += (prod & 1);
        prod>>=1;
    }
    return ret;
}


void benchmarkRotate2(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec)
{

    const vector<numType>& s = ansatz->getStart();
    vector<numType> startVec;
    {
        std::shared_ptr<compressor> comp;
        if (s.getIsCompressed(comp))
            compressor::deCompressVector(s,startVec,comp);
        else
            startVec.copy(s);
    }


    auto lie = ansatz->getLie();
    std::vector<bool> indexActive;
    indexActive.assign(startVec.size(),false);
    uint32_t allOnes = -1;
    for (uint32_t i = 0; i < indexActive.size(); i++)
    {
        if (dot(i,allOnes,32) == 10)
            indexActive[i] = true;
    }

    std::vector<std::vector<uint32_t>> iGenerators;
    std::vector<std::vector<uint32_t>> jGenerators;
    std::vector<std::vector<numType>> dataGenerators;
    {
        std::shared_ptr<compressor> lieCompressor;
        bool lieisCompressed = lie->getCompressor(lieCompressor);

        for (size_t i = 0; i < rotationPath.size(); i++)
        {
            const ansatz::rotationElement& rp = rotationPath[i];

            const matrixType &rotationGenerator = *lie->getLieAlgebraMatrix(rp.first);
            for (auto d = rotationGenerator.begin(); d < rotationGenerator.end(); d++)
            {
                if (abs(*d) != 1)
                    logger().log("Magnitude not 1 but ", *d);
            }
            std::vector<uint32_t> intoIs;
            std::vector<uint32_t> intoJs;

            std::vector<numType> data;
            auto iIt = rotationGenerator.iItBegin();
            auto jIt = rotationGenerator.jItBegin();

            for (auto d = rotationGenerator.begin(); d < rotationGenerator.end(); d++)
            {
                uint32_t iIdx = *iIt;
                uint32_t jIdx = *jIt;
                if (lieisCompressed)
                {
                    lieCompressor->deCompressIndex(iIdx,iIdx);
                    lieCompressor->deCompressIndex(jIdx,jIdx);
                }
                if (std::real(*d) > 0 && indexActive[iIdx] && indexActive[jIdx])
                {
                    intoIs.push_back(iIdx);
                    intoJs.push_back(jIdx);
                    // data.push_back(*d);
                }
                ++iIt;
                ++jIt;
            }

            iGenerators.push_back(std::move(intoIs));
            jGenerators.push_back(std::move(intoJs));
            dataGenerators.push_back(std::move(data));
        }
    }

    //Hardcoded for now



    //Compress the matrices
    std::vector<long int> compressPerm(indexActive.size());
    std::vector<long int> decompressPerm;
    uint32_t activeCount = 0;
    for (uint32_t i = 0; i < indexActive.size(); i++)
    {
        if (indexActive[i])
        {
            compressPerm[i] = activeCount;
            decompressPerm.push_back(i);
            activeCount++;
        }
        else
        {
            compressPerm[i] = -1;
        }
    }
    for (uint32_t rpCount = 0; rpCount < iGenerators.size(); rpCount++)
    {
        for (uint32_t i = 0; i < iGenerators[rpCount].size(); i++)
        {
            iGenerators[rpCount][i] = compressPerm[iGenerators[rpCount][i]];
            jGenerators[rpCount][i] = compressPerm[jGenerators[rpCount][i]];
        }
    }



    auto start = std::chrono::high_resolution_clock::now();
    for (int loopcount = 0; loopcount < 100; loopcount++)
    {
        bool onTemp1 = true;
        vector<numType> src(activeCount);
        for (uint32_t i = 0; i < startVec.size(); i++)
        {
            long int newIdx = compressPerm[i];
            if (newIdx >= 0)
                src[newIdx] = startVec[i];
        }

        for (size_t i = 0; i < rotationPath.size(); i++)
        {
            realNumType theta = rotationPath[i].second;

            // vector<numType> & dst = onTemp1 ? temp2 : temp1;

            double S = 0;
            double C = 0;
            mysincos(theta,&S,&C);

            auto iIdx = iGenerators[i].begin();
            auto jIdx = jGenerators[i].begin();
            auto iEnd = iGenerators[i].end();


            while(iIdx != iEnd)
            {
                numType srcI = src[*iIdx];
                numType srcJ = src[*jIdx];
                src[*iIdx] = srcJ*(S) + srcI*C;
                src[*jIdx] = -srcI*(S) + srcJ*C;

                ++jIdx;
                ++iIdx;
            }
            onTemp1 = !onTemp1;
        }
        //TODO this is not done in the other rotate branch.
        destVec.resize(startVec.size(),false,nullptr);

        for (uint32_t i = 0; i < src.size(); i++)
        {
            long int newIdx = decompressPerm[i];
            if (newIdx >= 0)
                destVec[newIdx] = src[i];
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("Rotate2 Time taken:",duration);


}



void benchmarkMemAccess(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());
    vector<numType> destVec;

    const ansatz::rotationElement& rp = rotationPath[0];
    const matrixType &rotationGenerator = *ansatz->getLie()->getLieAlgebraMatrix(rp.first);
    const sparseMatrix<numType,numType>& lhs = rotationGenerator;
    auto start = std::chrono::high_resolution_clock::now();
    int numIt = 50*rotationPath.size()*2;
    for (int i = 0; i < numIt; i++)
    {
        destVec.resize(startVec.size(),false,nullptr); // need memset version

        auto d = lhs.begin();
        auto iIdx = lhs.iItBegin();
        auto jIdx = lhs.jItBegin();
        auto iEnd = lhs.iItEnd();


        while(iIdx != iEnd)
        {
            destVec[*iIdx] += startVec[*jIdx]*(*d);
            ++jIdx;
            ++iIdx;
            ++d;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("memaccess Time taken:",duration);
    logger().log("NumIt:", numIt);

}

void benchmarkMemAccess2(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());
    vector<numType> destVec;

    const ansatz::rotationElement& rp = rotationPath[0];
    const matrixType &rotationGenerator = *ansatz->getLie()->getLieAlgebraMatrix(rp.first);
    const sparseMatrix<numType,numType>& lhs = rotationGenerator;
    auto start = std::chrono::high_resolution_clock::now();
    int numIt = 50*rotationPath.size()*2;
    for (int i = 0; i < numIt; i++)
    {
        destVec.resize(startVec.size(),false,nullptr); // need memset version

        auto d = lhs.begin();
        auto dEnd = lhs.end();
        auto iIdx = lhs.iItBegin();
        auto jIdx = lhs.jItBegin();


        while(iIdx != lhs.iItEnd() && d < dEnd)
        {
            destVec[*iIdx] += startVec[*jIdx];
            ++jIdx;
            ++iIdx;
            ++d;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("memaccess2 Time taken:",duration);
    logger().log("NumIt:", numIt);

}

void benchmarkMemAccess3(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());
    vector<numType> destVec;

    const ansatz::rotationElement& rp = rotationPath[0];
    const matrixType &rotationGenerator = *ansatz->getLie()->getLieAlgebraMatrix(rp.first);
    const sparseMatrix<numType,numType>& lhs = rotationGenerator;
    auto start = std::chrono::high_resolution_clock::now();
    int numIt = 50*rotationPath.size()*2;
    for (int i = 0; i < numIt; i++)
    {
        destVec.resize(startVec.size(),false,nullptr); // need memset version

        auto d = lhs.begin();
        auto dEnd = lhs.end();
        auto iIdx = lhs.iItBegin();
        auto jIdx = lhs.jItBegin();


        while(iIdx != lhs.iItEnd() && d < dEnd)
        {
            destVec[*jIdx] += startVec[*jIdx];
            ++jIdx;
            ++iIdx;
            ++d;
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("memaccess3 Time taken:",duration);
    logger().log("NumIt:", numIt);

}

void benchmarkMemAccess4(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());
    vector<numType> destVec;

    const ansatz::rotationElement& rp = rotationPath[0];
    const matrixType &rotationGenerator = *ansatz->getLie()->getLieAlgebraMatrix(rp.first);
    const sparseMatrix<numType,numType>& lhs = rotationGenerator;
    auto start = std::chrono::high_resolution_clock::now();
    int numIt = 50*rotationPath.size()*2;
    for (int i = 0; i < numIt; i++)
    {
        destVec.resize(startVec.size(),false,nullptr); // need memset version
        destVec.resize(startVec.size(),false,nullptr); // need memset version
        destVec.resize(startVec.size(),false,nullptr); // need memset version
        destVec.resize(startVec.size(),false,nullptr); // need memset version
        destVec.resize(startVec.size(),false,nullptr); // need memset version
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("memaccess4 Time taken:",duration);
    logger().log("NumIt:", numIt);

}

void benchmark(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rp, sparseMatrix<realNumType,numType>& Ham)
{
    vector<numType> dest2;
    benchmarkRotate3(ansatz,rp,dest2);
    benchmarkDeriv(ansatz,rp,Ham);


    // vector<numType> dest1;
    // vector<numType> dest2;
    // benchmarkRotate(ansatz,rp,dest1);
    // benchmarkRotate2(ansatz,rp,dest2);
    // benchmarkRotate3(ansatz,rp,dest2);
    // logger().log("mag1:", dest1.dot(dest1));
    // logger().log("mag2:", dest2.dot(dest2));
    // logger().log("overlap:", dest1.dot(dest2));

    // benchmarkMemAccess(ansatz,rp);
    //benchmarkMemAccess2(ansatz,rp);
    // benchmarkMemAccess3(ansatz,rp);
    // benchmarkMemAccess4(ansatz,rp);

}
