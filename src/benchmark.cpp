/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "benchmark.h"
#include "logger.h"
#include "myComplex.h"
#include "threadpool.h"
#include "fusedevolve.h"
#include <chrono>
#include <numeric>
#include "math.h"


constexpr char dot(const uint64_t a, const uint64_t b, int dim)
{
    uint64_t prod = a&b;
    char ret = 0; //at most 64
    for (int i = 0; i < dim; i++)
    {
        ret += (prod & 1);
        prod>>=1;
    }
    return ret;
}

template <typename T>
std::vector<std::vector<std::size_t>> Benchmark_sortPermutation(const T& vec)
{
    /*    typedef std::array<bool,localVectorSize> signMap; //true means +ve
    typedef std::array<uint64_t,localVectorSize> localVectorMap;
    typedef std::vector<std::pair<localVectorMap,signMap>> localVector;
    std::array<localVector,localVectorSize> vec;
*/
    std::vector<std::vector<std::size_t>> ret;
    ret.resize(vec.size());
    for (size_t idx = 0; idx < vec.size(); idx++)
    {
        size_t N = vec[idx].size();
        std::vector<std::size_t> p(N);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
                  [&](std::size_t i, std::size_t j){ return vec[idx][i].first[0] < vec[idx][j].first[0]; });
        ret[idx] = std::move(p);
    }
    return ret;
}
template <typename T>
void Benchmark_applyPermutationAndStore(const std::vector<T>& vec, const std::vector<std::size_t>& perm, std::vector<T>& dest)
{
    size_t N = perm.size();
    dest.resize(N);
    std::transform(perm.begin(), perm.end(), dest.begin(),
                   [&](std::size_t i){ return vec[i]; });
}

void benchmarkDeriv(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, std::shared_ptr<HamiltonianMatrix<realNumType,numType>> & Ham)
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
        ansatz->getDerivativeVec(Ham,deriv);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("deriv Time taken:",duration);
}

void benchmarkRotate(stateAnsatz* ansatz, const std::vector<ansatz::rotationElement>& rp, vector<numType>& destVec, int loopcount,
                     const vector<numType>*, realNumType**)
{
    vector<numType> startVec;
    startVec.copy(ansatz->getStart());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loopcount; i++)
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
        compressor::deCompressVector<numType>(startVec,destVec,comp);
    }
}

void benchmarkRotate3(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec, int loopCount = 100)
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
    for (int i = 0; i < loopCount; i++)
    {
        ansatz->updateAngles(angles);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("Rotate3 Time taken:",duration);


    std::shared_ptr<compressor> comp;
    if (ansatz->getLie()->getCompressor(comp))
    {
        compressor::deCompressVector<numType>(ansatz->getVec(),destVec,comp);
    }
}
#ifndef useComplex
//State and sign. True is +ve
std::pair<uint64_t,bool> applyExcToBasisState(uint64_t state, const stateRotate::exc& a)
{
    uint64_t activeBits =  0;
    uint64_t createBits = 0;
    uint64_t annihilateBits = 0;
    if (a[0] < 0 && a[1] < 0)
        return std::make_pair(state,true);

    if (a[2] > -1 && a[3] > -1)
    {
        if (a[0] == a[1] || a[2] == a[3])
        {
            fprintf(stderr,"Wrong order in creation annihilation operators");
            return std::make_pair(state,true);
        }
        createBits = (1ul<<a[0]) | (1ul<<a[1]);
        annihilateBits = (1ul<<a[2]) | (1ul<<a[3]);
        activeBits = createBits | annihilateBits;
    }
    else
    {
        createBits = (1ul<<a[0]);
        annihilateBits = (1ul<<a[1]);
        activeBits = createBits | annihilateBits;
    }

    uint64_t basisState = state;
    uint64_t resultState = basisState;


    numType phase = 0;
    uint64_t maskedBasisState = basisState & activeBits;

    if (createBits == annihilateBits) // number operator
    {
#ifdef useComplex
        if (((maskedBasisState & annihilateBits) ^ annihilateBits) == 0)
        {
            phase = iu;
        }
        else
        {
            phase = 0;
        }
        resultState = basisState;
#else
        logger().log("Complex number not supported in this build, please rebuild");
        __builtin_trap();
#endif
    }
    else
    { // excitation operator. These are different since we need to make it anti-hermitian which is done differently
        if (((maskedBasisState & annihilateBits) ^ annihilateBits) == 0 && (((maskedBasisState ^ annihilateBits) & createBits)) == 0)
        {// This allows operators like a^+_4 a^+_3 a_3 a_2 to be handled properly
            phase = 1;
            resultState = (basisState ^ annihilateBits) ^ createBits;
        }
        else if (((maskedBasisState & createBits) ^ createBits) == 0 && (((maskedBasisState ^ createBits) & annihilateBits)) == 0)
        {
            phase = -1;
            resultState = (basisState ^ createBits) ^ annihilateBits;
        }
        else
        {
            phase = 0;
        }
    }

    if (phase == 1)
    {
        return std::make_pair(resultState,true);
    }
    else if (phase == -1)
    {
        return std::make_pair(resultState,false);
    }
    else
    {
        return std::make_pair(state,false);
    }


}

/*
 * numberOfRotsThatExist --------- Max number of rots that exist. Used for bounds checking
 * activeRotsIdx ----------------- bitmap of which rotations are active
 * numberToFuse ------------------ number of active rotations
 * currentMap -------------------- A pointer into an array at least 1<<numberToFuse large. The destination
 * currentSigns ------------------ A pointer into an array at least ((1<<numberToFuse)/2)*numberToFuse large. The signs for the results. The first (1<<numberToFuse)/2 are the for the first rotation etc
 * rotIdx ------------------------ offset into activeRotsIdx
 * numberOfActiveRotationsSoFar -- Gives the number of rotations that have been computed so far. Needed for the stride length
 * initialLinks ------------------ The initial computation that gave the active rotations. This is done before because we need to store into a different array
 * rots -------------------------- Array giving the stateRotate::exc representation of the rotation
 */

template<typename indexType, indexType numberOfRotsThatExist>
void fillCurrentMap(indexType activeRotsIdx, indexType numberToFuse,
                    uint64_t* currentMap, bool* currentSigns,
                    indexType rotIdx, indexType numberOfActiveRotationsSoFar,
                    const std::array<std::pair<uint64_t,bool>,numberOfRotsThatExist>& initialLinks, const std::array<stateRotate::exc,numberOfRotsThatExist>& rots)
{
    const indexType localVectorSize = 1<<numberToFuse;

    indexType strideSize = 1<<numberOfActiveRotationsSoFar;
    indexType numberOfRotationsAddedOnThisLayer = 0;

    currentMap[strideSize+0] = initialLinks[rotIdx].first;
    //(localVectorSize/2)*numberOfActiveRotationsSoFar.

    //(localVectorSize/2) gives how many signs are relevant per layer.
    //numberOfActiveRotationsSoFar is how many layers have been initialised
    currentSigns[(localVectorSize/2)*numberOfActiveRotationsSoFar + numberOfRotationsAddedOnThisLayer++] = initialLinks[rotIdx].second; //should always be true

    //All of currentMap[0:strideSize] is filled out by the time this is called
    for (indexType idx = 1; idx < strideSize; idx++)
    {
        auto effect = applyExcToBasisState(currentMap[idx],rots[rotIdx]);
        assert(effect.first != currentMap[idx]);
        currentMap[idx + strideSize] = effect.first;
        currentSigns[numberOfRotationsAddedOnThisLayer++ + (localVectorSize/2)*numberOfActiveRotationsSoFar] = effect.second;
    }
    indexType nextRotIdx = rotIdx+1;
    while(nextRotIdx < numberOfRotsThatExist && ((1<<nextRotIdx) & activeRotsIdx) == 0)
        ++nextRotIdx;
    if (nextRotIdx < numberOfRotsThatExist)
        fillCurrentMap<indexType,numberOfRotsThatExist>(activeRotsIdx,numberToFuse,currentMap,currentSigns,nextRotIdx,numberOfActiveRotationsSoFar+1,initialLinks,rots);
    //else we are done and the vector has been determined

    //Now currentMap[0:1<<numberToFuse] is filled. i.e. the whole thing. Note that 2*(1<<MaxRotIdx) = 1<<numberToFuse
    //Asserts to check everything is consistent
    //Also fill out the signs for the rotations from [strideSize:1<<numberToFuse]?
    assert(localVectorSize % strideSize == 0);

    for (indexType startIdx = 2*strideSize; startIdx < localVectorSize-strideSize; startIdx += 2*strideSize)
    {
        for (indexType idx = startIdx; idx < startIdx +strideSize; idx++)
        {
            auto effect = applyExcToBasisState(currentMap[idx],rots[rotIdx]);
            assert(effect.first != currentMap[idx]);
            assert(effect.first == currentMap[idx + strideSize]);
            currentSigns[numberOfRotationsAddedOnThisLayer++ + (localVectorSize/2)*numberOfActiveRotationsSoFar] = effect.second;
        }
        //Note idx = startIdx + strideSize -> startIdx + 2*strideSize are skipped. Hence the notation on currentSigns
    }
    assert(numberOfRotationsAddedOnThisLayer == localVectorSize/2);
};



#define makeLayer0(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 1 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*0] ? 1 : -1)*temp2*S[0] + temp1 * C[0];\
scratchSpace[firstIndex + 1 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*0] ? 1 : -1)*temp1*S[0] + temp2 * C[0];\
if constexpr(toBraket)\
{\
        if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
        {\
            *(result[0]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*0]) * scratchSpace[firstIndex + 1 + i*rotCount]);\
            *(result[0]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 1 + i*rotCount + localVectorSize*0]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
        }\
        else\
        {\
            *(result[0]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*0]) * scratchSpace[firstIndex + 1 + i*rotCount]);\
            *(result[0]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 1 + i*rotCount + localVectorSize*0]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
        }\
}

#define makeLayer1(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 2 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp2*S[1] + temp1 * C[1];\
scratchSpace[firstIndex + 2 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp1*S[1] + temp2 * C[1];\
if constexpr(toBraket)\
{\
        if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
            *(result[1]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*1]) * scratchSpace[firstIndex + 2 + i*rotCount]);\
            *(result[1]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 2 + i*rotCount + localVectorSize*1]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
        else\
    {\
            *(result[1]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*1]) * scratchSpace[firstIndex + 2 + i*rotCount]);\
            *(result[1]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 2 + i*rotCount + localVectorSize*1]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
}

#define makeLayer2(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 4 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp2*S[2] + temp1 * C[2];\
scratchSpace[firstIndex + 4 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp1*S[2] + temp2 * C[2];\
if constexpr(toBraket)\
{\
        if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
            *(result[2]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*2]) * scratchSpace[firstIndex + 4 + i*rotCount]);\
            *(result[2]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 4 + i*rotCount + localVectorSize*2]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
        else\
    {\
            *(result[2]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*2]) * scratchSpace[firstIndex + 4 + i*rotCount]);\
            *(result[2]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 4 + i*rotCount + localVectorSize*2]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
}

#define makeLayer3(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 8 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp2*S[3] + temp1 * C[3];\
scratchSpace[firstIndex + 8 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp1*S[3] + temp2 * C[3];\
if constexpr(toBraket)\
{\
        if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
            *(result[3]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*3]) * scratchSpace[firstIndex + 8 + i*rotCount]);\
            *(result[3]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 8 + i*rotCount + localVectorSize*3]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
        else\
    {\
            *(result[3]) -= std::real(myConj(scratchSpacetoBraKet[firstIndex + 0 + i*rotCount + localVectorSize*3]) * scratchSpace[firstIndex + 8 + i*rotCount]);\
            *(result[3]) += std::real(myConj(scratchSpacetoBraKet[firstIndex + 8 + i*rotCount + localVectorSize*3]) * scratchSpace[firstIndex + 0 + i*rotCount]);\
    }\
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_1(numType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=1*/,size_t numberToRepeat, indexType localVectorSize,
                                       const numType* scratchSpacetoBraKet/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 2;
    constexpr indexType signsStride = (rotCount/2)*1;

    numType temp1;
    numType temp2;
#pragma GCC unroll 2
    for (size_t i = 0; i < numberToRepeat; i++)
    {
        // #define signs true || signs
        makeLayer0(0,1,0,toBraket)
        // #undef signs
    }
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_2(numType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType localVectorSize,
                                       const numType* scratchSpacetoBraKet/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 4;
    constexpr indexType signsStride = (rotCount/2)*2;

    numType temp1;
    numType temp2;
    for (size_t i = 0; i < numberToRepeat; i++)
    {
        // #define signs true || signs
        makeLayer0(0,2,0,toBraket)
        makeLayer0(2,2,1,toBraket)
        makeLayer1(0,2,0,toBraket)
        makeLayer1(1,2,1,toBraket)
        // #undef signs
    }
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_3(numType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType localVectorSize,
                                const numType* scratchSpacetoBraKet/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 8;
    constexpr indexType signsStride = (rotCount/2)*3;

    numType temp1;
    numType temp2;
    for (size_t i = 0; i < numberToRepeat; i++)
    {
        // #define signs true || signs
        makeLayer0(0,4,0,toBraket)
        makeLayer0(2,4,1,toBraket)
        makeLayer0(4,4,2,toBraket)
        makeLayer0(6,4,3,toBraket)

        makeLayer1(0,4,0,toBraket)
        makeLayer1(1,4,1,toBraket)
        makeLayer1(4,4,2,toBraket)
        makeLayer1(5,4,3,toBraket)

        makeLayer2(0,4,0,toBraket)
        makeLayer2(1,4,1,toBraket)
        makeLayer2(2,4,2,toBraket)
        makeLayer2(3,4,3,toBraket)
        // #undef signs
    }
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_4(numType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType localVectorSize,
                                      const numType* scratchSpacetoBraKet/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 16;
    constexpr indexType signsStride = (rotCount/2)*4;
    numType temp1;
    numType temp2;
    for (size_t i = 0; i < numberToRepeat; i++)
    {
        // #define signs true || signs
        makeLayer0(0,8,0,toBraket)
        makeLayer0(2,8,1,toBraket)
        makeLayer0(4,8,2,toBraket)
        makeLayer0(6,8,3,toBraket)
        makeLayer0(8,8,4,toBraket)
        makeLayer0(10,8,5,toBraket)
        makeLayer0(12,8,6,toBraket)
        makeLayer0(14,8,7,toBraket)

        makeLayer1(0,8,0,toBraket)
        makeLayer1(1,8,1,toBraket)
        makeLayer1(4,8,2,toBraket)
        makeLayer1(5,8,3,toBraket)
        makeLayer1(8,8,4,toBraket)
        makeLayer1(9,8,5,toBraket)
        makeLayer1(12,8,6,toBraket)
        makeLayer1(13,8,7,toBraket)

        makeLayer2(0,8,0,toBraket)
        makeLayer2(1,8,1,toBraket)
        makeLayer2(2,8,2,toBraket)
        makeLayer2(3,8,3,toBraket)
        makeLayer2(8,8,4,toBraket)
        makeLayer2(9,8,5,toBraket)
        makeLayer2(10,8,6,toBraket)
        makeLayer2(11,8,7,toBraket)

        makeLayer3(0,8,0,toBraket)
        makeLayer3(1,8,1,toBraket)
        makeLayer3(2,8,2,toBraket)
        makeLayer3(3,8,3,toBraket)
        makeLayer3(4,8,4,toBraket)
        makeLayer3(5,8,5,toBraket)
        makeLayer3(6,8,6,toBraket)
        makeLayer3(7,8,7,toBraket)
        // #undef signs
    }
}

template<typename indexType, indexType numberToFuse,indexType localVectorSize, bool toBraket, indexType numberToFuseArray>
inline void BENCHMARK_rotate(numType* scratchSpace,const bool* signs, const realNumType* S, const realNumType* C/*length=numberToFuse*/,size_t numberToRepeat,
                             const numType (&scratchSpacetoBraKet)[numberToFuseArray][localVectorSize], realNumType* const (&result)[numberToFuse]/*Result is stored at *(result[rotIdx]) where rotIdx \in [0,numberToFuse)*/)
{
    static_assert(numberToFuseArray >= numberToFuse,"BENCHMARK_rotate invalid array passed");
    //Call the optimised routines if they exist
    if constexpr(numberToFuse == 1)
        return BENCHMARK_rotate_1<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,(const numType*)scratchSpacetoBraKet,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 2)
        return BENCHMARK_rotate_2<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,(const numType*)scratchSpacetoBraKet,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 3)
        return BENCHMARK_rotate_3<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,(const numType*)scratchSpacetoBraKet,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 4)
        return BENCHMARK_rotate_4<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,(const numType*)scratchSpacetoBraKet,static_cast<realNumType*const*>(result));
    else
    {
        constexpr indexType pseudoVectorSize = 1<<numberToFuse;
        constexpr indexType signsStride = (pseudoVectorSize/2)*numberToFuse;
        constexpr indexType rotCount = 1<<numberToFuse;

        for (size_t i = 0; i < numberToRepeat; i++)
        {
            for (indexType rotIdx = 0; rotIdx < numberToFuse; rotIdx++)
            {
                indexType stride = 1<<rotIdx;
                indexType numberOfRotationsPerformed = 0;

                for (indexType startIdx = 0; startIdx < pseudoVectorSize-stride; startIdx += 2*stride)
                {
                    for (indexType idx = startIdx; idx < startIdx + stride; idx++)
                    {
                        numType temp1 = scratchSpace[idx + rotCount*i];
                        numType temp2 = scratchSpace[idx+stride + rotCount*i];
                        const bool sign = signs[numberOfRotationsPerformed++ + (pseudoVectorSize/2)*rotIdx + signsStride*i];
                        // #define sign true
                        scratchSpace[idx + rotCount*i] = (sign ? 1 : -1)*temp2*S[rotIdx] + temp1 * C[rotIdx];
                        scratchSpace[idx+stride + rotCount*i] = -(sign ? 1 : -1)*temp1*S[rotIdx] + temp2 * C[rotIdx];
                        if constexpr(toBraket)
                        {
                            if (sign)
                            {
                                *(result[rotIdx]) += std::real(myConj(scratchSpacetoBraKet[rotIdx][idx + rotCount*i]) * scratchSpace[idx+stride + rotCount*i]);
                                *(result[rotIdx]) -= std::real(myConj(scratchSpacetoBraKet[rotIdx][idx+stride + rotCount*i]) * scratchSpace[idx + rotCount*i]);
                            }
                            else
                            {
                                *(result[rotIdx]) -= std::real(myConj(scratchSpacetoBraKet[rotIdx][idx + rotCount*i]) * scratchSpace[idx+stride + rotCount*i]);
                                *(result[rotIdx]) += std::real(myConj(scratchSpacetoBraKet[rotIdx][idx+stride + rotCount*i]) * scratchSpace[idx + rotCount*i]);
                            }
                        }
                        // #undef sign
                    }
                }
            }
        }
    }
}

template<typename indexType, indexType numberToFuse, bool BraketWithTangentOfResult>
void benchmarkRotateFuseN(stateAnsatz* ansatz, const std::vector<ansatz::rotationElement>& rotationPath, vector<numType>& destVec, int loopCount = 100,
                          const vector<numType>* toBraKet = nullptr, realNumType** result = nullptr/*result is array of pointers to storage places. The array has length rotationPath.size()*/)
{
    // static_assert(numberToFuse >= 2, "Must fuse at least 2 gates");
    static_assert(numberToFuse < sizeof(indexType)*8 && std::is_unsigned_v<indexType>, "size stored in uintX so can fuse at most X-1 gates = 2^(X-1) elements. 2^X is out of the range and so cannot store the size needed");

    if constexpr (BraketWithTangentOfResult)
    {
        if (toBraKet == nullptr || result == nullptr)
            __builtin_trap();
    }
    else
    {
        //These are dereferences but their results are not used
        result = new realNumType*[rotationPath.size()];
        // toBraKet = new Matrix<numType>(1,rotationPath.size());
    }

    constexpr indexType localVectorSize = 1<<numberToFuse;
    numType scratchSpace[localVectorSize];
    numType scratchSpacetoBraKet[numberToFuse][localVectorSize];


    typedef std::array<bool,(localVectorSize/2)*numberToFuse> signMap; //true means +ve
    typedef std::array<uint64_t,localVectorSize> localVectorMap;
    typedef std::vector<std::pair<localVectorMap,signMap>> localVector;

    typedef std::vector<std::array<localVector,localVectorSize>> fusedAnsatz; //\Sum_{k=1}^{n} n choose k = 2^n for n >=0
    fusedAnsatz myFusedAnsatz;

    //preprocess
    assert(rotationPath.size() % numberToFuse == 0);

    vector<numType> startVec;
    startVec.copy(ansatz->getStart());

    std::vector<uint64_t> activebasisStates;
    activebasisStates.resize(startVec.size());

    std::shared_ptr<compressor> comp;
    ansatz->getLie()->getCompressor(comp);
    assert(comp != nullptr);
    for (uint64_t i = 0; i < startVec.size(); i++)
    {
        comp->deCompressIndex(i,activebasisStates[i]);
    }
    size_t totalRotcount = 0;
    for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
    {
        std::array<stateRotate::exc,numberToFuse> rots;
        for (indexType idx = 0; idx < numberToFuse; idx++)//TODO check that this exists in the path
            dynamic_cast<stateRotate*>(ansatz->getLie())->convertIdxToExc(rotationPath[i+idx].first,rots[idx]);

        //Structure is
        /*
         * With R0 first
         *   R0│  R1 │     R2
         * 0─┐───┬──────────────
         * 1─┘───│─┬────────────
         * 2─┐───┴─│────────────
         * 3─┘─────┴────────────
         *
         *
         * Therefore:
         * Starting with R0.
         * 0->1 Via R0|0>
         * 0->2 Via R1|0>
         * 1->3 Via R1R0|0> = R0  R1|0> = R0|2>
         * So 2->3 via R0 and the space is consistent
         *
         *For this we need them to commute!
         *
         */
        //Assuming they commute...

        // localVectors currentLocalVectors;
        // localVectors currentRot0OnlyVectors;
        // localVectors currentRot1OnlyVectors;

        //If rot0 and rot2 are active then the index is 0b101. Note that 0b000 is always empty
        std::array<localVector,localVectorSize> currentLocalVectors;
        std::array<indexType,localVectorSize> currentRotFilledSize;
        currentRotFilledSize.fill(localVectorSize); // causes a new one to be added to currentLocalVectors when needed

        for(uint64_t currentCompIndex = 0; currentCompIndex < activebasisStates.size(); currentCompIndex++)
        {
            uint64_t currentBasisState = activebasisStates[currentCompIndex];

            std::array<std::pair<uint64_t,bool>,numberToFuse> initialLinks;
            indexType activeRotIdx = 0;
            indexType numberOfActiveRots = 0;
            bool allPositive = true;
            for (uint8_t idx = 0; idx < numberToFuse; idx++)
            {
                initialLinks[idx] = applyExcToBasisState(currentBasisState,rots[idx]);
                if (initialLinks[idx].first != currentBasisState)
                {
                    if (initialLinks[idx].second == false)
                    {
                        allPositive = false; // deduplication of basis states
                        break;
                    }
                    activeRotIdx |= 1<<idx;
                    ++numberOfActiveRots;
                }
            }
            //All not active
            if (numberOfActiveRots == 0 || allPositive == false)
                continue;

            indexType& currentMapFilledSize = currentRotFilledSize[activeRotIdx];
            if (currentMapFilledSize == localVectorSize)
            {   if (currentLocalVectors[activeRotIdx].size() != 0) // for the first iteration
                {
                    localVectorMap& filledMap = currentLocalVectors[activeRotIdx].back().first;
                    for (uint64_t idx = 0; idx < localVectorSize; idx++)
                        comp->compressIndex(filledMap[idx],filledMap[idx]);
                    totalRotcount += numberOfActiveRots*localVectorSize/2;
                }
                currentLocalVectors[activeRotIdx].emplace_back(); // add new std::pair<localVectorMap,signMap> to back of this series of rotations

                currentMapFilledSize = 0;
            }
            localVectorMap& currentMap = currentLocalVectors[activeRotIdx].back().first;
            signMap& currentSigns = currentLocalVectors[activeRotIdx].back().second;
            indexType rotIdx = 0;
            while(rotIdx < numberToFuse && (activeRotIdx & (1<<rotIdx)) == 0)
                ++rotIdx;
            assert(rotIdx < numberToFuse);

            *(currentMap.begin()+currentMapFilledSize) = currentBasisState;
            uint64_t currentSignsStep = numberOfActiveRots*currentMapFilledSize/2;
            fillCurrentMap<indexType,numberToFuse>(activeRotIdx,numberOfActiveRots,currentMap.begin()+currentMapFilledSize,currentSigns.begin()+currentSignsStep,rotIdx,0,initialLinks,rots);
            currentMapFilledSize += 1<<numberOfActiveRots;
        }
        for (indexType idx = 0; idx <  localVectorSize; idx++)
        {
            if (currentLocalVectors[idx].size() != 0) // for the first iteration
            {
                indexType filledSize = currentRotFilledSize[idx];
                totalRotcount += dot(idx,(uint64_t)-1,sizeof(idx)*8)*localVectorSize/2;
                localVectorMap& potentiallyFilledMap = currentLocalVectors[idx].back().first;
                if (filledSize < localVectorSize)
                    potentiallyFilledMap[filledSize] = (uint64_t)-1;

                //The last one will not have been compressed
                for (uint64_t idx = 0; idx < filledSize; idx++)
                    comp->compressIndex(potentiallyFilledMap[idx],potentiallyFilledMap[idx]);
            }
        }
        // std::vector<std::vector<size_t>> sortPerm = Benchmark_sortPermutation(currentLocalVectors);
        // myFusedAnsatz.emplace_back();
        // for (indexType idx = 0; idx < localVectorSize; idx++)
        // {
        //     Benchmark_applyPermutationAndStore(currentLocalVectors[idx],sortPerm[idx],myFusedAnsatz.back()[idx]);
        // }
        myFusedAnsatz.emplace_back(std::move(currentLocalVectors));
    }

    logger().log("FuseN rotcount",totalRotcount);
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < loopCount; loop++)
    {
        startVec.copy(ansatz->getStart());
        if constexpr (BraketWithTangentOfResult)
        {
            for (indexType i = 0; i < rotationPath.size(); i++)
                *(result[i]) = 0;

        }
        for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
        {
            std::array<numType,numberToFuse> sines;
            std::array<numType,numberToFuse> cosines;
            for (indexType idx = 0; idx < numberToFuse; idx++)
            {
                mysincos(rotationPath[i+idx].second,&sines[idx],&cosines[idx]);
            }


            if constexpr(numberToFuse == 2)
            {
                std::future<void> futs[3];
                futs[0] = threadpool::getInstance(NUM_CORES).queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    numType scratchSpace[localVectorSize];
                    numType scratchSpacetoBraKet[numberToFuse][localVectorSize];
                    const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][1];
                    for (size_t j = 0; j < currentLocalVectors.size(); j++)
                    {
                        const localVectorMap& currentMap = currentLocalVectors[j].first;
                        const signMap& currentSigns = currentLocalVectors[j].second;
                        indexType filledSize = localVectorSize;

                        for (indexType idx = 0; idx < localVectorSize; idx++)
                        {
                            if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint64_t)-1)
                            {
                                filledSize = idx;
                                break;
                            }
                            scratchSpace[idx] = startVec[currentMap[idx]];
                            if constexpr(BraketWithTangentOfResult)
                            {
                                scratchSpacetoBraKet[0][idx] = toBraKet[i][currentMap[idx]];
                            }
                        }
                        if (!((filledSize == 4) || (filledSize == 2))) __builtin_unreachable();



                        if (filledSize == 4)
                        {
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],2,scratchSpacetoBraKet,{result[i]});
                        }
                        else
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],1,scratchSpacetoBraKet,{result[i]});

                        //Restore scratch space
                        for (indexType idx = 0; idx < filledSize; idx++)
                            startVec[currentMap[idx]] = scratchSpace[idx];
                    }
                });
                futs[1] = threadpool::getInstance(NUM_CORES).queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    numType scratchSpace[localVectorSize];
                    numType scratchSpacetoBraKet[numberToFuse][localVectorSize];
                    const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][2];
                    for (size_t j = 0; j < currentLocalVectors.size(); j++)
                    {
                        const localVectorMap& currentMap = currentLocalVectors[j].first;
                        const signMap& currentSigns = currentLocalVectors[j].second;
                        indexType filledSize = localVectorSize;

                        for (indexType idx = 0; idx < localVectorSize; idx++)
                        {
                            if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint64_t)-1)
                            {
                                filledSize = idx;
                                break;
                            }
                            scratchSpace[idx] = startVec[currentMap[idx]];
                            if constexpr(BraketWithTangentOfResult)
                            {
                                scratchSpacetoBraKet[0][idx] = toBraKet[i+1][currentMap[idx]];
                            }
                        }
                        assert(filledSize != 0);
                        if (!((filledSize == 4) || (filledSize == 2))) __builtin_unreachable();

                        if (filledSize == 4)
                        {
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[1],&cosines[1],2,scratchSpacetoBraKet,{result[1+i]});
                        }
                        else
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[1],&cosines[1],1,scratchSpacetoBraKet,{result[1+i]});
                        //Restore scratch space
                        for (indexType idx = 0; idx < filledSize; idx++)
                            startVec[currentMap[idx]] = scratchSpace[idx];
                    }
                });
                futs[2] = threadpool::getInstance(NUM_CORES).queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    numType scratchSpace[localVectorSize];
                    numType scratchSpacetoBraKet[numberToFuse][localVectorSize];
                    const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][3];
                    for (size_t j = 0; j < currentLocalVectors.size(); j++)
                    {
                        const localVectorMap& currentMap = currentLocalVectors[j].first;
                        const signMap& currentSigns = currentLocalVectors[j].second;

                        for (indexType idx = 0; idx < localVectorSize; idx++)
                        {
                            scratchSpace[idx] = startVec[currentMap[idx]];
                            if constexpr(BraketWithTangentOfResult)
                            {
                                scratchSpacetoBraKet[0][idx] = toBraKet[i][currentMap[idx]];
                                scratchSpacetoBraKet[1][idx] = toBraKet[i+1][currentMap[idx]];
                            }
                        }

                        BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],1,scratchSpacetoBraKet,{result[i],result[i+1]});

                        //Restore scratch space
                        for (indexType idx = 0; idx < localVectorSize; idx++)
                            startVec[currentMap[idx]] = scratchSpace[idx];
                    }
                });
                futs[0].wait();
                futs[1].wait();
                futs[2].wait();
            }
            else if constexpr((numberToFuse == 3 || numberToFuse == 4 ) && !BraketWithTangentOfResult)
            {
                std::future<void> futs[4];
                threadpool& pool  = threadpool::getInstance(NUM_CORES);
                futs[0] = pool.queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    constexpr indexType theOnes[4] =  {0b0001,0b0010,0b0100,0b1000};
                    constexpr indexType theOnesOffset[4] = {0,1,2,3};
                    numType scratchSpace[1<<numberToFuse];
                    numType scratchSpacetoBraKet[numberToFuse][1<<numberToFuse];
                    for (indexType abc = 0; abc < 4; abc++)
                    {
                        if constexpr (numberToFuse == 3) if (!(abc < 3)) break;
                        const indexType rotIdx = theOnes[abc];
                        const indexType activeRot = theOnesOffset[abc];

                        const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                        if (currentLocalVectors.size() == 0)
                            continue;
                        for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                        {
                            const localVectorMap& currentMap = currentLocalVectors[j].first;
                            const signMap& currentSigns = currentLocalVectors[j].second;

                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                        scratchSpacetoBraKet[0][idx] = toBraKet[i+activeRot][currentMap[idx]];
                            }
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[activeRot],&cosines[activeRot],localVectorSize/2,scratchSpacetoBraKet,{result[i+activeRot]});
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }

                        {
                            const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                            const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;
                            indexType filledSize = localVectorSize;
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                if (currentMap[idx] == (uint64_t)-1)
                                {
                                    filledSize = idx;
                                    break;
                                }
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                    scratchSpacetoBraKet[0][idx] = toBraKet[i+activeRot][currentMap[idx]];
                            }

                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[activeRot],&cosines[activeRot],filledSize/2,scratchSpacetoBraKet,{result[i+activeRot]});
                            for (indexType idx = 0; idx < filledSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }
                    }
                });
                futs[1] = pool.queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    constexpr indexType theTwos[6] =  {0b0011,0b0101,0b0110,0b1001,0b1010,0b1100};
                    constexpr indexType theTwosOffset[6][2] = {{0,1},{0,2},{1,2},{0,3},{1,3},{2,3}};
                    numType scratchSpace[1<<numberToFuse];
                    numType scratchSpacetoBraKet[numberToFuse][1<<numberToFuse];
                    for (indexType abc = 0; abc < 6; abc++)
                    {
                        if constexpr (numberToFuse == 3) if (!(abc < 3)) break;

                        const indexType rotIdx = theTwos[abc];
                        const indexType activeRot0 = theTwosOffset[abc][0];
                        const indexType activeRot1 = theTwosOffset[abc][1];
                        const numType S[2] = {sines[activeRot0],sines[activeRot1]};
                        const numType C[2] = {cosines[activeRot0],cosines[activeRot1]};
                        realNumType* resultArr[2] = {result[activeRot0+i], result[activeRot1+i]}; // This could be a const array

                        const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                        if (currentLocalVectors.size() == 0)
                            continue;
                        for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                        {
                            const localVectorMap& currentMap = currentLocalVectors[j].first;
                            const signMap& currentSigns = currentLocalVectors[j].second;

                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                {
                                    scratchSpacetoBraKet[activeRot0][idx] = toBraKet[i+activeRot0][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot1][idx] = toBraKet[i+activeRot1][currentMap[idx]];
                                }
                            }

                            BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,localVectorSize/4,scratchSpacetoBraKet,resultArr);

                            for (indexType idx = 0; idx < localVectorSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }

                        {
                            const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                            const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;

                            indexType filledSize = localVectorSize;
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                if (currentMap[idx] == (uint64_t)-1)
                                {
                                    filledSize = idx;
                                    break;
                                }
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                {
                                    scratchSpacetoBraKet[activeRot0][idx] = toBraKet[i+activeRot0][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot1][idx] = toBraKet[i+activeRot1][currentMap[idx]];
                                }
                            }

                            BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,filledSize/4,scratchSpacetoBraKet,resultArr);
                            for (indexType idx = 0; idx < filledSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }
                    }
                });

                futs[2] = pool.queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                {
                    constexpr indexType theThrees[4] =          {0b0111,0b1011,0b1101,0b1110};
                    constexpr indexType theThreesOffset[6][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
                    numType scratchSpace[1<<numberToFuse];
                    numType scratchSpacetoBraKet[numberToFuse][1<<numberToFuse];
                    for (indexType abc = 0; abc < 4; abc++)
                    {
                        if constexpr (numberToFuse == 3) if (!(abc < 1)) break;
                        const indexType rotIdx = theThrees[abc];
                        const indexType activeRot0 = theThreesOffset[abc][0];
                        const indexType activeRot1 = theThreesOffset[abc][1];
                        const indexType activeRot2 = theThreesOffset[abc][2];

                        const numType S[3] = {sines[activeRot0],sines[activeRot1],sines[activeRot2]};
                        const numType C[3] = {cosines[activeRot0],cosines[activeRot1],cosines[activeRot2]};
                        realNumType* resultArr[3] = {result[activeRot0+i], result[activeRot1+i],result[activeRot2+i]}; // This could be a const array

                        const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                        if (currentLocalVectors.size() == 0)
                            continue;
                        for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                        {
                            const localVectorMap& currentMap = currentLocalVectors[j].first;
                            const signMap& currentSigns = currentLocalVectors[j].second;

                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                {
                                    scratchSpacetoBraKet[activeRot0][idx] = toBraKet[i+activeRot0][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot1][idx] = toBraKet[i+activeRot1][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot2][idx] = toBraKet[i+activeRot2][currentMap[idx]];
                                }
                            }
                            BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,localVectorSize/8,scratchSpacetoBraKet,resultArr);
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }

                        {
                            const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                            const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;
                            indexType filledSize = localVectorSize;
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                if (currentMap[idx] == (uint64_t)-1)
                                {
                                    filledSize = idx;
                                    break;
                                }
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                {
                                    scratchSpacetoBraKet[activeRot0][idx] = toBraKet[i+activeRot0][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot1][idx] = toBraKet[i+activeRot1][currentMap[idx]];
                                    scratchSpacetoBraKet[activeRot2][idx] = toBraKet[i+activeRot2][currentMap[idx]];
                                }
                            }

                            BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,filledSize/8,scratchSpacetoBraKet,resultArr);
                            for (indexType idx = 0; idx < filledSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }
                    }
                });
                if constexpr(numberToFuse == 4)
                {
                    futs[3] = pool.queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result]()
                    {
                        numType scratchSpace[1<<numberToFuse];
                        numType scratchSpacetoBraKet[numberToFuse][1<<numberToFuse];
                        constexpr indexType rotIdx = 0b1111;

                        const numType* S = &sines[0];
                        const numType* C = &cosines[0];

                        const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                        for (size_t j = 0; j < currentLocalVectors.size(); j++)
                        {
                            const localVectorMap& currentMap = currentLocalVectors[j].first;
                            const signMap& currentSigns = currentLocalVectors[j].second;

                            for (indexType idx = 0; idx < localVectorSize; idx++)
                            {
                                scratchSpace[idx] = startVec[currentMap[idx]];
                                if constexpr(BraketWithTangentOfResult)
                                {
                                    scratchSpacetoBraKet[0][idx] = toBraKet[i+0][currentMap[idx]];
                                    scratchSpacetoBraKet[1][idx] = toBraKet[i+1][currentMap[idx]];
                                    scratchSpacetoBraKet[2][idx] = toBraKet[i+2][currentMap[idx]];
                                    scratchSpacetoBraKet[3][idx] = toBraKet[i+3][currentMap[idx]];
                                }
                            }
                            BENCHMARK_rotate<indexType,4,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,1,scratchSpacetoBraKet,{result[i]});
                            for (indexType idx = 0; idx < localVectorSize; idx++)
                                startVec[currentMap[idx]] = scratchSpace[idx];
                        }
                    });
                    futs[3].wait();
                }
                futs[0].wait();
                futs[1].wait();
                futs[2].wait();


            }
            else
            {
                std::vector<std::future<void>> futs;
                threadpool& pool = threadpool::getInstance(NUM_CORES);
                //0b000 is always empty
                constexpr indexType stepSize = localVectorSize;localVectorSize < 4? 1 : localVectorSize/4;
                for (indexType activeRotIdxStart = 1; activeRotIdxStart < localVectorSize; activeRotIdxStart+=stepSize)
                {
                    indexType activeRotIdxEnd = (activeRotIdxStart+stepSize < localVectorSize ?activeRotIdxStart+stepSize : localVectorSize);
                    futs.push_back(pool.queueWork([&myFusedAnsatz,i,&startVec,&toBraKet,&sines,&cosines,&result,activeRotIdxStart,activeRotIdxEnd]()
                    {
                        for (indexType activeRotIdx = activeRotIdxStart; activeRotIdx < activeRotIdxEnd; activeRotIdx++)
                        {
                            numType scratchSpace[localVectorSize];
                            numType scratchSpacetoBraKet[numberToFuse][localVectorSize];
                            const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][activeRotIdx];
                            std::array<indexType,numberToFuse> activeRots; // array with valid data until numberOfActiveRotations storing the index of the active rotations
                            indexType numberOfActiveRotations = 0;


                            for (indexType rotIdx = 0; rotIdx < numberToFuse; rotIdx++)
                            {
                                if (activeRotIdx & (1<<rotIdx))
                                    activeRots[numberOfActiveRotations++] = rotIdx;

                            }

                            for (size_t j = 0; j < currentLocalVectors.size(); j++)
                            {
                                const localVectorMap& currentMap = currentLocalVectors[j].first;
                                const signMap& currentSigns = currentLocalVectors[j].second;
                                indexType filledSize = localVectorSize;

                                for (indexType idx = 0; idx < localVectorSize; idx++)
                                {
                                    if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint64_t)-1)
                                    {
                                        filledSize = idx;
                                        break;
                                    }
                                    scratchSpace[idx] = startVec[currentMap[idx]];
                                    if constexpr(BraketWithTangentOfResult)
                                    {
                                        for (indexType rotIdx = 0; rotIdx < numberOfActiveRotations; rotIdx++)
                                        {
                                            scratchSpacetoBraKet[rotIdx][idx] = toBraKet[i+activeRots[rotIdx]][currentMap[idx]];
                                        }
                                    }
                                }
                                assert(filledSize != 0);
                                assert(filledSize <= 1<<numberToFuse);
                                if (filledSize == 0) __builtin_unreachable();
                                if (filledSize > 1<<numberToFuse) __builtin_unreachable();

                                assert(numberOfActiveRotations != 0);
                                assert(numberOfActiveRotations <= numberToFuse);
                                if (numberOfActiveRotations == 0) __builtin_unreachable();
                                if (numberOfActiveRotations > numberToFuse) __builtin_unreachable();

                                switch (numberOfActiveRotations)
                                {
                                case 1:
                                    if constexpr (numberToFuse >= 1)
                                    {
                                        const numType S[1] = {sines[activeRots[0]]};
                                        const numType C[1] = {cosines[activeRots[0]]};
                                        realNumType* resultArr[1] = {result[activeRots[0]+i]};
                                        BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/2,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 2:
                                    if constexpr (numberToFuse >= 2)
                                    {
                                        const numType S[2] = {sines[activeRots[0]],sines[activeRots[1]]};
                                        const numType C[2] = {cosines[activeRots[0]],cosines[activeRots[1]]};
                                        realNumType* resultArr[2] = {result[activeRots[0]+i], result[activeRots[1]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/4,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 3:
                                    if constexpr (numberToFuse >= 3)
                                    {
                                        const numType S[3] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]]};
                                        const numType C[3] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]]};
                                        realNumType* resultArr[3] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/8,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 4:
                                    if constexpr (numberToFuse >= 4)
                                    {
                                        const numType S[4] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]]};
                                        const numType C[4] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]]};
                                        realNumType* resultArr[4] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,4,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/16,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 5:
                                    if constexpr (numberToFuse >= 5)
                                    {
                                        const numType S[5] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]]};
                                        const numType C[5] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]]};
                                        realNumType* resultArr[5] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,5,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/32,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 6:
                                    if constexpr (numberToFuse >= 6)
                                    {
                                        const numType S[6] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]],sines[activeRots[5]]};
                                        const numType C[6] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]],cosines[activeRots[5]]};
                                        realNumType* resultArr[6] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i],result[activeRots[5]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,6,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/64,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 7:
                                    if constexpr (numberToFuse >= 7)
                                    {
                                        const numType S[7] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]],sines[activeRots[5]],sines[activeRots[6]]};
                                        const numType C[7] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]],cosines[activeRots[5]],cosines[activeRots[6]]};
                                        realNumType* resultArr[7] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i],result[activeRots[5]+i],result[activeRots[6]+i]}; // This could be a const array
                                        BENCHMARK_rotate<indexType,7,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/128,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 8:
                                    if constexpr (numberToFuse >= 8)
                                    {
                                        numType S[8]; numType C[8]; realNumType* resultArr[8];
                                        for (indexType x = 0; x < 8; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,8,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/256,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 9:
                                    if constexpr (numberToFuse >= 9)
                                    {
                                        numType S[9]; numType C[9]; realNumType* resultArr[9];
                                        for (indexType x = 0; x < 9; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,9,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/512,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 10:
                                    if constexpr (numberToFuse >= 10)
                                    {
                                        numType S[10]; numType C[10]; realNumType* resultArr[10];
                                        for (indexType x = 0; x < 10; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,10,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/1024,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 11:
                                    if constexpr (numberToFuse >= 11)
                                    {
                                        numType S[11]; numType C[11]; realNumType* resultArr[11];
                                        for (indexType x = 0; x < 11; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,11,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/2048,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 12:
                                    if constexpr (numberToFuse >= 12)
                                    {
                                        numType S[12]; numType C[12]; realNumType* resultArr[12];
                                        for (indexType x = 0; x < 12; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,12,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/4096,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 13:
                                    if constexpr (numberToFuse >= 13)
                                    {
                                        numType S[13]; numType C[13]; realNumType* resultArr[13];
                                        for (indexType x = 0; x < 13; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,13,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/8192,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 14:
                                    if constexpr (numberToFuse >= 14)
                                    {
                                        numType S[14]; numType C[14]; realNumType* resultArr[14];
                                        for (indexType x = 0; x < 14; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,14,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/16384,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                case 15:
                                    if constexpr (numberToFuse >= 15)
                                    {
                                        numType S[15]; numType C[15]; realNumType* resultArr[15];
                                        for (indexType x = 0; x < 15; x++)
                                        {
                                            S[x] = sines[activeRots[x]];
                                            C[x] = cosines[activeRots[x]];
                                            resultArr[x] = result[activeRots[x]+i];
                                        }
                                        BENCHMARK_rotate<indexType,15,localVectorSize,BraketWithTangentOfResult>(
                                            scratchSpace,currentSigns.begin(),S,C,filledSize/32768,scratchSpacetoBraKet,resultArr);
                                        break;
                                    }
                                    static_assert(numberToFuse < 16, "Only up to 16 handled");
                                default:
                                    logger().log("Unreachable!");
                                    __builtin_trap();
                                    break;
                                }



                                //Restore scratch space
                                for (indexType idx = 0; idx < filledSize; idx++)
                                    startVec[currentMap[idx]] = scratchSpace[idx];
                            }
                        }
                    }));
                    for (auto& f : futs)
                        f.wait();
                }
            }
        }
        //Start now contains the full evolution

        //Do this loopcount times
    }



    compressor::deCompressVector<numType>(startVec,destVec,comp);

    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("FusedRotate" + std::to_string(numberToFuse) +" Time taken:",duration);
    if constexpr (!BraketWithTangentOfResult)
    {
        delete[] result;
        // delete toBraKet;
    }
}

void benchmarkRotate5(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec, int loopCount = 100)
{
    constexpr uint8_t numberToFuse = 2;
    constexpr uint8_t localVectorSize = 1<<numberToFuse;
    numType scratchSpace[localVectorSize];

    typedef std::array<bool,localVectorSize> signMap; //true means +ve
    typedef std::array<uint64_t,localVectorSize> localVectorMap;
    typedef std::vector<std::pair<localVectorMap,signMap>> localVectors;

    typedef std::vector<std::array<localVectors,3>> fusedAnsatz; //TODO magic number 3
    fusedAnsatz myFusedAnsatz;

    //preprocess
    assert(rotationPath.size() % numberToFuse == 0);

    vector<numType> startVec;
    startVec.copy(ansatz->getStart());

    std::vector<uint64_t> activebasisStates;
    activebasisStates.resize(startVec.size());
    size_t totalRotcount = 0;
    std::shared_ptr<compressor> comp;
    ansatz->getLie()->getCompressor(comp);
    assert(comp != nullptr);
    for (uint64_t i = 0; i < startVec.size(); i++)
    {
        comp->deCompressIndex(i,activebasisStates[i]);
    }
    for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
    {
        //Hard coding the 2 for now
        stateRotate::exc rot0;
        stateRotate::exc rot1;
        dynamic_cast<stateRotate*>(ansatz->getLie())->convertIdxToExc(rotationPath[i].first,rot0);
        dynamic_cast<stateRotate*>(ansatz->getLie())->convertIdxToExc(rotationPath[i+1].first,rot1);

        //Structure is
        /*
         * With R0 first
         *   R0│  R1 │     R2
         * 0─┐───┬──────────────
         * 1─┘───│─┬────────────
         * 2─┐───┴─│────────────
         * 3─┘─────┴────────────
         *
         *
         * Therefore:
         * Starting with R0.
         * 0->1 Via R0|0>
         * 0->2 Via R1|0>
         * 1->3 Via R1R0|0> = R0  R1|0> = R0|2>
         * So 2->3 via R0 and the space is consistent
         *
         *For this we need them to commute!
         *
         */
        //Assuming they commute...

        localVectors currentLocalVectors;
        localVectors currentRot0OnlyVectors;
        localVectors currentRot1OnlyVectors;

        localVectorMap rot0Only;
        signMap rot0OnlySigns;
        uint8_t rot0OnlyFilledSize = 0;

        localVectorMap rot1Only;
        signMap rot1OnlySigns;
        uint8_t rot1OnlyFilledSize = 0;



        for(uint64_t currentCompIndex = 0; currentCompIndex < activebasisStates.size(); currentCompIndex++)
        {
            // uint8_t filledSize = 0;
            localVectorMap currentMap;
            signMap currentSigns;

            currentSigns[0] = true;
            uint64_t rot0i = activebasisStates[currentCompIndex];
            auto rot0j = applyExcToBasisState(rot0i,rot0);
            auto rot1j = applyExcToBasisState(rot0i,rot1);

            if (rot0j.first == rot0i)
            {
                if (rot1j.first == rot0i)
                    continue;
                //else this is a rot1 only
                //Only accept if phase is positive. This we we can optimise that and deduplicate
                if (rot1j.second == false)
                    continue;
                if (rot1OnlyFilledSize == localVectorSize)
                {
                    for (uint64_t idx = 0; idx < localVectorSize; idx++)
                        comp->compressIndex(rot1Only[idx],rot1Only[idx]);
                    currentRot1OnlyVectors.push_back({rot1Only,rot1OnlySigns});
                    totalRotcount += 2;
                    rot1OnlyFilledSize = 0;
                }
                rot1Only[rot1OnlyFilledSize] = rot0i;
                rot1Only[rot1OnlyFilledSize+1] = rot1j.first;
                rot1OnlySigns[rot1OnlyFilledSize/2+2] = rot1j.second;
                rot1OnlyFilledSize +=2;
                continue;
            }
            //Each set of two comes up 4 times ij, kl & ji , lk, & kl , ij & lk, ji
            // Deduplicate by only accepting if ij has positive phase & ik has positive phase

            if (rot1j.first == rot0i)
            {
                //this is a rot0 only vector
                if (rot0j.second == false)
                    continue;
                if (rot0OnlyFilledSize == localVectorSize)
                {
                    for (uint64_t idx = 0; idx < localVectorSize; idx++)
                        comp->compressIndex(rot0Only[idx],rot0Only[idx]);
                    currentRot0OnlyVectors.push_back({rot0Only,rot0OnlySigns});
                    totalRotcount += 2;
                    rot0OnlyFilledSize = 0;
                }

                rot0Only[rot0OnlyFilledSize] = rot0i;
                rot0Only[rot0OnlyFilledSize+1] = rot0j.first;
                rot0OnlySigns[rot0OnlyFilledSize/2] = rot0j.second;
                rot0OnlyFilledSize += 2;
                continue;
            }

            if (rot0j.second == false || rot1j.second == false)
                continue;
            //ij has positive phase & ik has positive phase

            //This will be a filledSize = 4
            currentMap[0] = rot0i;
            currentMap[1] = rot0j.first;
            currentSigns[0] = rot0j.second;
            // filledSize = 2;

            currentMap[2] = rot1j.first;
            currentSigns[2] = rot1j.second;

            rot1j = applyExcToBasisState(currentMap[1],rot1);
            assert(rot1j.first != rot0i);

            currentMap[3] = rot1j.first;
            currentSigns[3] = rot1j.second;
            // filledSize = 4;

            rot0j = applyExcToBasisState(currentMap[2],rot0);

            assert(rot0j.first != currentMap[2]);
            assert(rot0j.first == currentMap[3]);
            currentSigns[1] = rot0j.second;

            //currentMap contains basisIndexes, need to contain compressedIndexes
            for (uint64_t idx = 0; idx < localVectorSize; idx++)
                comp->compressIndex(currentMap[idx],currentMap[idx]);


            currentLocalVectors.push_back({currentMap,currentSigns});
            totalRotcount += 4;
        }
        if (rot0OnlyFilledSize < localVectorSize)
            rot0Only[rot0OnlyFilledSize] = -1;
        if (rot0OnlyFilledSize != 0)
        {
            for (uint64_t idx = 0; idx < rot0OnlyFilledSize; idx++)
                comp->compressIndex(rot0Only[idx],rot0Only[idx]);
            currentRot0OnlyVectors.push_back({rot0Only,rot0OnlySigns});
            totalRotcount += 2;
        }

        if (rot1OnlyFilledSize < localVectorSize)
            rot1Only[rot1OnlyFilledSize] = -1;
        if (rot1OnlyFilledSize != 0)
        {
            for (uint64_t idx = 0; idx < rot1OnlyFilledSize; idx++)
                comp->compressIndex(rot1Only[idx],rot1Only[idx]);
            currentRot1OnlyVectors.push_back({rot1Only,rot1OnlySigns});
            totalRotcount += 2;
        }

        myFusedAnsatz.push_back({currentRot0OnlyVectors,currentRot1OnlyVectors,currentLocalVectors});
    }
    logger().log("FuseN rotcount",totalRotcount);
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < loopCount; loop++)
    {
        startVec.copy(ansatz->getStart());
        for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
        {
            numType S1;
            numType S2;
            numType C1;
            numType C2;
            mysincos(rotationPath[i].second,&S1,&C1);
            mysincos(rotationPath[i+1].second,&S2,&C2);


            //rot0 only
            for (size_t j = 0; j < myFusedAnsatz[i/numberToFuse][0].size(); j++)
            {
                const localVectorMap& currentMap = myFusedAnsatz[i/numberToFuse][0][j].first;
                const signMap& currentSigns = myFusedAnsatz[i/numberToFuse][0][j].second;
                uint8_t filledSize = localVectorSize;
                for (uint8_t idx = 0; idx < localVectorSize; idx++)
                {
                    if (j == myFusedAnsatz[i/numberToFuse][0].size()-1)
                    {
                        if (currentMap[idx] == (uint64_t)-1)
                        {
                            filledSize = j;
                            break;
                        }
                    }
                    uint64_t vecIdx = currentMap[idx];
                    assert(vecIdx < startVec.size());
                    scratchSpace[idx] = startVec[vecIdx];
                }
                assert(filledSize == 2 || filledSize == 4);
                //scratchSpace now good
                //First rotation

                numType temp1 = scratchSpace[0];
                numType temp2 = scratchSpace[1];
                scratchSpace[0] = (currentSigns[0] ? 1 : -1)*temp2*S1 + temp1 * C1;
                scratchSpace[1] = -(currentSigns[0] ? 1 : -1)*temp1*S1 + temp2 * C1;
                if (filledSize == 4)
                {
                    temp1 = scratchSpace[2];
                    temp2 = scratchSpace[3];
                    scratchSpace[2] = (currentSigns[1] ? 1 : -1)*temp2*S1 + temp1 * C1;
                    scratchSpace[3] = -(currentSigns[1] ? 1 : -1)*temp1*S1 + temp2 * C1;
                }

                //Restore scratch space
                for (uint8_t idx = 0; idx < filledSize; idx++)
                    startVec[currentMap[idx]] = scratchSpace[idx];
            }
            //rot1 only
            for (size_t j = 0; j < myFusedAnsatz[i/numberToFuse][1].size(); j++)
            {
                const localVectorMap& currentMap = myFusedAnsatz[i/numberToFuse][1][j].first;
                const signMap& currentSigns = myFusedAnsatz[i/numberToFuse][1][j].second;
                uint8_t filledSize = localVectorSize;

                for (uint8_t idx = 0; idx < localVectorSize; idx++)
                {
                    if (j == myFusedAnsatz[i/numberToFuse][1].size()-1)
                    {
                        if (currentMap[idx] == (uint64_t)-1)
                        {
                            filledSize = j;
                            break;
                        }
                    }
                    scratchSpace[idx] = startVec[currentMap[idx]];
                }

                assert(filledSize == 2 || filledSize == 4);
                //Second rotation
                numType temp1 = scratchSpace[0];
                numType temp2 = scratchSpace[1];
                scratchSpace[0] = (currentSigns[2] ? 1 : -1)*temp2*S2 + temp1 * C2;
                scratchSpace[1] = -(currentSigns[2] ? 1 : -1)*temp1*S2 + temp2 * C2;
                if (filledSize == 4)
                {
                    temp1 = scratchSpace[2];
                    temp2 = scratchSpace[3];
                    scratchSpace[2] = (currentSigns[3] ? 1 : -1)*temp2*S2 + temp1 * C2;
                    scratchSpace[3] = -(currentSigns[3] ? 1 : -1)*temp1*S2 + temp2 * C2;
                }

                //Restore scratch space
                for (uint8_t idx = 0; idx < filledSize; idx++)
                    startVec[currentMap[idx]] = scratchSpace[idx];
            }
            //rot0 and rot1
            for (size_t j = 0; j < myFusedAnsatz[i/numberToFuse][2].size(); j++)
            {
                const localVectorMap& currentMap = myFusedAnsatz[i/numberToFuse][2][j].first;
                const signMap& currentSigns = myFusedAnsatz[i/numberToFuse][2][j].second;

                for (uint8_t idx = 0; idx < localVectorSize; idx++)
                    scratchSpace[idx] = startVec[currentMap[idx]];

                //scratchSpace now good
                //First rotation

                numType temp1 = scratchSpace[0];
                numType temp2 = scratchSpace[1];
                scratchSpace[0] = (currentSigns[0] ? 1 : -1)*temp2*S1 + temp1 * C1;
                scratchSpace[1] = -(currentSigns[0] ? 1 : -1)*temp1*S1 + temp2 * C1;

                temp1 = scratchSpace[2];
                temp2 = scratchSpace[3];
                scratchSpace[2] = (currentSigns[1] ? 1 : -1)*temp2*S1 + temp1 * C1;
                scratchSpace[3] = -(currentSigns[1] ? 1 : -1)*temp1*S1 + temp2 * C1;

                //Second rotation
                temp1 = scratchSpace[0];
                temp2 = scratchSpace[2];
                scratchSpace[0] = (currentSigns[2] ? 1 : -1)*temp2*S2 + temp1 * C2;
                scratchSpace[2] = -(currentSigns[2] ? 1 : -1)*temp1*S2 + temp2 * C2;

                temp1 = scratchSpace[1];
                temp2 = scratchSpace[3];
                scratchSpace[1] = (currentSigns[3] ? 1 : -1)*temp2*S2 + temp1 * C2;
                scratchSpace[3] = -(currentSigns[3] ? 1 : -1)*temp1*S2 + temp2 * C2;

                //Restore scratch space
                for (uint8_t idx = 0; idx < localVectorSize; idx++)
                    startVec[currentMap[idx]] = scratchSpace[idx];
            }

        }
        //Start now contains the full evolution

        //Do this loopcount times
    }



    compressor::deCompressVector<numType>(startVec,destVec,comp);

    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("FusedRotate Time taken:",duration);

}

void benchmarkRotate4(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec, int loopCount = 100)
{
    constexpr uint8_t numberToFuse = 2;
    constexpr uint8_t localVectorSize = 1<<numberToFuse;
    numType scratchSpace[localVectorSize];

    typedef std::array<bool,localVectorSize> signMap; //true means +ve
    typedef std::array<uint64_t,localVectorSize> localVectorMap;
    // typedef std::array<std::pair<uint8_t,uint8_t>,localVectorSize> localSparseMatrix;
    typedef std::vector<std::pair<localVectorMap,signMap>> localVectors;
    std::vector<uint64_t> onlyrot1Start;
    typedef std::vector<localVectors> fusedAnsatz;
    fusedAnsatz myFusedAnsatz;
    //preprocess
    assert(rotationPath.size() % numberToFuse == 0);
    for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
    {
        //Hard coding the 2 for now
        matrixType* rot0 = ansatz->getLie()->getLieAlgebraMatrix(rotationPath[i].first);
        matrixType* rot1 = ansatz->getLie()->getLieAlgebraMatrix(rotationPath[i+1].first);



        std::list<uint64_t> rot0is(rot0->iItBegin(),rot0->iItEnd());
        std::list<uint64_t> rot0js(rot0->jItBegin(),rot0->jItEnd());

        std::list<uint64_t> rot1is(rot1->iItBegin(),rot1->iItEnd());
        std::list<uint64_t> rot1js(rot1->jItBegin(),rot1->jItEnd());




        //Structure is
        /*
         * With R0 first
         *   R0│  R1 │     R2
         * 0─┐───┬──────────────
         * 1─┘───│─┬────────────
         * 2─┐───┴─│────────────
         * 3─┘─────┴────────────
         *
         *
         * Therefore:
         * Starting with R0.
         * 0->1 Via R0|0>
         * 0->2 Via R1|0>
         * 1->3 Via R1R0|0> = R0  R1|0> = R0|2>
         * So 2->3 via R0 and the space is consistent
         *
         *For this we need them to commute!
         *
         */
        //Assuming they commute...
        uint8_t filledSize = 0;
        localVectors currentLocalVectors;
        while(rot0is.size() > 0 && rot0js.size() > 0 /*&& rot1is.size() > 0 && rot1js.size() > 0*/)
        {
            auto rot0iIt = rot0is.begin();
            auto rot0jIt = rot0js.begin();
            if (rot0iIt == rot0is.end() || rot0jIt == rot0js.end())
            {
                assert(rot0iIt == rot0is.end() && rot0jIt == rot0js.end());
                break;
            }

            localVectorMap currentMap;
            signMap currentSigns;
            {
                currentSigns[0] = true;

                currentMap[0] = *rot0iIt;
                currentMap[1] = *rot0jIt;
                filledSize = 2;

                //This is super expensive!
                auto rot1i = rot1is.begin();
                auto rot1j = rot1js.begin();

                while (rot1i != rot1is.end() && *rot1i != currentMap[0])
                {
                    ++rot1i;
                    ++rot1j;
                }
                decltype(rot1is.begin()) a;
                decltype(rot1js.begin()) b;

                if (rot1i == rot1is.end())
                {
                    assert(rot1j == rot1js.end());
                    //Try and find the other way around. Since we dont store duplicates
                    rot1i = rot1is.begin();
                    rot1j = rot1js.begin();

                    while (rot1j != rot1js.end() && *rot1j != currentMap[0])
                    {
                        ++rot1i;
                        ++rot1j;
                    }
                    if (rot1j == rot1js.end())
                    {
                        //This isnt shared
                        //Do cleanup
                        rot0is.erase(rot0iIt);

                        rot0js.erase(rot0jIt);
                        goto store;
                    }
                    a = rot1i;
                    b = rot1j;
                    currentSigns[2] = false;
                    //TODO store the sign change!
                }
                else
                {
                    a = rot1i;
                    b = rot1j;
                    currentSigns[2] = true;
                }


                assert(a != rot1is.end());
                assert(b != rot1js.end());
                //done with a and b

                decltype(rot1is.begin()) c;
                decltype(rot1js.begin()) d;

                rot1i = rot1is.begin();
                rot1j = rot1js.begin();

                while (rot1i != rot1is.end() && *rot1i != currentMap[1])
                {
                    ++rot1i;
                    ++rot1j;
                }
                if (rot1i == rot1is.end())
                {
                    assert(rot1j == rot1js.end());
                    //Try and find the other way around. Since we dont store duplicates
                    rot1i = rot1is.begin();
                    rot1j = rot1js.begin();

                    while (rot1j != rot1js.end() && *rot1j != currentMap[1])
                    {
                        ++rot1i;
                        ++rot1j;
                    }
                    if (rot1j == rot1js.end())
                    {
                        //This isnt shared
                        assert(false);
                        rot0is.erase(rot0iIt);

                        rot0js.erase(rot0jIt);

                        rot1is.erase(a);
                        rot1is.erase(c);

                        rot1js.erase(b);
                        rot1js.erase(d);
                        goto store;
                    }
                    c = rot1i;
                    d = rot1j;
                    currentSigns[3] = false;
                    //TODO store the sign change!
                }
                else
                {
                    c = rot1i;
                    d = rot1j;
                    currentSigns[3] = true;
                }
                filledSize = 4;
                assert(c != rot1is.end());
                assert(d != rot1js.end());

                currentMap[2] = *b == currentMap[0] ? *a : *b; //links with currentMap[0]
                currentMap[3] = *d == currentMap[1] ? *c : *d; //links with currentMap[1]

                auto e = std::find(rot0is.begin(),rot0is.end(),currentMap[2]);
                auto f = std::find(rot0js.begin(),rot0js.end(),currentMap[3]);

                if (e == rot0is.end())
                {
                    assert(f == rot0js.end());
                    e = std::find(rot0is.begin(),rot0is.end(),currentMap[3]);
                    f = std::find(rot0js.begin(),rot0js.end(),currentMap[2]);
                    currentSigns[1] = false;
                }
                else
                    currentSigns[1] = true;

                assert(e != rot0is.end());
                assert(f != rot0js.end());


                rot0is.erase(rot0iIt);
                rot0is.erase(e);

                rot0js.erase(rot0jIt);
                rot0js.erase(f);

                rot1is.erase(a);
                rot1is.erase(c);

                rot1js.erase(b);
                rot1js.erase(d);
            }
        store:
            if (filledSize < 4)
            {
                currentMap[filledSize] = -1; // denote end of local vector
            }
            currentLocalVectors.push_back({currentMap,currentSigns});
        }
        localVectorMap currentMap;
        signMap currentSigns = {true,true,true,true};
        if (rot1is.size() > 0 && rot1js.size() > 0)
            onlyrot1Start.push_back(currentLocalVectors.size());
        else
            onlyrot1Start.push_back(-1);

        while (rot1is.size() > 0 && rot1js.size() > 0)
        {
            auto rot1i = rot1is.begin();
            auto rot1j = rot1js.begin();
            currentMap[0] = *rot1i;
            currentMap[1] = *rot1j;
            rot1i = rot1is.erase(rot1i);
            rot1j = rot1js.erase(rot1j);
            if (rot1i == rot1is.end())
            {
                assert(rot1j == rot1js.end());
                currentMap[2] = -1;
                currentLocalVectors.push_back({currentMap,currentSigns});
                break;
            }
            rot1i = rot1is.begin();
            rot1j = rot1js.begin();
            currentMap[2] = *rot1i;
            currentMap[3] = *rot1j;
            rot1i = rot1is.erase(rot1i);
            rot1j = rot1js.erase(rot1j);
            currentLocalVectors.push_back({currentMap,currentSigns});

        }
        assert(rot0is.size() == 0 && rot0js.size() == 0 && rot1is.size() == 0 && rot1js.size() == 0);
        myFusedAnsatz.push_back(currentLocalVectors);
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    vector<numType> start;
    for (int loop = 0; loop < loopCount; loop++)
    {
        start.copy(ansatz->getStart());
        for (size_t i = 0; i < rotationPath.size(); i+=numberToFuse)
        {
            numType S1;
            numType S2;
            numType C1;
            numType C2;
            mysincos(rotationPath[i].second,&S1,&C1);
            mysincos(rotationPath[i+1].second,&S2,&C2);
            uint64_t onlyRot1StartIdx = onlyrot1Start[i/numberToFuse];
            size_t j = 0;
            for ( ;j < myFusedAnsatz[i/numberToFuse].size() && j < onlyRot1StartIdx ; j++)
            {
                const localVectorMap& currentMap = myFusedAnsatz[i/numberToFuse][j].first;
                const signMap& currentSigns = myFusedAnsatz[i/numberToFuse][j].second;

                uint8_t filledSize = 0;
                for (; filledSize < localVectorSize; filledSize++)
                {
                    if (currentMap[filledSize] == (uint64_t)-1)
                        break;
                    scratchSpace[filledSize] = start[currentMap[filledSize]];
                }
                //scratchSpace now good
                //First rotation
                if (filledSize == 1 || filledSize == 3)
                {
                    assert(false);
                    __builtin_unreachable();
                }

                if (filledSize == 0)
                    continue;

                numType temp1 = scratchSpace[0];
                numType temp2 = scratchSpace[1];
                scratchSpace[0] = (currentSigns[0] ? 1 : -1)*temp2*S1 + temp1 * C1;
                scratchSpace[1] = -(currentSigns[0] ? 1 : -1)*temp1*S1 + temp2 * C1;
                if (filledSize == 2)
                    goto restore;
                temp1 = scratchSpace[2];
                temp2 = scratchSpace[3];
                scratchSpace[2] = (currentSigns[1] ? 1 : -1)*temp2*S1 + temp1 * C1;
                scratchSpace[3] = -(currentSigns[1] ? 1 : -1)*temp1*S1 + temp2 * C1;

                //Second rotation
                temp1 = scratchSpace[0];
                temp2 = scratchSpace[2];
                scratchSpace[0] = (currentSigns[2] ? 1 : -1)*temp2*S2 + temp1 * C2;
                scratchSpace[2] = -(currentSigns[2] ? 1 : -1)*temp1*S2 + temp2 * C2;

                temp1 = scratchSpace[1];
                temp2 = scratchSpace[3];
                scratchSpace[1] = (currentSigns[3] ? 1 : -1)*temp2*S2 + temp1 * C2;
                scratchSpace[3] = -(currentSigns[3] ? 1 : -1)*temp1*S2 + temp2 * C2;
                restore:
                //Restore scratch space
                for (uint8_t k = 0; k < filledSize; k++)
                {
                    start[myFusedAnsatz[i/numberToFuse][j].first[k]] = scratchSpace[k];
                }
            }
            //finish of the only rot1 elements
            for ( ;j < myFusedAnsatz[i/numberToFuse].size(); j++)
            {
                const localVectorMap& currentMap = myFusedAnsatz[i/numberToFuse][j].first;
                uint8_t filledSize = 0;
                for (; filledSize < localVectorSize; filledSize++)
                {
                    if (currentMap[filledSize] == (uint64_t)-1)
                        break;
                    scratchSpace[filledSize] = start[currentMap[filledSize]];
                }
                assert(filledSize == 2 || filledSize == 4);
                numType temp1 = scratchSpace[0];
                numType temp2 = scratchSpace[1];
                scratchSpace[0] = temp2*S2 + temp1 * C2;
                scratchSpace[1] = -temp1*S2 + temp2 * C2;

                if (filledSize == 4)
                {
                    temp1 = scratchSpace[2];
                    temp2 = scratchSpace[3];
                    scratchSpace[2] = temp2*S2 + temp1 * C2;
                    scratchSpace[3] = -temp1*S2 + temp2 * C2;
                }
                //Restore scratch space
                for (uint8_t k = 0; k < filledSize; k++)
                {
                    start[myFusedAnsatz[i/numberToFuse][j].first[k]] = scratchSpace[k];
                }
            }

        }
        //Start now contains the full evolution

        //Do this loopcount times
    }

    std::shared_ptr<compressor> comp;
    if (ansatz->getLie()->getCompressor(comp))
    {
        compressor::deCompressVector<numType>(start,destVec,comp);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("FusedRotate Time taken:",duration);

}

#endif


void benchmarkRotate2(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rotationPath, vector<numType>& destVec)
{

    const vector<numType>& s = ansatz->getStart();
    vector<numType> startVec;
    {
        std::shared_ptr<compressor> comp;
        if (s.getIsCompressed(comp))
            compressor::deCompressVector<numType>(s,startVec,comp);
        else
            startVec.copy(s);
    }


    auto lie = ansatz->getLie();
    std::vector<bool> indexActive;
    indexActive.assign(startVec.size(),false);
    uint64_t allOnes = -1;
    for (uint64_t i = 0; i < indexActive.size(); i++)
    {
        if (dot(i,allOnes,32) == 10)
            indexActive[i] = true;
    }

    std::vector<std::vector<uint64_t>> iGenerators;
    std::vector<std::vector<uint64_t>> jGenerators;
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
            std::vector<uint64_t> intoIs;
            std::vector<uint64_t> intoJs;

            std::vector<numType> data;
            auto iIt = rotationGenerator.iItBegin();
            auto jIt = rotationGenerator.jItBegin();

            for (auto d = rotationGenerator.begin(); d < rotationGenerator.end(); d++)
            {
                uint64_t iIdx = *iIt;
                uint64_t jIdx = *jIt;
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
    uint64_t activeCount = 0;
    for (uint64_t i = 0; i < indexActive.size(); i++)
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
    for (uint64_t rpCount = 0; rpCount < iGenerators.size(); rpCount++)
    {
        for (uint64_t i = 0; i < iGenerators[rpCount].size(); i++)
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
        for (uint64_t i = 0; i < startVec.size(); i++)
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

        for (uint64_t i = 0; i < src.size(); i++)
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

    // const ansatz::rotationElement& rp = rotationPath[0];
    // const matrixType &rotationGenerator = *ansatz->getLie()->getLieAlgebraMatrix(rp.first);
    // const sparseMatrix<numType,numType>& lhs = rotationGenerator;
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
void benchmarkApplyHamMyVectorise(vector<numType>& dest, const sparseMatrix<realNumType,numType>& Ham, int loopcount = 100)
{
    Matrix<numType> BlockedMat(dest.size(),loopcount);
    Matrix<numType> temp;
    for (size_t i =0; i < dest.size();i++)
    {
        for (int j = 0; j < loopcount; j++)
        {
            BlockedMat.at(i,j) = dest[i];
        }
    }
    std::shared_ptr<compressor> comp;
    bool isCompressed = dest.getIsCompressed(comp);
    temp.resize(dest.size(),loopcount,isCompressed,comp);

    auto startTime = std::chrono::high_resolution_clock::now();
    auto iItBegin = Ham.iItBegin();
    auto iItEnd = Ham.iItEnd();
    auto jItBegin = Ham.jItBegin();
    auto dataBegin = Ham.begin();
    auto work = [&](decltype(Ham.iItBegin()) iItBegin, decltype(Ham.iItBegin()) jItBegin, decltype(Ham.begin()) dataBegin,decltype(Ham.iItBegin()) iItEnd)
    {
        while (iItBegin != iItEnd)
        {
#pragma GCC unroll 8
#pragma GCC ivdep
            for (int k = 0; k < loopcount; k++)
            {
                temp.at(*iItBegin,k) += BlockedMat.at(*jItBegin,k)* *dataBegin;
            }
            ++iItBegin;
            ++jItBegin;
            ++dataBegin;
        }
    };
    threadpool& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futs;
    int stepSize = (iItEnd - iItBegin)/1;
    futs.reserve(stepSize +1);
    while (iItBegin < iItEnd)
    {
        auto end = std::min(iItBegin+stepSize,iItEnd);
        futs.push_back(pool.queueWork([=,&work](){work(iItBegin,jItBegin,dataBegin,end);}));
        iItBegin += stepSize;
        jItBegin += stepSize;
        dataBegin += stepSize;
    }
    for (auto& f : futs)
        f.wait();

    auto endTime = std::chrono::high_resolution_clock::now();

    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Apply Hamiltonian Without Eigen time taken:",duration);
}

void benchmarkApplyHamEigenDirect(const vector<numType>& dest,const  Eigen::SparseMatrix<double,Eigen::RowMajor>& HamEm, int loopcount = 100)
{
    Matrix<numType> BlockedMat(dest.size(),loopcount);
    Matrix<numType> temp;
    for (size_t i =0; i < dest.size();i++)
    {
        for (int j = 0; j < loopcount; j++)
        {
            BlockedMat.at(i,j) = dest[i];
        }
    }
    std::shared_ptr<compressor> comp;
    bool isCompressed = dest.getIsCompressed(comp);
    temp.resize(dest.size(),loopcount,isCompressed,comp);

    auto startTime = std::chrono::high_resolution_clock::now();
    Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> BlockedEm(&BlockedMat.at(0,0),BlockedMat.m_iSize,BlockedMat.m_jSize);
    Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> temp3(&temp.at(0,0),temp.m_iSize,temp.m_jSize);
    temp3.noalias() = HamEm*BlockedEm;

    auto endTime = std::chrono::high_resolution_clock::now();

    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Apply Hamiltonian With Eigen no copy time taken:",duration);
}

void benchmarkApplyHamEigenDirect2(const vector<numType>& dest,const  Eigen::SparseMatrix<double,Eigen::ColMajor> HamEm, int loopcount = 100)
{
    Matrix<numType> BlockedMat(loopcount,dest.size());
    Matrix<numType> temp;
    for (size_t i =0; i < dest.size();i++)
    {
        for (int j = 0; j < loopcount; j++)
        {
            BlockedMat.at(j,i) = dest[i];
        }
    }
    std::shared_ptr<compressor> comp;
    bool isCompressed = dest.getIsCompressed(comp);
    temp.resize(loopcount,dest.size(),isCompressed,comp);

    auto startTime = std::chrono::high_resolution_clock::now();
    Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::ColMajor>,Eigen::Aligned32> BlockedEm(&BlockedMat.at(0,0),BlockedMat.m_iSize,BlockedMat.m_jSize);
    Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::ColMajor>,Eigen::Aligned32> temp3(&temp.at(0,0),temp.m_iSize,temp.m_jSize);
    temp3.noalias() = BlockedEm * HamEm;

    auto endTime = std::chrono::high_resolution_clock::now();

    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Apply Hamiltonian2 With Eigen no copy time taken:",duration);
}

void benchmarkApplyHam(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> __attribute__ ((unused))rp, sparseMatrix<realNumType,numType>& Ham, int loopcount = 100)
{
    vector<numType> dest;
    vector<numType> temp;
    dest.copy(ansatz->getVec());
    auto startTime = std::chrono::high_resolution_clock::now();
    for (int i =0; i < loopcount;i++)
    {
        Ham.multiply(dest,temp);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Apply Hamiltonian time taken:",duration);

    //Now do it with Eigen
    sparseMatrix<realNumType,numType>::EigenSparseMatrix HamEm = Ham;

    vector<numType>::EigenVector VecEm = dest;
    vector<numType>::EigenVector temp2;
    Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> BlockedEm(VecEm.rows(),loopcount);
    Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> temp3;


    for (int i =0; i < loopcount;i++)
    {
        BlockedEm.col(i) = VecEm;
    }


    startTime = std::chrono::high_resolution_clock::now();
    temp3.noalias() = HamEm*BlockedEm;
    endTime = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Apply Hamiltonian With Eigen time taken:",duration);

    // benchmarkApplyHamMyVectorise(dest,Ham,loopcount);
    benchmarkApplyHamEigenDirect(dest,HamEm,loopcount);
    benchmarkApplyHamEigenDirect2(dest,HamEm,loopcount);



}
void benchmark(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rp, std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham,
               sparseMatrix<realNumType,numType>::EigenSparseMatrix& compressMatrix, sparseMatrix<realNumType,numType>::EigenSparseMatrix &m_deCompressMatrix)
{
    vector<numType> dest1;
    vector<numType> dest2;
    vector<numType> dest3;
    vector<numType> dest4;
    vector<numType> dest5;
    vector<numType> dest6;
    benchmarkRotate3(ansatz,rp,dest1,1);
    const vector<numType>& endResult = ansatz->getVec();
    std::shared_ptr<compressor> comp;
    bool isCompressed = endResult.getIsCompressed(comp);

    vector<realNumType> result(rp.size());
    vector<realNumType> resultExpt(rp.size());
    Matrix<numType> toBrakets;
    toBrakets.resize(rp.size(),endResult.size(),isCompressed,comp);
    std::vector<vector<numType>> lotsOfVectors;

    realNumType** resultPtrs = new realNumType*[rp.size()];
    for (size_t i = 0; i < rp.size(); i++)
    {
        resultPtrs[i] = &(result[i]);
        toBrakets.getJVectorView(i).copy(endResult);
        lotsOfVectors.emplace_back();
        lotsOfVectors.back().copy(endResult);

    }


    // benchmarkApplyHam(ansatz,rp,Ham,1000);
    // delete[] resultPtrs;
    // return;

    // size_t naiveRotCount = 0;
    // for (auto& r : rp)
    // {
    //     naiveRotCount += ansatz->getLie()->getLieAlgebraMatrix(r.first)->getiSize();
    // }
    // logger().log("Naive rot count:", naiveRotCount);
    // // benchmarkRotate5(ansatz,rp,dest2,40);
    // // benchmarkRotate4(ansatz,rp,dest2,1);
    auto __attribute__ ((unused))runParallel = [&](void (*func)(stateAnsatz*, const std::vector<ansatz::rotationElement>&, vector<numType>&, int,
                                        const vector<numType>*, realNumType**), int jobs, int repeat)
    {
        realNumType** resultPtrsl = new realNumType*[rp.size()];
        vector<realNumType> resultl(rp.size());
        for (size_t i = 0; i < rp.size(); i++)
            resultPtrsl[i] = &(resultl[i]);


        auto startTime = std::chrono::high_resolution_clock::now();
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        std::vector<std::future<void>> futs;
        futs.reserve(jobs);
        for (int i = 0; i < jobs; i++)
        {
            if (i == 0)
                futs.push_back(pool.queueWork([&](){vector<numType> dest; func(ansatz,rp,dest,repeat,lotsOfVectors.data(),resultPtrs);}));
            else
                futs.push_back(pool.queueWork([&](){vector<numType> dest; func(ansatz,rp,dest,repeat,lotsOfVectors.data(),resultPtrsl);}));
        }

        for (auto& f : futs)
            f.wait();

        auto endTime = std::chrono::high_resolution_clock::now();
        long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
        logger().log("Combined Time taken:",duration);
        delete[] resultPtrsl;
    };

    // runParallel(benchmarkRotateFuseN<2>);
    // runParallel(benchmarkRotateFuseN<4>);
    // runParallel(benchmarkRotateFuseNLarge<12>);
    // runParallel(benchmarkRotate3);


    //Compute expected Result

    auto __attribute__ ((unused))computeExpectedBraket = [](stateAnsatz* ansatz, const std::vector<ansatz::rotationElement>& rp, vector<numType>&, int repeats,
                                     const vector<numType>* vecPtr, realNumType** resultStore)
    {
        const vector<numType>* vecPtrBKP = vecPtr;
        for (int i =0; i < repeats; i++)
        {
            vector<numType> start;
            start.copy(ansatz->getStart());
            vecPtr = vecPtrBKP;
            realNumType** ptr = resultStore;
            for (auto& r: rp)
            {
                realNumType S;
                realNumType C;
                mysincos(r.second,&S,&C);

                ansatz->getLie()->getLieAlgebraMatrix(r.first)->rotateAndBraketWithTangentOfResult(S,C,start,start,*vecPtr++,**ptr++);
            }
        }
    };
    // runParallel(computeExpectedBraket,4,40);
    // resultExpt.copy(result);
#ifndef useComplex
    /*
    runParallel(benchmarkRotateFuseN<uint8_t,2,true>,4,40);
    runParallel(benchmarkRotateFuseN<uint8_t,4,true>,4,40);
    runParallel(benchmarkRotateFuseN<uint8_t,6,true>,4,40);
    runParallel(benchmarkRotateFuseN<uint16_t,12,true>,4,40);
    // realNumType exptMag = resultExpt.dot(resultExpt);
    // logger().log("resultAccuracy",result.dot(resultExpt)/exptMag);

    logger().log("No Braket");
    runParallel(benchmarkRotateFuseN<uint8_t,2,false>,4,40);
    runParallel(benchmarkRotateFuseN<uint8_t,4,false>,4,40);
    runParallel(benchmarkRotateFuseN<uint8_t,6,false>,4,40);
    runParallel(benchmarkRotateFuseN<uint16_t,12,false>,4,40);
    runParallel(benchmarkRotate,4,40);

    // benchmarkRotateFuseN<uint8_t,2,false>(ansatz,rp,dest3,40);
    // benchmarkRotateFuseN<uint8_t,4,false>(ansatz,rp,dest4,40);
    // logger().log("dest1 normSq",dest1.dot(dest1));
    // logger().log("dest3 dot dest1 ",dest3.dot(dest1));
    // logger().log("dest4 dot dest1 ",dest4.dot(dest1));
    // // benchmarkRotateFuseN<5>(ansatz,rp,dest5,40);
    // benchmarkRotateFuseNLarge<12>(ansatz,rp,dest5,40);
    // dest5.copy(dest2);
    // benchmarkRotateFuseN<6>(ansatz,rp,dest6,40);
    // benchmarkRotateFuseNLarge<15,uint16_t>(ansatz,rp,dest6,40);

    // dest6.copy(dest2);*/
    logger().log("No Parallel");
    // benchmarkRotateFuseN<uint8_t,2,false>(ansatz,rp,dest2,40);
    // benchmarkRotateFuseN<uint8_t,4,false>(ansatz,rp,dest2,40);
    // benchmarkRotateFuseN<uint8_t,6,false>(ansatz,rp,dest2,40);
    // benchmarkRotateFuseN<uint16_t,12,false>(ansatz,rp,dest2,40);
    // benchmarkRotate3(ansatz,rp,dest1,40);
#endif


    FusedEvolve FE(ansatz->getStart(),Ham,compressMatrix,m_deCompressMatrix);
    std::vector<stateRotate::exc> excs;
    std::vector<realNumType> angles;
    for (size_t idx = 0; idx < rp.size(); idx++)//TODO check that this exists in the path
    {
        excs.emplace_back();
        dynamic_cast<stateRotate*>(ansatz->getLie())->convertIdxToExc(rp[idx].first,excs.back());
        angles.push_back(rp[idx].second);
    }
    FE.updateExc(excs);
    FE.updateExc(excs);

    FE.evolve(dest3,angles);
    auto startTime = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < 40; i++)
    {
        FE.evolve(dest3,angles);
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("FusedEvolve evolve Time taken total:",duration);
    startTime = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < 40; i++)
    {
        ansatz->updateAngles(angles);
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();

    logger().log("Ansatz updateAngles Time taken total:",duration);
    compressor::deCompressVector<numType>(dest3,dest2,comp);
    // benchmarkRotateFuseN<uint8_t,1,false>(ansatz,rp,dest3,40);
    benchmarkRotate3(ansatz,rp,dest1,40);
    logger().log("overlap", dest1.dot(dest2));
    // logger().log("overlap", dest3.dot(dest1));
    // logger().log("mag3", dest3.dot(dest3));
    logger().log("mag2", dest2.dot(dest2));
    logger().log("mag1", dest1.dot(dest1));

    logger().log("Derivative");
    vector<realNumType> derivExpt;
    vector<realNumType> derivFound;
    ansatz->getDerivativeVec(Ham,derivExpt);
    startTime = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < 40; i++)
    {
        FE.evolveDerivative(dest3,derivFound,angles);
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("FusedEvolve derivative Time taken total:",duration);

    startTime = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < 40; i++)
    {
        ansatz->getDerivativeVec(Ham,derivExpt);
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Ansatz getDerivativeVec Time taken total:",duration);

    logger().log("expt.expt",derivExpt.dot(derivExpt));
    logger().log("found.found",derivFound.dot(derivFound));
    logger().log("found.expt",derivFound.dot(derivExpt));
    logger().log("Hessian");
    Eigen::MatrixXd HessianExpt;
    Eigen::MatrixXd HessianFound;
    ansatz->updateAngles(angles);
    startTime = std::chrono::high_resolution_clock::now();
    ansatz->getHessianAndDerivative(Ham,HessianExpt,derivExpt,&compressMatrix);
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Ansatz Hessian Time taken total:",duration);

    startTime = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1; i++)
    {
    FE.evolveHessian(HessianFound,derivFound,angles); //TODO this is now the compressed derivative.
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    logger().log("Fused Ansatz Hessian Time taken total:",duration);

    logger().log("expt.expt",derivExpt.dot(derivExpt));
    logger().log("found.found",derivFound.dot(derivFound));
    logger().log("found.expt",derivFound.dot(derivExpt));
    logger().log("HessianExpt norm", HessianExpt.norm());
    logger().log("HessianFound norm", HessianFound.norm());
    logger().log("diff norm", (HessianFound-HessianExpt).norm());
    delete[] resultPtrs;
    return;
    logger().log("Impossible");
    logger().log("dest1 normSq",dest1.dot(dest1));



    logger().log("dest2 dot dest1 ",dest2.dot(dest1));
    logger().log("dest2 normSq",dest2.dot(dest2));

    logger().log("dest3 dot dest1 ",dest3.dot(dest1));
    logger().log("dest3 dot dest2 ",dest3.dot(dest2));
    logger().log("dest3 normSq",dest3.dot(dest3));

    logger().log("dest4 dot dest1 ",dest4.dot(dest1));
    logger().log("dest4 dot dest2 ",dest4.dot(dest2));
    logger().log("dest4 dot dest3 ",dest4.dot(dest3));
    logger().log("dest4 normSq",dest4.dot(dest4));

    logger().log("dest5 dot dest1 ",dest5.dot(dest1));
    logger().log("dest5 dot dest2 ",dest5.dot(dest2));
    logger().log("dest5 dot dest3 ",dest5.dot(dest3));
    logger().log("dest5 dot dest4 ",dest5.dot(dest4));
    logger().log("dest5 normSq",dest5.dot(dest5));

    logger().log("dest6 dot dest1 ",dest6.dot(dest1));
    logger().log("dest6 dot dest2 ",dest6.dot(dest2));
    logger().log("dest6 dot dest3 ",dest6.dot(dest3));
    logger().log("dest6 dot dest4 ",dest6.dot(dest4));
    logger().log("dest6 dot dest5 ",dest6.dot(dest5));
    logger().log("dest6 normSq",dest6.dot(dest6));


    // benchmarkDeriv(ansatz,rp,Ham);


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

