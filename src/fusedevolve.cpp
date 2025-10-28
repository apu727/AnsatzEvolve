/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "fusedevolve.h"
#include "logger.h"
#include "myComplex.h"
#include "threadpool.h"
#include <chrono>
#include <future>
#include <numeric>
//The setup functions
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

//State and sign. True is +ve
std::pair<uint32_t,bool> applyExcToBasisState_(uint32_t state, const stateRotate::exc& a)
{
    constexpr bool isComplex = !std::is_same_v<realNumType,numType>;
    bool complexSet = false;
    uint32_t activeBits =  0;
    uint32_t createBits = 0;
    uint32_t annihilateBits = 0;

    numType phase = 1;
    if (a[0] < 0 && a[1] < 0)
        return std::make_pair(state,true);

    if (a[2] > -1 && a[3] > -1)
    {
        if (a[0] == a[1] || a[2] == a[3])
        {
            fprintf(stderr,"Wrong order in creation annihilation operators");
            return std::make_pair(state,true);
        }
        createBits = (1<<a[0]) | (1<<a[1]);
        annihilateBits = (1<<a[2]) | (1<<a[3]);
        uint32_t signMask = ((1<<a[0])-1) ^ ((1<<a[1])-1) ^((1<<a[2])-1) ^((1<<a[3])-1);
        signMask = signMask & ~((1<<a[0]) | (1<<a[1]) | (1<<a[2]) | (1<<a[3]));
        activeBits = createBits | annihilateBits;
        phase *= (popcount(state & signMask) & 1) ? -1 : 1;
        if (a[0] < a[1])
            phase *= -1;
        if (a[2] < a[3])
            phase *= -1;
    }
    else
    {
        createBits = (1<<a[0]);
        annihilateBits = (1<<a[1]);
        activeBits = createBits | annihilateBits;

        uint32_t signMask = ((1<<a[0])-1) ^ ((1<<a[1])-1);
        signMask = signMask & ~((1<<a[0]) | (1<<a[1]));
        phase *= (popcount(state & signMask) & 1) ? -1 : 1;
    }
    if (isComplex)
    {
        complexSet = state & 1;
        state = state >>1;
    }

    uint32_t basisState = state;
    uint32_t resultState = basisState;



    uint32_t maskedBasisState = basisState & activeBits;

    if (createBits == annihilateBits) // number operator
    {
#ifdef useComplex
        if (((maskedBasisState & annihilateBits) ^ annihilateBits) == 0)
        {
            phase *= iu;
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
            phase *= 1;
            resultState = (basisState ^ annihilateBits) ^ createBits;
        }
        else if (((maskedBasisState & createBits) ^ createBits) == 0 && (((maskedBasisState ^ createBits) & annihilateBits)) == 0)
        {
            phase *= -1;
            resultState = (basisState ^ createBits) ^ annihilateBits;
        }
        else
        {
            phase = 0;
        }
    }

    if (phase == numType(1))
    {
        if (isComplex)
            return std::make_pair((resultState<<1) + (complexSet? 1 : 0),true);
        else
            return std::make_pair(resultState,true);
    }
    else if (phase == numType(-1))
    {
        if (isComplex)
            return std::make_pair((resultState<<1) + (complexSet? 1 : 0),false);
        else
            return std::make_pair(resultState,false);
    }
    else if (phase == iu)
    {
        assert(isComplex);
        return std::make_pair((resultState<<1)+(complexSet? 0 : 1),(complexSet ? false : true));
    }
    else
    {
        if (isComplex)
            return std::make_pair((resultState<<1) + (complexSet? 1 : 0),false);
        else
            return std::make_pair(resultState,false);
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
                    uint32_t* currentMap, bool* currentSigns,
                    indexType rotIdx, indexType numberOfActiveRotationsSoFar,
                    const std::array<std::pair<uint32_t,bool>,numberOfRotsThatExist>& initialLinks, const std::array<stateRotate::exc,numberOfRotsThatExist>& rots)
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
        auto effect = applyExcToBasisState_(currentMap[idx],rots[rotIdx]);
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
            auto effect = applyExcToBasisState_(currentMap[idx],rots[rotIdx]);
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
temp3 = scratchSpacehPsi[firstIndex + 0 + i*rotCount];\
temp4 = scratchSpacehPsi[firstIndex + 1 + i*rotCount];\
scratchSpacehPsi[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*0] ? 1 : -1)*temp4*S[0] + temp3 * C[0];\
scratchSpacehPsi[firstIndex + 1 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*0] ? 1 : -1)*temp3*S[0] + temp4 * C[0];\
if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
        *(result[0]) += std::real(myConj(temp3) * temp2);\
        *(result[0]) -= std::real(myConj(temp4) * temp1);\
    }\
    else\
    {\
        *(result[0]) -= std::real(myConj(temp3) * temp2);\
        *(result[0]) += std::real(myConj(temp4) * temp1);\
    }\
}

#define makeLayer1(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 2 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp2*S[1] + temp1 * C[1];\
scratchSpace[firstIndex + 2 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp1*S[1] + temp2 * C[1];\
if constexpr(toBraket)\
{\
    temp3 = scratchSpacehPsi[firstIndex + 0 + i*rotCount];\
    temp4 = scratchSpacehPsi[firstIndex + 2 + i*rotCount];\
    scratchSpacehPsi[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp4*S[1] + temp3 * C[1];\
    scratchSpacehPsi[firstIndex + 2 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*1] ? 1 : -1)*temp3*S[1] + temp4 * C[1];\
    if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
        *(result[1]) += std::real(myConj(temp3) * temp2);\
        *(result[1]) -= std::real(myConj(temp4) * temp1);\
    }\
    else\
    {\
        *(result[1]) -= std::real(myConj(temp3) * temp2);\
        *(result[1]) += std::real(myConj(temp4) * temp1);\
    }\
}

#define makeLayer2(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 4 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp2*S[2] + temp1 * C[2];\
scratchSpace[firstIndex + 4 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp1*S[2] + temp2 * C[2];\
if constexpr(toBraket)\
{\
    temp3 = scratchSpacehPsi[firstIndex + 0 + i*rotCount];\
    temp4 = scratchSpacehPsi[firstIndex + 4 + i*rotCount];\
    scratchSpacehPsi[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp4*S[2] + temp3 * C[2];\
    scratchSpacehPsi[firstIndex + 4 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*2] ? 1 : -1)*temp3*S[2] + temp4 * C[2];\
    if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
        *(result[2]) += std::real(myConj(temp3) * temp2);\
        *(result[2]) -= std::real(myConj(temp4) * temp1);\
    }\
    else\
    {\
        *(result[2]) -= std::real(myConj(temp3) * temp2);\
        *(result[2]) += std::real(myConj(temp4) * temp1);\
    }\
}

#define makeLayer3(firstIndex,GatesPerLayer, numInLayer,toBraket)\
temp1 = scratchSpace[firstIndex + 0 + i*rotCount];\
temp2 = scratchSpace[firstIndex + 8 + i*rotCount];\
scratchSpace[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp2*S[3] + temp1 * C[3];\
scratchSpace[firstIndex + 8 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp1*S[3] + temp2 * C[3];\
if constexpr(toBraket)\
{\
    temp3 = scratchSpacehPsi[firstIndex + 0 + i*rotCount];\
    temp4 = scratchSpacehPsi[firstIndex + 8 + i*rotCount];\
    scratchSpacehPsi[firstIndex + 0 + i*rotCount] =  (signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp4*S[3] + temp3 * C[3];\
    scratchSpacehPsi[firstIndex + 8 + i*rotCount] = -(signs[numInLayer + i*signsStride + GatesPerLayer*3] ? 1 : -1)*temp3*S[3] + temp4 * C[3];\
    if (signs[numInLayer+i*signsStride + GatesPerLayer*0])\
    {\
        *(result[3]) += std::real(myConj(temp3) * temp2);\
        *(result[3]) -= std::real(myConj(temp4) * temp1);\
    }\
    else\
    {\
        *(result[3]) -= std::real(myConj(temp3) * temp2);\
        *(result[3]) += std::real(myConj(temp4) * temp1);\
    }\
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_1(realNumType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=1*/,size_t numberToRepeat, indexType,
                               realNumType* scratchSpacehPsi/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 2;
    constexpr indexType signsStride = (rotCount/2)*1;

    realNumType temp1;
    realNumType temp2;
    realNumType temp3;
    realNumType temp4;
#pragma GCC unroll 2
    for (size_t i = 0; i < numberToRepeat; i++)
    {
        // #define signs true || signs
        makeLayer0(0,1,0,toBraket)
        // #undef signs
    }
}

template<typename indexType, bool toBraket>
inline void BENCHMARK_rotate_2(realNumType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType,
                               realNumType* scratchSpacehPsi/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 4;
    constexpr indexType signsStride = (rotCount/2)*2;

    realNumType temp1;
    realNumType temp2;
    realNumType temp3;
    realNumType temp4;
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
inline void BENCHMARK_rotate_3(realNumType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType,
                               realNumType* scratchSpacehPsi/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 8;
    constexpr indexType signsStride = (rotCount/2)*3;

    realNumType temp1;
    realNumType temp2;
    realNumType temp3;
    realNumType temp4;
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
inline void BENCHMARK_rotate_4(realNumType* scratchSpace, const bool* signs, const realNumType* S, const realNumType* C/*length=4*/,size_t numberToRepeat, indexType,
                               realNumType* scratchSpacehPsi/*[numberToFuse][localVectorSize]*/, realNumType*const* result)
{
    constexpr indexType rotCount = 16;
    constexpr indexType signsStride = (rotCount/2)*4;
    realNumType temp1;
    realNumType temp2;
    realNumType temp3;
    realNumType temp4;
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

template<typename indexType, indexType numberToFuse,indexType localVectorSize, bool toBraket>
inline void BENCHMARK_rotate(realNumType* scratchSpace,const bool* signs, const realNumType* S, const realNumType* C/*length=numberToFuse*/,size_t numberToRepeat,
                             realNumType*  scratchSpacehPsi, realNumType* const (&result)[numberToFuse]/*Result is stored at *(result[rotIdx]) where rotIdx \in [0,numberToFuse)*/)
{
    //Call the optimised routines if they exist
    if constexpr(numberToFuse == 1)
        return BENCHMARK_rotate_1<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,scratchSpacehPsi,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 2)
        return BENCHMARK_rotate_2<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,scratchSpacehPsi,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 3)
        return BENCHMARK_rotate_3<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,scratchSpacehPsi,static_cast<realNumType*const*>(result));
    else if constexpr(numberToFuse == 4)
        return BENCHMARK_rotate_4<indexType,toBraket>(scratchSpace,signs,S,C,numberToRepeat,localVectorSize,scratchSpacehPsi,static_cast<realNumType*const*>(result));
    else
    {
        constexpr indexType pseudoVectorSize = 1<<numberToFuse;
        constexpr uint32_t signsStride = (pseudoVectorSize/2)*numberToFuse;
        static_assert((pseudoVectorSize/2)*numberToFuse < 1<<16);
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
                        realNumType temp1 = scratchSpace[idx + rotCount*i];
                        realNumType temp2 = scratchSpace[idx+stride + rotCount*i];
                        const bool sign = signs[numberOfRotationsPerformed++ + (pseudoVectorSize/2)*rotIdx + signsStride*i];
                        // #define sign true
                        scratchSpace[idx + rotCount*i] =         (sign ? 1 : -1) * temp2 * S[rotIdx] + temp1 * C[rotIdx];
                        scratchSpace[idx+stride + rotCount*i] = -(sign ? 1 : -1) * temp1 * S[rotIdx] + temp2 * C[rotIdx];
                        if constexpr(toBraket)
                        {
                            realNumType temp3 = scratchSpacehPsi[idx + rotCount*i];
                            realNumType temp4 = scratchSpacehPsi[idx+stride + rotCount*i];
                            scratchSpacehPsi[idx + rotCount*i]        =  (sign ? 1 : -1) * temp4 * S[rotIdx] + temp3 * C[rotIdx];
                            scratchSpacehPsi[idx+stride + rotCount*i] = -(sign ? 1 : -1) * temp3 * S[rotIdx] + temp4 * C[rotIdx];
                            if (sign)
                            {
                                *(result[rotIdx]) += std::real(myConj(temp3) * temp2);
                                *(result[rotIdx]) -= std::real(myConj(temp4) * temp1);
                            }
                            else
                            {
                                *(result[rotIdx]) -= std::real(myConj(temp3) * temp2);
                                *(result[rotIdx]) += std::real(myConj(temp4) * temp1);
                            }
                        }
                        // #undef sign
                    }
                }
            }
        }
    }
}
template<typename indexType, indexType numberToFuse>
inline void storeTangents(realNumType* scratchSpace, const uint32_t*  const currentMap, const bool* signs, realNumType*__restrict__ const (&tangentLocation)[numberToFuse], size_t numberToRepeat)
{

    constexpr indexType pseudoVectorSize = 1<<numberToFuse;
    constexpr uint32_t signsStride = (pseudoVectorSize/2)*numberToFuse;
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
                    const bool sign = signs[numberOfRotationsPerformed++ + (pseudoVectorSize/2)*rotIdx + signsStride*i];
                    tangentLocation[rotIdx][currentMap[idx + rotCount*i]] =  (sign ? 1 : -1) * scratchSpace[idx+stride + rotCount*i];
                    tangentLocation[rotIdx][currentMap[idx+stride + rotCount*i]] = -(sign ? 1 : -1) * scratchSpace[idx + rotCount*i];
                }
            }
        }
    }
}
template<typename indexType, indexType numberToFuse>
auto setupFuseNDiagonal(const std::vector<stateRotate::exc>& excPath, const vector<numType>& startVec)
{
    //Diagonal elements always commute. Theyre diagonal...
    //Further they act as scalars on each given basis state. Therefore trivially they commute. Further the angles can be added
    static_assert(numberToFuse < sizeof(indexType)*8);
    constexpr bool isComplex = !std::is_same_v<realNumType,numType>;
    if (!isComplex)
        __builtin_trap();//only complex exc can be diagonal and antihermitian

    constexpr indexType numberOfPairsOfRotations = 1<<numberToFuse;
    typedef std::vector<uint32_t> localVector; //each element represents a 2D subspace and a sign
    typedef std::vector<std::array<localVector,numberOfPairsOfRotations>> fusedDiagonalAnsatz; //\Sum_{k=1}^{n} n choose k = 2^n for n >=0.
    //Each element of the array is a specific set of rotations active. Each element of the vector is a repetition

    fusedDiagonalAnsatz myFusedAnsatz;
    std::vector<uint32_t> activebasisStates;
    activebasisStates.resize(startVec.size()*(isComplex?2:1));

    std::shared_ptr<compressor> comp;
    bool isCompressed = startVec.getIsCompressed(comp);
    if (isCompressed)
    {
        for (uint32_t i = 0; i < startVec.size(); i++)
        {

            if (isComplex)
            {
                comp->deCompressIndex(i,activebasisStates[2*i]);
                activebasisStates[2*i] = activebasisStates[2*i]<<1;
                activebasisStates[2*i+1] = activebasisStates[2*i] +1;
            }
            else
                comp->deCompressIndex(i,activebasisStates[i]);
        }
    }
    else
    {
        std::iota(activebasisStates.begin(),activebasisStates.end(),0);
    }

    for (size_t i = 0; i < excPath.size(); i+=numberToFuse)
    {
        std::array<stateRotate::exc,numberToFuse> rots;
        for (indexType idx = 0; idx < numberToFuse; idx++)//TODO check that this exists in the path
        {
            rots[idx] = excPath[i+idx];
            assert(rots[idx].isDiagonal());
        }

        //If rot0 and rot2 are active then the index is 0b101. Note that 0b000 is always empty
        std::array<localVector,numberOfPairsOfRotations> currentLocalVectors;

        for(uint32_t currentCompIndex = 0; currentCompIndex < activebasisStates.size(); currentCompIndex++)
        {
            uint32_t currentBasisState = activebasisStates[currentCompIndex];
            if (currentBasisState & 1)
                continue;//These are already the complex element and will lead to a negative phase

            std::array<std::pair<uint32_t,bool>,numberToFuse> initialLinks;
            indexType activeRotIdx = 0;
            indexType numberOfActiveRots = 0;
            for (uint8_t idx = 0; idx < numberToFuse; idx++)
            {
                initialLinks[idx] = applyExcToBasisState_(currentBasisState,rots[idx]);
                if (initialLinks[idx].first != currentBasisState)
                {
                    assert(initialLinks[idx].second);
                    activeRotIdx |= 1<<idx;
                    ++numberOfActiveRots;
                    assert(currentBasisState == (initialLinks[idx].first & -2));
                }
            }
            //All not active
            if (numberOfActiveRots == 0)
                continue;


            localVector& currentMap = currentLocalVectors[activeRotIdx];
            currentMap.push_back(currentBasisState);

        }
        for (indexType activeRotIdx = 0; activeRotIdx <  numberOfPairsOfRotations; activeRotIdx++)
        {
            if (currentLocalVectors[activeRotIdx].size() != 0) // for the first iteration
            {
                if (isCompressed)
                {
                    for (uint32_t idx = 0; idx < currentLocalVectors[activeRotIdx].size(); idx++)
                    {
                        bool __attribute__ ((unused))complexOffset = currentLocalVectors[activeRotIdx][idx] & 1;
                        assert(complexOffset == false);
                        comp->compressIndex(currentLocalVectors[activeRotIdx][idx]>>1,currentLocalVectors[activeRotIdx][idx]);
                        currentLocalVectors[activeRotIdx][idx] = currentLocalVectors[activeRotIdx][idx] << 1;
                    }
                }
            }
        }
        myFusedAnsatz.emplace_back(std::move(currentLocalVectors));
    }

    return myFusedAnsatz;
}

#define DiagonalEvolve(offset)

template<typename indexType, indexType numberToFuse, bool BraketWithTangentOfResult, bool storeTangent, bool parallelise = false,
         //Various expressions that are needed to get the types right
         indexType numberOfPairsOfRotations = 1<<numberToFuse,
         typename localVector = std::vector<uint32_t>,
         typename fusedDiagonalAnsatz = std::array<localVector,numberOfPairsOfRotations>>

#define RunFuseNKernel(indexInUnroll)\
scratchSpace[2*indexInUnroll + 0] = startVec[currLocalVector[indexInUnroll + doneCount]];\
scratchSpace[2*indexInUnroll + 1] = startVec[currLocalVector[indexInUnroll + doneCount]+1];\
startVec[currLocalVector[indexInUnroll + doneCount]] = -S*(scratchSpace[2*indexInUnroll + 1]) + C*scratchSpace[2*indexInUnroll + 0];\
startVec[currLocalVector[indexInUnroll + doneCount]+1] = S*(scratchSpace[2*indexInUnroll + 0]) + C*scratchSpace[2*indexInUnroll + 1];\
if constexpr(BraketWithTangentOfResult)\
{\
    scratchSpacehPsi[2*indexInUnroll + 0] = hPsi[currLocalVector[indexInUnroll + doneCount]];\
    scratchSpacehPsi[2*indexInUnroll + 1] = hPsi[currLocalVector[indexInUnroll + doneCount]+1];\
    hPsi[currLocalVector[indexInUnroll + doneCount]] = -S*(scratchSpacehPsi[2*indexInUnroll + 1]) + C*scratchSpacehPsi[2*indexInUnroll + 0];\
    hPsi[currLocalVector[indexInUnroll + doneCount]+1] = S*(scratchSpacehPsi[2*indexInUnroll + 0]) + C*scratchSpacehPsi[2*indexInUnroll + 1];\
    accumulatedDot -= scratchSpacehPsi[2*indexInUnroll + 0] * scratchSpace[2*indexInUnroll + 1];\
    accumulatedDot += scratchSpacehPsi[2*indexInUnroll + 1] * scratchSpace[2*indexInUnroll + 0];\
}\
if constexpr(storeTangent)\
{\
    for (uint8_t r = 0; r < numberOfActiveRots; r++)\
    {\
        tangentStore[activeRots[r]+i][currLocalVector[indexInUnroll + doneCount]] = -scratchSpace[2*indexInUnroll + 1];\
        tangentStore[activeRots[r]+i][currLocalVector[indexInUnroll + doneCount]+1] = scratchSpace[2*indexInUnroll + 0];\
    }\
}

void RunFuseNDiagonal(fusedDiagonalAnsatz* const myFusedAnsatz, realNumType* startVec, const realNumType* angles, size_t nAngles,
              realNumType* hPsi = nullptr, realNumType** result = nullptr/*result is array of pointers to storage places. The array has length nAngles. IT IS NOT ZEROED BY THIS FUNCTION!*/,
              realNumType** tangentStore = nullptr/* tangents are stored before the evolution of each fused gate in myFusedAnsatz*/)
{
    for (size_t i = 0; i < nAngles; i+= numberToFuse)
    {
        fusedDiagonalAnsatz& currAnsatz = myFusedAnsatz[i/numberToFuse];
        for (indexType activeRotIdx = 1; activeRotIdx < numberOfPairsOfRotations; activeRotIdx++)
        {
            const localVector& currLocalVector = currAnsatz[activeRotIdx];
            if (currLocalVector.size() == 0)
                continue;
            realNumType totalAngle = 0;
            uint8_t activeRots[numberToFuse];
            uint8_t numberOfActiveRots = 0;
            for (indexType rotIdx = 0; rotIdx < numberToFuse; rotIdx++)
            {
                if (activeRotIdx & (1<<rotIdx))
                {
                    totalAngle += angles[i+rotIdx];
                    if constexpr(BraketWithTangentOfResult || storeTangent)
                    {
                        activeRots[numberOfActiveRots++] = rotIdx;
                    }
                }
            }
            realNumType S;
            realNumType C;
            mysincos(totalAngle,&S,&C); // Its unclear whether this is faster or by expanding (cos + sin)(cos + sin). Unlikely to be bottleneck?



            //Manual unroll because im almost certain the compiler wont know that currLocalVector are all different
            constexpr uint8_t unrollCount = 8;
            realNumType scratchSpace[2*unrollCount];
            realNumType scratchSpacehPsi[2*unrollCount];
            realNumType accumulatedDot = 0;
            size_t doneCount= 0;
            for (; doneCount + unrollCount < currLocalVector.size(); doneCount+= unrollCount)
            {
                RunFuseNKernel(0)
                RunFuseNKernel(1)
                RunFuseNKernel(2)
                RunFuseNKernel(3)
                RunFuseNKernel(4)
                RunFuseNKernel(5)
                RunFuseNKernel(6)
                RunFuseNKernel(7)
            }
            //finish off
            for (; doneCount < currLocalVector.size(); doneCount++)
            {
                RunFuseNKernel(0)
            }
            if constexpr(BraketWithTangentOfResult)
            {
                for (uint8_t r = 0; r < numberOfActiveRots; r++)
                {
                    (*result[activeRots[r]+i]) += accumulatedDot;
                }
            }
        }
    }
}
template<typename indexType, indexType numberToFuse>
auto setupFuseN(const std::vector<stateRotate::exc>& excPath, const vector<numType>& startVec)
{//Note it is assumed that all Excs commute in ExcPath
    static_assert(numberToFuse < sizeof(indexType)*8);
    constexpr bool isComplex = !std::is_same_v<realNumType,numType>;
    constexpr indexType localVectorSize = 1<<numberToFuse;
    typedef std::array<bool,(localVectorSize/2)*numberToFuse> signMap; //true means +ve
    typedef std::array<uint32_t,localVectorSize> localVectorMap;
    typedef std::vector<std::pair<localVectorMap,signMap>> localVector;

    typedef std::vector<std::array<localVector,localVectorSize>> fusedAnsatz; //\Sum_{k=1}^{n} n choose k = 2^n for n >=0
    fusedAnsatz myFusedAnsatz;
    //preprocess
    assert(excPath.size() % numberToFuse == 0);

    std::vector<uint32_t> activebasisStates;
    activebasisStates.resize(startVec.size()*(isComplex?2:1));

    std::shared_ptr<compressor> comp;
    bool isCompressed = startVec.getIsCompressed(comp);
    if (isCompressed)
    {
        for (uint32_t i = 0; i < startVec.size(); i++)
        {

            if (isComplex)
            {
                comp->deCompressIndex(i,activebasisStates[2*i]);
                activebasisStates[2*i] = activebasisStates[2*i]<<1;
                activebasisStates[2*i+1] = activebasisStates[2*i] +1;
            }
            else
                comp->deCompressIndex(i,activebasisStates[i]);
        }
    }
    else
    {
        std::iota(activebasisStates.begin(),activebasisStates.end(),0);
    }

    size_t totalRotcount = 0;
    for (size_t i = 0; i < excPath.size(); i+=numberToFuse)
    {
        std::array<stateRotate::exc,numberToFuse> rots;
        for (indexType idx = 0; idx < numberToFuse; idx++)//TODO check that this exists in the path
            rots[idx] = excPath[i+idx];

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

        for(uint32_t currentCompIndex = 0; currentCompIndex < activebasisStates.size(); currentCompIndex++)
        {
            uint32_t currentBasisState = activebasisStates[currentCompIndex];

            std::array<std::pair<uint32_t,bool>,numberToFuse> initialLinks;
            indexType activeRotIdx = 0;
            indexType numberOfActiveRots = 0;
            bool allPositive = true;
            for (uint8_t idx = 0; idx < numberToFuse; idx++)
            {
                initialLinks[idx] = applyExcToBasisState_(currentBasisState,rots[idx]);
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
                    if (isCompressed)
                    {
                        for (uint32_t idx = 0; idx < localVectorSize; idx++)
                        {
                            if (isComplex)
                            {
                                bool complexOffset = filledMap[idx] & 1;
                                // assert(complexOffset == false);
                                comp->compressIndex(filledMap[idx]>>1,filledMap[idx]);
                                filledMap[idx] = (filledMap[idx] << 1) + (complexOffset ? 1 : 0);
                            }
                            else
                                comp->compressIndex(filledMap[idx],filledMap[idx]);
                        }
                    }
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
            // for (indexType t = 1; t < (1<<numberOfActiveRots); t++)
            //     *(currentMap.begin() + currentMapFilledSize + t) = (uint32_t)-1;

            uint32_t currentSignsStep = numberOfActiveRots*currentMapFilledSize/2;
            fillCurrentMap<indexType,numberToFuse>(activeRotIdx,numberOfActiveRots,currentMap.begin()+currentMapFilledSize,currentSigns.begin()+currentSignsStep,rotIdx,0,initialLinks,rots);
            // for (indexType t = 0; t < (1<<numberOfActiveRots); t++)
            //     if (*(currentMap.begin() + currentMapFilledSize + t) == (uint32_t)-1)
            //         __builtin_trap();
            for (size_t l = 0; l < 1ul<<numberOfActiveRots; l++)
                for (size_t m = l+1; m < 1ul<<numberOfActiveRots; m++)
                    assert(currentMap[currentMapFilledSize + l] != currentMap[currentMapFilledSize + m]);
            currentMapFilledSize += 1<<numberOfActiveRots;
        }
        for (indexType idx = 0; idx <  localVectorSize; idx++)
        {
            if (currentLocalVectors[idx].size() != 0) // for the first iteration
            {
                indexType filledSize = currentRotFilledSize[idx];
                totalRotcount += dot(idx,(uint32_t)-1,sizeof(idx)*8)*localVectorSize/2;
                localVectorMap& potentiallyFilledMap = currentLocalVectors[idx].back().first;
                if (filledSize < localVectorSize)
                    potentiallyFilledMap[filledSize] = (uint32_t)-1;

                //The last one will not have been compressed
                if (isCompressed)
                {
                    for (uint32_t idx = 0; idx < filledSize; idx++)
                    {
                        if (isComplex)
                        {
                            bool complexOffset = potentiallyFilledMap[idx] & 1;
                            // assert(complexOffset == false);
                            comp->compressIndex(potentiallyFilledMap[idx]>>1,potentiallyFilledMap[idx]);
                            potentiallyFilledMap[idx] = (potentiallyFilledMap[idx] << 1) + (complexOffset ? 1 : 0);
                        }
                        else
                            comp->compressIndex(potentiallyFilledMap[idx],potentiallyFilledMap[idx]);
                    }
                }
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
    // logger().log("FuseN rotcount",totalRotcount);
    return myFusedAnsatz;
}

template<typename indexType, indexType numberToFuse, bool BraketWithTangentOfResult, bool storeTangent, bool parallelise = false,
//Various expressions that are needed to get the types right
         indexType localVectorSize = 1<<numberToFuse,
         typename signMap = std::array<bool,(localVectorSize/2)*numberToFuse>, //true means +ve
         typename localVectorMap = std::array<uint32_t,localVectorSize>,
         typename localVector = std::vector<std::pair<localVectorMap,signMap>>,
         typename fusedAnsatz = /*std::vector<*/std::array<localVector,localVectorSize>>/*>*/

void RunFuseN(fusedAnsatz* const myFusedAnsatz, realNumType* startVec, const realNumType* angles, size_t nAngles,
                          realNumType* hPsi = nullptr, realNumType** result = nullptr/*result is array of pointers to storage places. The array has length nAngles. IT IS NOT ZEROED BY THIS FUNCTION!*/,
              realNumType** tangentStore = nullptr/* tangents are stored before the evolution of each fused gate in myFusedAnsatz*/)
{
    realNumType scratchSpace[localVectorSize];
    realNumType scratchSpacehPsi[localVectorSize];
    auto startTime = std::chrono::high_resolution_clock::now();
#ifdef MakeParallelEvolveCode
    threadpool& pool  = threadpool::getInstance(NUM_CORES);
#endif
    if constexpr (BraketWithTangentOfResult)
    {
        if (hPsi == nullptr || result == nullptr)
            __builtin_trap();
    }
    else
    {
        //These are dereferenced but their results are not used
        result = new realNumType*[nAngles];
        // toBraKet = new Matrix<numType>(1,rotationPath.size());
    }
    if constexpr(storeTangent)
    {
        assert(tangentStore != nullptr);
    }

    // if constexpr (BraketWithTangentOfResult)
    // {
    //     for (indexType i = 0; i < nAngles; i++)
    //         *(result[i]) = 0;

    // }
    assert(nAngles % numberToFuse == 0);
    for (size_t i = 0; i < nAngles; i+=numberToFuse)
    {
        std::array<realNumType,numberToFuse> sines;
        std::array<realNumType,numberToFuse> cosines;
        for (indexType idx = 0; idx < numberToFuse; idx++)
        {
            mysincos(angles[i+idx],&sines[idx],&cosines[idx]);
        }


        if constexpr(numberToFuse == 2)
        {
#ifdef MakeParallelEvolveCode
            std::future<void> futs[3];
            futs[0] = threadpool::getInstance(NUM_CORES).queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore]()
#endif
            {
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][1];
                for (size_t j = 0; j < currentLocalVectors.size(); j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;
                    indexType filledSize = localVectorSize;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];

                    }

                    if (!((filledSize == 4) || (filledSize == 2))) __builtin_unreachable();


                    if constexpr(storeTangent)
                        storeTangents<indexType,1>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i]},filledSize/2);
                    if (filledSize == 4)
                    {
                        BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],2,scratchSpacehPsi,{result[i]});
                    }
                    else
                        BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],1,scratchSpacehPsi,{result[i]});

                    //Restore scratch space
                    for (indexType idx = 0; idx < filledSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }

            }
#ifdef MakeParallelEvolveCode
            ,!parallelise);
#endif
#ifdef MakeParallelEvolveCode
            futs[1] = threadpool::getInstance(NUM_CORES).queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore]()
#endif
            {
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][2];
                for (size_t j = 0; j < currentLocalVectors.size(); j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;
                    indexType filledSize = localVectorSize;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,1>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i+1]},filledSize/2);
                    assert(filledSize != 0);
                    if (!((filledSize == 4) || (filledSize == 2))) __builtin_unreachable();

                    if (filledSize == 4)
                    {
                        BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[1],&cosines[1],2,scratchSpacehPsi,{result[1+i]});
                    }
                    else
                        BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[1],&cosines[1],1,scratchSpacehPsi,{result[1+i]});
                    //Restore scratch space
                    for (indexType idx = 0; idx < filledSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }

            }
#ifdef MakeParallelEvolveCode
            ,!parallelise);
            if constexpr(BraketWithTangentOfResult)
            {
                futs[0].wait();
                futs[1].wait();
                //This is because result is modified
            }
#endif
#ifdef MakeParallelEvolveCode
            futs[2] = threadpool::getInstance(NUM_CORES).queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore]()
#endif
            {
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][3];
                for (size_t j = 0; j < currentLocalVectors.size(); j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,2>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i],tangentStore[i+1]},1);
                    BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[0],&cosines[0],1,scratchSpacehPsi,{result[i],result[i+1]});

                    //Restore scratch space
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }
            }
#ifdef MakeParallelEvolveCode
,!parallelise || BraketWithTangentOfResult); // no point threading since only this is running anyway
            if (!BraketWithTangentOfResult)
            {
                futs[0].wait();
                futs[1].wait();
                futs[2].wait();
            }
#endif

        }
        else if constexpr((numberToFuse == 3 || numberToFuse == 4 ) && !BraketWithTangentOfResult)
        {




#ifdef MakeParallelEvolveCode
            std::future<void> futs[4+6+4+1];
#endif

            for (indexType abc = 0; abc < 4; abc++)
            {

#ifdef MakeParallelEvolveCode
            futs[abc] = pool.queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore,abc]()
#endif
            {
                constexpr indexType theOnes[4] =  {0b0001,0b0010,0b0100,0b1000};
                constexpr indexType theOnesOffset[4] = {0,1,2,3};
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                if constexpr (numberToFuse == 3) if (theOnes[abc] & 0b1000)
#ifdef MakeParallelEvolveCode
                        return;
#else
                        continue;
#endif
                const indexType rotIdx = theOnes[abc];
                const indexType activeRot = theOnesOffset[abc];

                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                if (currentLocalVectors.size() == 0)
#ifdef MakeParallelEvolveCode
                    return;
#else
                    continue;
#endif
                for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,1>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i+activeRot]},localVectorSize/2);
                    BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[activeRot],&cosines[activeRot],localVectorSize/2,scratchSpacehPsi,{result[i+activeRot]});
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }

                {
                    const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                    const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;
                    indexType filledSize = localVectorSize;
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        if (currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,1>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i+activeRot]},filledSize/2);
                    BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],&sines[activeRot],&cosines[activeRot],filledSize/2,scratchSpacehPsi,{result[i+activeRot]});
                    for (indexType idx = 0; idx < filledSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }}
#ifdef MakeParallelEvolveCode
,!parallelise);
#endif
            }
#ifdef MakeParallelEvolveCode
            if constexpr(BraketWithTangentOfResult) for (indexType abc = 0; abc < 4; abc++) futs[abc].wait();
#endif

            for (indexType abc = 0; abc < 6; abc++)
            {
#ifdef MakeParallelEvolveCode
            futs[abc+4] = pool.queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore,abc]()
#endif
            {
                constexpr indexType theTwos[6] =          {0b0011,0b1100,0b0101,0b1010,0b0110,0b1001};
                constexpr indexType theTwosOffset[6][2] = {{0,1}, {2,3}, {0,2}, {1,3}, {1,2}, {0,3}};
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                if constexpr (numberToFuse == 3) if (theTwos[abc] & 0b1000)
#ifdef MakeParallelEvolveCode
                        return;
#else
                        continue;
#endif

                const indexType rotIdx = theTwos[abc];
                const indexType activeRot0 = theTwosOffset[abc][0];
                const indexType activeRot1 = theTwosOffset[abc][1];
                const realNumType S[2] = {sines[activeRot0],sines[activeRot1]};
                const realNumType C[2] = {cosines[activeRot0],cosines[activeRot1]};
                realNumType* resultArr[2] = {result[activeRot0+i], result[activeRot1+i]}; // This could be a const array

                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                if (currentLocalVectors.size() == 0)
#ifdef MakeParallelEvolveCode
                    return;
#else
                    continue;
#endif
                for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,2>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[activeRot0+i], tangentStore[activeRot1+i]},localVectorSize/4);
                    BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,localVectorSize/4,scratchSpacehPsi,resultArr);

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }

                {
                    const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                    const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;

                    indexType filledSize = localVectorSize;
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        if (currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,2>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[activeRot0+i], tangentStore[activeRot1+i]},filledSize/4);
                    BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,filledSize/4,scratchSpacehPsi,resultArr);
                    for (indexType idx = 0; idx < filledSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }}
#ifdef MakeParallelEvolveCode
                ,!parallelise);
            if constexpr(BraketWithTangentOfResult)
            {//Do in pairs, again because of Result
                if (abc %2 == 1)
                {
                    futs[abc].wait();
                    futs[abc-1].wait();
                }
            }
#endif
            }

            for (indexType abc = 0; abc < 4; abc++)
            {
#ifdef MakeParallelEvolveCode
            futs[abc+4+6] = pool.queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore,abc]()
#endif
            {
                constexpr indexType theThrees[4] =          {0b0111,0b1011,0b1101,0b1110};
                constexpr indexType theThreesOffset[6][3] = {{0,1,2},{0,1,3},{0,2,3},{1,2,3}};
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                if constexpr (numberToFuse == 3) if (theThrees[abc] & 0b1000)
#ifdef MakeParallelEvolveCode
                        return;
#else
                        continue;
#endif
                const indexType rotIdx = theThrees[abc];
                const indexType activeRot0 = theThreesOffset[abc][0];
                const indexType activeRot1 = theThreesOffset[abc][1];
                const indexType activeRot2 = theThreesOffset[abc][2];

                const realNumType S[3] = {sines[activeRot0],sines[activeRot1],sines[activeRot2]};
                const realNumType C[3] = {cosines[activeRot0],cosines[activeRot1],cosines[activeRot2]};
                realNumType* resultArr[3] = {result[activeRot0+i], result[activeRot1+i],result[activeRot2+i]}; // This could be a const array

                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                if (currentLocalVectors.size() == 0)
#ifdef MakeParallelEvolveCode
                    return;
#else
                    continue;
#endif
                for (size_t j = 0; j < currentLocalVectors.size()-1; j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,3>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[activeRot0+i], tangentStore[activeRot1+i],tangentStore[activeRot2+i]},localVectorSize/8);
                    BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,localVectorSize/8,scratchSpacehPsi,resultArr);
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }

                {
                    const localVectorMap& currentMap = currentLocalVectors[currentLocalVectors.size()-1].first;
                    const signMap& currentSigns = currentLocalVectors[currentLocalVectors.size()-1].second;
                    indexType filledSize = localVectorSize;
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        if (currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,3>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[activeRot0+i], tangentStore[activeRot1+i],tangentStore[activeRot2+i]},filledSize/8);
                    BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,filledSize/8,scratchSpacehPsi,resultArr);
                    for (indexType idx = 0; idx < filledSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }
                }
#ifdef MakeParallelEvolveCode
,!parallelise || BraketWithTangentOfResult); // dont queue if it is braket because of result overwriting.
#endif
            }

            if constexpr(numberToFuse == 4)
            {
#ifdef MakeParallelEvolveCode
                futs[4+6+4] = pool.queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore]()
#endif
            {
#ifdef MakeParallelEvolveCode
                realNumType scratchSpace[localVectorSize];
                realNumType scratchSpacehPsi[localVectorSize];
#endif
                constexpr indexType rotIdx = 0b1111;

                const realNumType* S = &sines[0];
                const realNumType* C = &cosines[0];

                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][rotIdx];
                for (size_t j = 0; j < currentLocalVectors.size(); j++)
                {
                    const localVectorMap& currentMap = currentLocalVectors[j].first;
                    const signMap& currentSigns = currentLocalVectors[j].second;

                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
                    }
                    if constexpr(storeTangent)
                        storeTangents<indexType,4>(scratchSpace,currentMap.begin(),currentSigns.begin(),{tangentStore[i], tangentStore[i+1],tangentStore[i+2],tangentStore[i+3]},1);
                    BENCHMARK_rotate<indexType,4,localVectorSize,BraketWithTangentOfResult>(scratchSpace,&currentSigns[0],S,C,1,scratchSpacehPsi,{result[i], result[i+1],result[i+2],result[i+3]});
                    for (indexType idx = 0; idx < localVectorSize; idx++)
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }
                }
#ifdef MakeParallelEvolveCode
,!parallelise);

                futs[4+6+4].wait();
#endif
            }
#ifdef MakeParallelEvolveCode
            if constexpr(!BraketWithTangentOfResult) for (indexType abc = 0; abc < 4+6+4; abc++) futs[abc].wait();
#endif


        }
        else
        {
            // std::future<void> futs[localVectorSize-1];
            //0b000 is always empty
            for (indexType activeRotIdx = 1; activeRotIdx < localVectorSize; activeRotIdx++)
            {
            // futs[activeRotIdx-1] = pool.queueWork([myFusedAnsatz,i,startVec,&sines,&cosines,result,hPsi,tangentStore,activeRotIdx]()
            // {

                const localVector& currentLocalVectors = myFusedAnsatz[i/numberToFuse][activeRotIdx];
                if (currentLocalVectors.size() == 0)
                    continue;
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
                        if (j == currentLocalVectors.size()-1 && currentMap[idx] == (uint32_t)-1)
                        {
                            filledSize = idx;
                            break;
                        }
                        scratchSpace[idx] = startVec[currentMap[idx]];
                        if constexpr(BraketWithTangentOfResult)
                            scratchSpacehPsi[idx] = hPsi[currentMap[idx]];
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
                            const realNumType S[1] = {sines[activeRots[0]]};
                            const realNumType C[1] = {cosines[activeRots[0]]};
                            realNumType* resultArr[1] = {result[activeRots[0]+i]};
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[1] = {tangentStore[activeRots[0]+i]};
                                storeTangents<indexType,1>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/2);
                            }
                            BENCHMARK_rotate<indexType,1,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/2,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 2:
                        if constexpr (numberToFuse >= 2)
                        {
                            const realNumType S[2] = {sines[activeRots[0]],sines[activeRots[1]]};
                            const realNumType C[2] = {cosines[activeRots[0]],cosines[activeRots[1]]};
                            realNumType* resultArr[2] = {result[activeRots[0]+i], result[activeRots[1]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[2] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i]};
                                storeTangents<indexType,2>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/4);
                            }
                            BENCHMARK_rotate<indexType,2,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/4,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 3:
                        if constexpr (numberToFuse >= 3)
                        {
                            const realNumType S[3] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]]};
                            const realNumType C[3] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]]};
                            realNumType* resultArr[3] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[3] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i],tangentStore[activeRots[2]+i]};
                                storeTangents<indexType,3>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/8);
                            }
                            BENCHMARK_rotate<indexType,3,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/8,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 4:
                        if constexpr (numberToFuse >= 4)
                        {
                            const realNumType S[4] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]]};
                            const realNumType C[4] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]]};
                            realNumType* resultArr[4] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[4] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i],tangentStore[activeRots[2]+i],tangentStore[activeRots[3]+i]};
                                storeTangents<indexType,4>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/16);
                            }
                            BENCHMARK_rotate<indexType,4,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/16,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 5:
                        if constexpr (numberToFuse >= 5)
                        {
                            const realNumType S[5] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]]};
                            const realNumType C[5] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]]};
                            realNumType* resultArr[5] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[5] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i],tangentStore[activeRots[2]+i],tangentStore[activeRots[3]+i],tangentStore[activeRots[4]+i]};
                                storeTangents<indexType,5>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/32);
                            }
                            BENCHMARK_rotate<indexType,5,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/32,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 6:
                        if constexpr (numberToFuse >= 6)
                        {
                            const realNumType S[6] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]],sines[activeRots[5]]};
                            const realNumType C[6] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]],cosines[activeRots[5]]};
                            realNumType* resultArr[6] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i],result[activeRots[5]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[6] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i],tangentStore[activeRots[2]+i],tangentStore[activeRots[3]+i],tangentStore[activeRots[4]+i],tangentStore[activeRots[5]+i]};
                                storeTangents<indexType,6>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/64);
                            }
                            BENCHMARK_rotate<indexType,6,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/64,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 7:
                        if constexpr (numberToFuse >= 7)
                        {
                            const realNumType S[7] = {sines[activeRots[0]],sines[activeRots[1]],sines[activeRots[2]],sines[activeRots[3]],sines[activeRots[4]],sines[activeRots[5]],sines[activeRots[6]]};
                            const realNumType C[7] = {cosines[activeRots[0]],cosines[activeRots[1]],cosines[activeRots[2]],cosines[activeRots[3]],cosines[activeRots[4]],cosines[activeRots[5]],cosines[activeRots[6]]};
                            realNumType* resultArr[7] = {result[activeRots[0]+i], result[activeRots[1]+i],result[activeRots[2]+i],result[activeRots[3]+i],result[activeRots[4]+i],result[activeRots[5]+i],result[activeRots[6]+i]}; // This could be a const array
                            if constexpr(storeTangent)
                            {
                                realNumType* const tangentStoreArr[7] = {tangentStore[activeRots[0]+i],tangentStore[activeRots[1]+i],tangentStore[activeRots[2]+i],tangentStore[activeRots[3]+i],tangentStore[activeRots[4]+i],tangentStore[activeRots[5]+i],tangentStore[activeRots[6]+i]};
                                storeTangents<indexType,7>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/128);
                            }
                            BENCHMARK_rotate<indexType,7,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/128,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 8:
                        if constexpr (numberToFuse >= 8)
                        {
                            realNumType S[8]; realNumType C[8]; realNumType* resultArr[8]; realNumType* tangentStoreArr[8];
                            for (indexType x = 0; x < 8; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,8>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/256);
                            BENCHMARK_rotate<indexType,8,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/256,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 9:
                        if constexpr (numberToFuse >= 9)
                        {
                            realNumType S[9]; realNumType C[9]; realNumType* resultArr[9]; realNumType* tangentStoreArr[9];
                            for (indexType x = 0; x < 9; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,9>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/512);
                            BENCHMARK_rotate<indexType,9,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/512,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 10:
                        if constexpr (numberToFuse >= 10)
                        {
                            realNumType S[10]; realNumType C[10]; realNumType* resultArr[10]; realNumType* tangentStoreArr[10];
                            for (indexType x = 0; x < 10; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,10>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/1024);
                            BENCHMARK_rotate<indexType,10,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/1024,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 11:
                        if constexpr (numberToFuse >= 11)
                        {
                            realNumType S[11]; realNumType C[11]; realNumType* resultArr[11]; realNumType* tangentStoreArr[11];
                            for (indexType x = 0; x < 11; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,11>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/2048);
                            BENCHMARK_rotate<indexType,11,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/2048,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 12:
                        if constexpr (numberToFuse >= 12)
                        {
                            realNumType S[12]; realNumType C[12]; realNumType* resultArr[12]; realNumType* tangentStoreArr[12];
                            for (indexType x = 0; x < 12; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,12>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/4096);
                            BENCHMARK_rotate<indexType,12,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/4096,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 13:
                        if constexpr (numberToFuse >= 13)
                        {
                            realNumType S[13]; realNumType C[13]; realNumType* resultArr[13]; realNumType* tangentStoreArr[13];
                            for (indexType x = 0; x < 13; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,13>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/8192);
                            BENCHMARK_rotate<indexType,13,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/8192,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 14:
                        if constexpr (numberToFuse >= 14)
                        {
                            realNumType S[14]; realNumType C[14]; realNumType* resultArr[14]; realNumType* tangentStoreArr[14];
                            for (indexType x = 0; x < 14; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,14>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/16384);
                            BENCHMARK_rotate<indexType,14,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/16384,scratchSpacehPsi,resultArr);
                            break;
                        }
                    case 15:
                        if constexpr (numberToFuse >= 15)
                        {
                            realNumType S[15]; realNumType C[15]; realNumType* resultArr[15]; realNumType* tangentStoreArr[15];
                            for (indexType x = 0; x < 15; x++)
                            {
                                S[x] = sines[activeRots[x]];
                                C[x] = cosines[activeRots[x]];
                                resultArr[x] = result[activeRots[x]+i];
                                if constexpr(storeTangent) tangentStoreArr[x] = tangentStore[activeRots[x]+i];
                            }
                            if constexpr(storeTangent)
                                storeTangents<indexType,15>(scratchSpace,currentMap.begin(),currentSigns.begin(),tangentStoreArr,filledSize/32768);
                            BENCHMARK_rotate<indexType,15,localVectorSize,BraketWithTangentOfResult>(
                                scratchSpace,currentSigns.begin(),S,C,filledSize/32768,scratchSpacehPsi,resultArr);
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
                    {
                        startVec[currentMap[idx]] = scratchSpace[idx];
                        if constexpr(BraketWithTangentOfResult)
                            hPsi[currentMap[idx]] = scratchSpacehPsi[idx];
                    }
                }
                // },!parallelise || BraketWithTangentOfResult);
            }
            // if constexpr (!(!parallelise || BraketWithTangentOfResult)) for (indexType activeRotIdx = 1; activeRotIdx < localVectorSize; activeRotIdx++) futs[activeRotIdx-1].wait();
        }
    }
    //Start now contains the full evolution
    auto endTime = std::chrono::high_resolution_clock::now();
    auto __attribute__ ((unused))duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    // logger().log("FusedRotate" + std::to_string(numberToFuse) +" Time taken:",duration);
    if constexpr (!BraketWithTangentOfResult)
    {
        delete[] result;
        // delete toBraKet;
    }
}

#define SetupFuseN(N,dataType)\
case N:\
{\
    constexpr int8_t num = N;\
    void* FAVoid = new fusedAnsatzX<num>(setupFuseN<dataType,num>(v,m_start));\
    m_fusedAnsatzes.push_back(FAVoid);\
    m_fusedSizes.push_back(num);\
    break;\
}

#define SetupFuseNDiagonalMacro(N,dataType)\
case -N:\
{\
    constexpr int8_t num = N;\
    void* FAVoid = new fusedDiagonalAnsatzX<num>(setupFuseNDiagonal<dataType,num>(v,m_start));\
    m_fusedAnsatzes.push_back(FAVoid);\
    m_fusedSizes.push_back(-num);\
    break;\
}

//The class
void FusedEvolve::regenCache()
{
    if constexpr (logTimings) logger().log("build cache");
    //This is an O(n^2) algorithm in the worst case that keeps swapping operators as long as they commute.
    cleanup();
    if constexpr (logTimings)logger().log("Incoming excs are:");
    for (auto& e : m_excs)
    {
        if constexpr (logTimings) fprintf(stderr,"%3.1hhd, %3.1hhd, %3.1hhd, %3.1hhd\n",e[0],e[1],e[2],e[3]);
        assert(e.isDiagonal() == e.hasDiagonal());//Somethign like a^\dagger_1 a^\dagger_2 a_2 a_3 is unhandled
    }


    m_excPerm.resize(m_excs.size());
    std::iota(m_excPerm.begin(),m_excPerm.end(),0);
    if (m_excPerm.size() > 1/*if it is one then nothing matters*/)
    {
        size_t startCommute = 0;
        size_t endCommute = 1;
        while (endCommute < m_excPerm.size())
        {
            //extend the box to be as big as possible
            bool commuteWithAll = true;
            while(commuteWithAll && endCommute < m_excPerm.size())
            {
                for (size_t k = startCommute; k < endCommute; k++)
                {
                    bool canBeGrouped = m_excs[m_excPerm[k]].commutes(m_excs[m_excPerm[endCommute]]);
                    canBeGrouped = canBeGrouped && (m_excs[m_excPerm[k]].isDiagonal() == m_excs[m_excPerm[endCommute]].isDiagonal());//diagonals commute but can only be grouped together.
                    if (!canBeGrouped)
                    {
                        commuteWithAll = false;
                        break;
                    }
                }
                if (commuteWithAll)
                    endCommute++;
            }

            //see if we can move any forwards. This requires it to commute with the box or be diagonal if the box is diagonal, And commute with all subsequent elements
            //E.g. the rearrangement 1234,11,5678->1234,5678,11 is allowed
            //As also                11,22,5678,33-> 11,22,33,5678
            //This is allowed but pointless 11,22,5678,34,->11,22,34,5678

            for (size_t trial = endCommute; trial < m_excPerm.size(); trial++)
            {
                bool commuteWithAll = true;
                for (size_t k = startCommute; k < trial; k++)
                {
                    if (m_excs[m_excPerm[startCommute]].isDiagonal() != m_excs[m_excPerm[trial]].isDiagonal())
                    {//only allow diagonals to match diagonals. 1234 commutes with 55 but we cannot group
                        commuteWithAll = false;
                        break;
                    }
                    if (!(m_excs[m_excPerm[k]].commutes(m_excs[m_excPerm[trial]])))
                    {
                        commuteWithAll = false;
                        break;
                    }
                }
                if (commuteWithAll)
                {
                    //Then trial can be added to the group at endCommute
                    for (size_t k = trial; k > endCommute; k--)
                        std::swap(m_excPerm[k],m_excPerm[k-1]);
                    endCommute++;
                }
            }

            startCommute = endCommute;
            endCommute = startCommute+1;
        }

    }
    m_excInversePerm.assign(m_excs.size(),0);
    for (size_t i = 0; i < m_excs.size(); i++)
    {
        size_t j = m_excPerm[i];
        m_excInversePerm[j] = i;
    }
    m_commuteBoundaries.push_back(0);

    for (long i = 0; i < (long)m_excPerm.size()-1; i++)
    {
        for (size_t j = m_commuteBoundaries.back(); j < (size_t)i; j++ )
        {
            bool canExtend = m_excs[m_excPerm[i]].commutes(m_excs[m_excPerm[j]]);
            canExtend = canExtend && (m_excs[m_excPerm[i]].isDiagonal() == m_excs[m_excPerm[j]].isDiagonal()); //diagonals commute but can only be grouped together

            if (!canExtend)
            {
                m_commuteBoundaries.push_back(i);
                break;
            }
        }
    }
    m_commuteBoundaries.push_back(m_excPerm.size());

    std::vector<stateRotate::exc> excs(m_excs.size());
    std::transform(m_excPerm.begin(),m_excPerm.end(),excs.begin(),[this](size_t i){return m_excs[i];});

    if constexpr (logTimings)logger().log("Permuted to:");

    for (size_t i = 0; i < excs.size(); i++)
    {
        const auto& e = excs[i];
        if constexpr (logTimings)fprintf(stderr,"%2.1zu: %3.1hhd, %3.1hhd, %3.1hhd, %3.1hhd\n",i,e[0],e[1],e[2],e[3]);
    }
    if constexpr (logTimings) logger().log("With boundaries",m_commuteBoundaries);



    for (long i = 0; i < (long)m_commuteBoundaries.size()-1;i++)
    {
        size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];


        std::vector<stateRotate::exc> v(excs.begin() + m_commuteBoundaries[i],excs.begin()+m_commuteBoundaries[i+1]);
        int8_t fuseSize = diff;
        if (diff > maxFuse)
        {
            for (int8_t i = maxFuse;i > 0; i--)
            {
                if (diff % i == 0)
                {
                    fuseSize = i;
                    break;
                }
            }
        }
        if (m_excs[m_excPerm[m_commuteBoundaries[i]]].isDiagonal())
            fuseSize *= -1;
        if constexpr (logTimings) logger().log("Fused",fuseSize);
        switch (fuseSize)
        {
            //Negative means diagonal
            SetupFuseNDiagonalMacro(1,uint8_t);
            SetupFuseNDiagonalMacro(2,uint8_t);
            SetupFuseNDiagonalMacro(3,uint8_t);
            SetupFuseNDiagonalMacro(4,uint8_t);
            SetupFuseNDiagonalMacro(5,uint8_t);
            SetupFuseNDiagonalMacro(6,uint8_t);
            SetupFuseNDiagonalMacro(7,uint8_t);
            SetupFuseNDiagonalMacro(8,uint16_t);
            SetupFuseNDiagonalMacro(9,uint16_t);
            SetupFuseNDiagonalMacro(10,uint16_t);
            SetupFuseNDiagonalMacro(11,uint16_t);
            SetupFuseNDiagonalMacro(12,uint16_t);
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;
            SetupFuseN(1,uint8_t);
            SetupFuseN(2,uint8_t);
            SetupFuseN(3,uint8_t);
            SetupFuseN(4,uint8_t);
            SetupFuseN(5,uint8_t);
            SetupFuseN(6,uint8_t);
            SetupFuseN(7,uint8_t);
            SetupFuseN(8,uint16_t);
            SetupFuseN(9,uint16_t);
            SetupFuseN(10,uint16_t);
            SetupFuseN(11,uint16_t);
            SetupFuseN(12,uint16_t);
        default:
            __builtin_trap();
            static_assert(maxFuse <=12);
        }
    }
    m_excsCached = true;

}

void FusedEvolve::cleanup()
{
    for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
    {
        switch (m_fusedSizes[i])
        {
        case -1:
            delete static_cast<fusedDiagonalAnsatzX<1>*>(m_fusedAnsatzes[i]);
            break;
        case -2:
            delete static_cast<fusedDiagonalAnsatzX<2>*>(m_fusedAnsatzes[i]);
            break;
        case -3:
            delete static_cast<fusedDiagonalAnsatzX<3>*>(m_fusedAnsatzes[i]);
            break;
        case -4:
            delete static_cast<fusedDiagonalAnsatzX<4>*>(m_fusedAnsatzes[i]);
            break;
        case -5:
            delete static_cast<fusedDiagonalAnsatzX<5>*>(m_fusedAnsatzes[i]);
            break;
        case -6:
            delete static_cast<fusedDiagonalAnsatzX<6>*>(m_fusedAnsatzes[i]);
            break;
        case -7:
            delete static_cast<fusedDiagonalAnsatzX<7>*>(m_fusedAnsatzes[i]);
            break;
        case -8:
            delete static_cast<fusedDiagonalAnsatzX<8>*>(m_fusedAnsatzes[i]);
            break;
        case -9:
            delete static_cast<fusedDiagonalAnsatzX<9>*>(m_fusedAnsatzes[i]);
            break;
        case -10:
            delete static_cast<fusedDiagonalAnsatzX<10>*>(m_fusedAnsatzes[i]);
            break;
        case -11:
            delete static_cast<fusedDiagonalAnsatzX<11>*>(m_fusedAnsatzes[i]);
            break;
        case -12:
            delete static_cast<fusedDiagonalAnsatzX<12>*>(m_fusedAnsatzes[i]);
            break;
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;
        case 1:
            delete static_cast<fusedAnsatzX<1>*>(m_fusedAnsatzes[i]);
            break;
        case 2:
            delete static_cast<fusedAnsatzX<2>*>(m_fusedAnsatzes[i]);
            break;
        case 3:
            delete static_cast<fusedAnsatzX<3>*>(m_fusedAnsatzes[i]);
            break;
        case 4:
            delete static_cast<fusedAnsatzX<4>*>(m_fusedAnsatzes[i]);
            break;
        case 5:
            delete static_cast<fusedAnsatzX<5>*>(m_fusedAnsatzes[i]);
            break;
        case 6:
            delete static_cast<fusedAnsatzX<6>*>(m_fusedAnsatzes[i]);
            break;
        case 7:
            delete static_cast<fusedAnsatzX<7>*>(m_fusedAnsatzes[i]);
            break;
        case 8:
            delete static_cast<fusedAnsatzX<8>*>(m_fusedAnsatzes[i]);
            break;
        case 9:
            delete static_cast<fusedAnsatzX<9>*>(m_fusedAnsatzes[i]);
            break;
        case 10:
            delete static_cast<fusedAnsatzX<10>*>(m_fusedAnsatzes[i]);
            break;
        case 11:
            delete static_cast<fusedAnsatzX<11>*>(m_fusedAnsatzes[i]);
            break;
        case 12:
            delete static_cast<fusedAnsatzX<12>*>(m_fusedAnsatzes[i]);
            break;
        default:
            __builtin_trap();
        }
    }
    m_commuteBoundaries.clear();
    m_fusedAnsatzes.clear();
    m_excPerm.clear();
    m_excInversePerm.clear();
}

FusedEvolve::FusedEvolve(const vector<numType> &start, std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham,
                         Eigen::SparseMatrix<realNumType, Eigen::RowMajor> compressMatrix, Eigen::SparseMatrix<realNumType, Eigen::RowMajor> deCompressMatrix)
{
    m_start.copy(start);
    m_Ham = Ham;
    m_lieIsCompressed = m_start.getIsCompressed(m_compressor);
    m_compressMatrix = std::move(compressMatrix);
    m_compressMatrixPsi.resize(m_compressMatrix.rows()+1,m_compressMatrix.cols()+1);
    m_compressMatrixPsi.middleRows(0,m_compressMatrix.rows()) = m_compressMatrix.middleRows(0,m_compressMatrix.rows());

    m_compressMatrixPsi.coeffRef(m_compressMatrix.rows(),m_compressMatrix.cols()) = 1;
    m_deCompressMatrix = std::move(deCompressMatrix);
}

FusedEvolve::~FusedEvolve()
{
    cleanup();
}

void FusedEvolve::updateExc(const std::vector<stateRotate::exc>& excs)
{
    if (excs != m_excs)
    {
        m_excs = excs;
        m_excsCached = false;
    }
}
#define Evolve(N,dataType)\
case N:\
{\
    constexpr uint8_t num = N;\
    fusedAnsatzX<num>* FA =  static_cast<fusedAnsatzX<num>*>(m_fusedAnsatzes[i]);\
    RunFuseN<dataType,num,false,false,true>(FA->data(),(realNumType*)&dest[0],permAngles.data() + m_commuteBoundaries[i],diff);\
    break;\
}

#define EvolveDiagonal(N,dataType)\
case -N:\
{\
        constexpr uint8_t num = N;\
        fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i]);\
        RunFuseNDiagonal<dataType,num,false,false,true>(FA->data(),(realNumType*)&dest[0],permAngles.data() + m_commuteBoundaries[i],diff);\
        break;\
}

void FusedEvolve::evolve(vector<numType>& dest, const std::vector<realNumType>& angles, vector<numType>* specifiedStart)
{
    if (!m_excsCached)
        regenCache();
    if (specifiedStart == nullptr)
        dest.copy(m_start);
    else
        dest.copy(*specifiedStart);

    std::vector<realNumType> permAngles(angles.size());
    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return angles[i];});

    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
    {
        size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
        switch (m_fusedSizes[i])
        {
        EvolveDiagonal(1,uint8_t)
        EvolveDiagonal(2,uint8_t)
        EvolveDiagonal(3,uint8_t)
        EvolveDiagonal(4,uint8_t)
        EvolveDiagonal(5,uint8_t)
        EvolveDiagonal(6,uint8_t)
        EvolveDiagonal(7,uint8_t)
        EvolveDiagonal(8,uint16_t)
        EvolveDiagonal(9,uint16_t)
        EvolveDiagonal(10,uint16_t)
        EvolveDiagonal(11,uint16_t)
        EvolveDiagonal(12,uint16_t)
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;
        Evolve(1,uint8_t)
        Evolve(2,uint8_t)
        Evolve(3,uint8_t)
        Evolve(4,uint8_t)
        Evolve(5,uint8_t)
        Evolve(6,uint8_t)
        Evolve(7,uint8_t)
        Evolve(8,uint16_t)
        Evolve(9,uint16_t)
        Evolve(10,uint16_t)
        Evolve(11,uint16_t)
        Evolve(12,uint16_t)

        default:
            __builtin_trap();
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    if constexpr (logTimings) logger().log("FusedEvolve Time taken:",duration);
}

void FusedEvolve::evolveMultiple(Matrix<numType> &destMatrix, const Matrix<realNumType>::EigenMatrix &anglesMatrix, vector<numType> *specifiedStart)
{
    if (!m_excsCached)
        regenCache();
    destMatrix.resize(anglesMatrix.rows(),specifiedStart == nullptr ? m_start.size() : specifiedStart->size(),m_lieIsCompressed,m_compressor,false);
    for (long i = 0; i < anglesMatrix.rows(); i++)
    {
        if (specifiedStart == nullptr)
            destMatrix.getJVectorView(i).copy(m_start);
        else
            destMatrix.getJVectorView(i).copy(*specifiedStart);
    }

    Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor> permAnglesMatrix(anglesMatrix.rows(),anglesMatrix.cols());

    // std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return angles[i];});
    for (size_t i = 0; i < m_excPerm.size(); i++)
    {
        permAnglesMatrix.col(i) = anglesMatrix.col(m_excPerm[i]);
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> futs;
    threadpool& pool = threadpool::getInstance(NUM_CORES);
    long stepSize = std::max(permAnglesMatrix.rows()/(long)NUM_CORES,1l);
    for (long angleIdx = 0; angleIdx < permAnglesMatrix.rows(); angleIdx += stepSize)
    {
        long angleIdxStop = std::min(angleIdx + stepSize,permAnglesMatrix.rows());
        futs.push_back(pool.queueWork(
        [this,&destMatrix,&permAnglesMatrix,angleIdx,angleIdxStop]()
          {
                for (long idx = angleIdx; idx < angleIdxStop; idx++)
                {
                    auto dest = destMatrix.getJVectorView(idx);
                    auto permAngles = Eigen::Map<const Eigen::Matrix<realNumType,1,-1,Eigen::RowMajor>>(&permAnglesMatrix(idx,0),1,permAnglesMatrix.cols());
                    for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
                    {
                        size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
                        switch (m_fusedSizes[i])
                        {
                            EvolveDiagonal(1,uint8_t)
                            EvolveDiagonal(2,uint8_t)
                            EvolveDiagonal(3,uint8_t)
                            EvolveDiagonal(4,uint8_t)
                            EvolveDiagonal(5,uint8_t)
                            EvolveDiagonal(6,uint8_t)
                            EvolveDiagonal(7,uint8_t)
                            EvolveDiagonal(8,uint16_t)
                            EvolveDiagonal(9,uint16_t)
                            EvolveDiagonal(10,uint16_t)
                            EvolveDiagonal(11,uint16_t)
                            EvolveDiagonal(12,uint16_t)
                        case 0:
                            logger().log("Unhandled case 0");
                            __builtin_trap();
                            break;

                            Evolve(1,uint8_t)
                            Evolve(2,uint8_t)
                            Evolve(3,uint8_t)
                            Evolve(4,uint8_t)
                            Evolve(5,uint8_t)
                            Evolve(6,uint8_t)
                            Evolve(7,uint8_t)
                            Evolve(8,uint16_t)
                            Evolve(9,uint16_t)
                            Evolve(10,uint16_t)
                            Evolve(11,uint16_t)
                            Evolve(12,uint16_t)

                        default:
                            __builtin_trap();
                        }
                    }
                }
          }));
    }
    for (auto& f : futs)
        f.wait();
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    if constexpr (logTimings) logger().log("FusedEvolveMultiple Time taken:",duration);
}


#define EvolveDer(N,dataType)\
case N:\
{\
    constexpr uint8_t num = N;\
    fusedAnsatzX<num>* FA =  static_cast<fusedAnsatzX<num>*>(m_fusedAnsatzes[i-1]);\
    RunFuseN<dataType,num,true,false>(FA->data(),(realNumType*)&dest[0],permAngles.data() + m_commuteBoundaries[i-1],diff,(realNumType*)&hPsi[0],derivLocs.data() + m_commuteBoundaries[i-1]);\
    break;\
}

#define EvolveDerDiag(N,dataType)\
case -N:\
{\
        constexpr uint8_t num = N;\
        fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i-1]);\
        RunFuseNDiagonal<dataType,num,true,false>(FA->data(),(realNumType*)&dest[0],permAngles.data() + m_commuteBoundaries[i-1],diff,(realNumType*)&hPsi[0],derivLocs.data() + m_commuteBoundaries[i-1]);\
        break;\
}
void FusedEvolve::evolveDerivative(const vector<numType> &finalVector, vector<realNumType>& deriv, const std::vector<realNumType> &angles, realNumType* Energy)
{
    static_assert(std::is_same_v<realNumType,numType> || std::is_same_v<std::complex<realNumType>,numType>);//we do magic bithacking so need to assure this
    if (!m_excsCached)
        regenCache();
        // regenCache();
    auto startTime = std::chrono::high_resolution_clock::now();
    deriv.resize(angles.size(),false,nullptr); // need memset version


    std::vector<realNumType> permAngles(angles.size());
    std::vector<realNumType*> derivLocs(angles.size());
    //Negative of the angles because we evolve backwards
    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return -angles[i];});
    std::transform(m_excPerm.begin(),m_excPerm.end(),derivLocs.begin(),[&deriv](size_t i){return &deriv[i];});

    //finalVector = |Psi>. we aim to calculate <Psi|H | d/dtheta_i Psi>.
    //Both <Psi|H and Psi> are evolved backwards and the dot product calculated on the fly
    //Note that we go backwards in the fusedAnsatzes. but each fusion is still evolved forwards. This is fine because its commutes.
    vector<numType> dest;
    dest.copy(finalVector);
    // logger().log("dest.dot start Before deriv", dest.dot(m_start)); // this should not be 1

    vector<numType> hPsi;
    {
        // hPsi.resize(finalVector.size(),m_lieIsCompressed,m_compressor,false);
        // Eigen::Map<const Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> currentMap(&finalVector.at(0,0),finalVector.m_iSize,finalVector.m_jSize);
        // Eigen::Map<Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> destMap(&hPsi.at(0,0),hPsi.m_iSize,hPsi.m_jSize);
        // destMap.noalias() = currentMap*m_HamEm;
        m_Ham->apply(finalVector,hPsi);
    }
    if (Energy)
        *Energy = hPsi.dot(finalVector);



    for (size_t i = m_fusedAnsatzes.size(); i > 0; i--)
    {
        size_t diff = m_commuteBoundaries[i] -m_commuteBoundaries[i-1];
        switch (m_fusedSizes[i-1])
        {
            EvolveDerDiag(1,uint8_t);
            EvolveDerDiag(2,uint8_t);
            EvolveDerDiag(3,uint8_t);
            EvolveDerDiag(4,uint8_t);
            EvolveDerDiag(5,uint8_t);
            EvolveDerDiag(6,uint8_t);
            EvolveDerDiag(7,uint8_t);
            EvolveDerDiag(8,uint16_t);
            EvolveDerDiag(9,uint16_t);
            EvolveDerDiag(10,uint16_t);
            EvolveDerDiag(11,uint16_t);
            EvolveDerDiag(12,uint16_t);
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;
            EvolveDer(1,uint8_t);
            EvolveDer(2,uint8_t);
            EvolveDer(3,uint8_t);
            EvolveDer(4,uint8_t);
            EvolveDer(5,uint8_t);
            EvolveDer(6,uint8_t);
            EvolveDer(7,uint8_t);
            EvolveDer(8,uint16_t);
            EvolveDer(9,uint16_t);
            EvolveDer(10,uint16_t);
            EvolveDer(11,uint16_t);
            EvolveDer(12,uint16_t);
        default:
            __builtin_trap();
        }
    }
    deriv*=2;
    Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> gradVector_mu(&deriv[0],deriv.size(),1);
    Eigen::VectorXd derivCompressed = m_compressMatrix * gradVector_mu;
    deriv.copyFromBuffer(derivCompressed.data(),derivCompressed.rows());
    // logger().log("dest.dot start deriv", dest.dot(m_start)); // this should be 1
    // logger().log("hpsi.dot start deriv", hPsi.dot(m_start)); // this should be E
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    if constexpr (logTimings) logger().log("DerivTimeTaken (ms)", duration);

}
#define GenerateTangents(N, dataType)\
case N:\
{\
    constexpr uint8_t num = N;\
    fusedAnsatzX<num>* FA =  static_cast<fusedAnsatzX<num>*>(m_fusedAnsatzes[i]);\
    RunFuseN<dataType,num,false,true>(FA->data(),(realNumType*)&psi[0],permAngles.data() + m_commuteBoundaries[i],diff,nullptr,nullptr,(realNumType**)(TPtrs.data() + m_commuteBoundaries[i]));\
break;\
}

#define GenerateTangentsDiag(N, dataType)\
case -N:\
{\
        constexpr uint8_t num = N;\
        fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i]);\
        RunFuseNDiagonal<dataType,num,false,true>(FA->data(),(realNumType*)&psi[0],permAngles.data() + m_commuteBoundaries[i],diff,nullptr,nullptr,(realNumType**)(TPtrs.data() + m_commuteBoundaries[i]));\
        break;\
}

#define EvolveTangents(N,dataType)\
case N:\
{\
    constexpr uint8_t num = N;\
    fusedAnsatzX<num>* FA =  static_cast<fusedAnsatzX<num>*>(m_fusedAnsatzes[i]);\
    if (ang < m_commuteBoundaries[i])\
    {\
        RunFuseN<dataType,num,false,false>(FA->data(),(realNumType*)TPtrs[ang],permAngles.data() + m_commuteBoundaries[i],diff);\
    }\
    else\
    {\
        for (size_t currDiff = 0; currDiff < diff; currDiff += num)\
        {\
            numType** startTPtr = TPtrs.data() + m_commuteBoundaries[i] + currDiff;\
            for (numType** TPtr = startTPtr; TPtr < startTPtr + num; TPtr++)\
            {\
                if (*TPtr == TPtrs[ang])\
                {\
                    RunFuseN<dataType,num,false,false>(FA->data() + currDiff/num,(realNumType*)*TPtr,permAngles.data() + m_commuteBoundaries[i] + currDiff,diff-currDiff);\
                    break;\
                }\
            }\
        }\
    }\
    break;\
}

#define EvolveTangentsDiag(N,dataType)\
case -N:\
{\
        constexpr uint8_t num = N;\
        fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i]);\
        if (ang < m_commuteBoundaries[i])\
    {\
            RunFuseNDiagonal<dataType,num,false,false>(FA->data(),(realNumType*)TPtrs[ang],permAngles.data() + m_commuteBoundaries[i],diff);\
    }\
        else\
    {\
            for (size_t currDiff = 0; currDiff < diff; currDiff += num)\
        {\
                numType** startTPtr = TPtrs.data() + m_commuteBoundaries[i] + currDiff;\
                for (numType** TPtr = startTPtr; TPtr < startTPtr + num; TPtr++)\
            {\
                    if (*TPtr == TPtrs[ang])\
                {\
                        RunFuseNDiagonal<dataType,num,false,false>(FA->data() + currDiff/num,(realNumType*)*TPtr,permAngles.data() + m_commuteBoundaries[i] + currDiff,diff-currDiff);\
                        break;\
                }\
            }\
        }\
    }\
        break;\
}

#define GenerateHPsiT(N,dataType)\
case N:\
{\
constexpr uint8_t num = N;\
fusedAnsatzX<num>* FA =  static_cast<fusedAnsatzX<num>*>(m_fusedAnsatzes[i-1]);\
RunFuseN<dataType,num,true,false>(FA->data(),(realNumType*)&Ts.at(angleIdx,0),permAngles.data() + m_commuteBoundaries[i-1],diff,(realNumType*)&localHpsi[0],derivLocs.data() + m_commuteBoundaries[i-1]);\
break;\
}

#define GenerateHPsiTDiagonal(N,dataType)\
case -N:\
{\
        constexpr uint8_t num = N;\
        fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i-1]);\
        RunFuseNDiagonal<dataType,num,true,false>(FA->data(),(realNumType*)&Ts.at(angleIdx,0),permAngles.data() + m_commuteBoundaries[i-1],diff,(realNumType*)&localHpsi[0],derivLocs.data() + m_commuteBoundaries[i-1]);\
        break;\
}

void FusedEvolve::evolveHessian(Eigen::MatrixXd &Hessian, vector<realNumType>& derivCompressed,const std::vector<realNumType> &angles, Eigen::Matrix<numType,-1,-1>* TsCopy, realNumType* Energy)
{
    //TODO
    // constexpr bool isComplex = !std::is_same_v<realNumType,numType>;
    if (!m_excsCached)
        regenCache();
    //There are two types of quantities we need
    // T means a tangent
    // <T|H|T> and <Psi|H|TT>
    //THT is calculated via back evolution of <T|H| and |Psi>
    // <Psi|H|TT> is calculated via back evolution of <Psi|H| and |T>
    //But this means |T> and therefore <T|H| are known at the end. So we can calculate <T|H|T> for `free'

    //A not memory efficient version.
    vector<numType> hPsi;
    // vector<numType> psi;
    Matrix<numType> Ts;
    // Matrix<numType> HTs;
    Ts.resize(angles.size()+1,m_start.size(),m_lieIsCompressed,m_compressor);
    //The last one is psi
    vectorView<Matrix<numType>,Eigen::RowMajor> psi = Ts.getJVectorView(angles.size());
    // HTs.resize(angles.size(),m_start.size(),m_lieIsCompressed,m_compressor);
    psi.copy(m_start);
    hPsi.resize(psi.size(),m_lieIsCompressed,m_compressor,false);
    Eigen::MatrixXd THT;
    Hessian.resize(angles.size(),angles.size());
    Hessian.setZero();

    //Get the |T>s
    //We could save one evolution by getting psi aswell here

    std::vector<realNumType> permAngles(angles.size());
    std::vector<numType*> TPtrs(angles.size());
    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return angles[i];});
    std::transform(m_excPerm.begin(),m_excPerm.end(),TPtrs.begin(),[&Ts](size_t i){return &Ts.at(i,0);});

    //Do an evolution and gather all the Ts
    //The T ptrs are still at their respective evolutions. They need to be evolved to the end aswell


    auto time1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
    {
        size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
        switch (m_fusedSizes[i])
        {
        GenerateTangentsDiag(1,uint8_t)
        GenerateTangentsDiag(2,uint8_t)
        GenerateTangentsDiag(3,uint8_t)
        GenerateTangentsDiag(4,uint8_t)
        GenerateTangentsDiag(5,uint8_t)
        GenerateTangentsDiag(6,uint8_t)
        GenerateTangentsDiag(7,uint8_t)
        GenerateTangentsDiag(8,uint16_t)
        GenerateTangentsDiag(9,uint16_t)
        GenerateTangentsDiag(10,uint16_t)
        GenerateTangentsDiag(11,uint16_t)
        GenerateTangentsDiag(12,uint16_t)
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;

        GenerateTangents(1,uint8_t)
        GenerateTangents(2,uint8_t)
        GenerateTangents(3,uint8_t)
        GenerateTangents(4,uint8_t)
        GenerateTangents(5,uint8_t)
        GenerateTangents(6,uint8_t)
        GenerateTangents(7,uint8_t)
        GenerateTangents(8,uint16_t)
        GenerateTangents(9,uint16_t)
        GenerateTangents(10,uint16_t)
        GenerateTangents(11,uint16_t)
        GenerateTangents(12,uint16_t)
        default:
            __builtin_trap();
        }

    }
    std::vector<std::future<void>> futs;
    threadpool& pool = threadpool::getInstance(NUM_CORES);
    for (size_t ang = 0; ang < angles.size(); ang++)
    {
        futs.push_back(pool.queueWork([ang,this,&TPtrs,&permAngles]()
          {
              for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
              {
                  size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
                  if (ang >= m_commuteBoundaries[i+1])
                      continue;
                  switch (m_fusedSizes[i])
                  {
                  // EvolveTangentsDiag(1,uint8_t)
                  case -1:
                  {
                      constexpr uint8_t num = 1;
                      fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i]);
                      if (ang < m_commuteBoundaries[i])
                      {
                          RunFuseNDiagonal<uint8_t,num,false,false>(FA->data(),(realNumType*)TPtrs[ang],permAngles.data() + m_commuteBoundaries[i],diff);
                      }
                      else
                      {
                          for (size_t currDiff = 0; currDiff < diff; currDiff += num)
                          {
                              numType** startTPtr = TPtrs.data() + m_commuteBoundaries[i] + currDiff;
                              for (numType** TPtr = startTPtr; TPtr < startTPtr + num; TPtr++)
                              {
                                  if (*TPtr == TPtrs[ang])
                                  {
                                      RunFuseNDiagonal<uint8_t,num,false,false>(FA->data() + currDiff/num,(realNumType*)*TPtr,permAngles.data() + m_commuteBoundaries[i] + currDiff,diff-currDiff);
                                      break;
                                  }
                              }
                          }
                      }
                      break;
                  }
                  EvolveTangentsDiag(2,uint8_t)
                  EvolveTangentsDiag(3,uint8_t)
                  EvolveTangentsDiag(4,uint8_t)
                  EvolveTangentsDiag(5,uint8_t)
                  EvolveTangentsDiag(6,uint8_t)
                  EvolveTangentsDiag(7,uint8_t)
                  EvolveTangentsDiag(8,uint16_t)
                  EvolveTangentsDiag(9,uint16_t)
                  EvolveTangentsDiag(10,uint16_t)
                  EvolveTangentsDiag(11,uint16_t)
                  EvolveTangentsDiag(12,uint16_t)
                  case 0:
                      logger().log("Unhandled case 0");
                      __builtin_trap();
                      break;
                  EvolveTangents(1,uint8_t)
                  EvolveTangents(2,uint8_t)
                  EvolveTangents(3,uint8_t)
                  EvolveTangents(4,uint8_t)
                  EvolveTangents(5,uint8_t)
                  EvolveTangents(6,uint8_t)
                  EvolveTangents(7,uint8_t)
                  EvolveTangents(8,uint16_t)
                  EvolveTangents(9,uint16_t)
                  EvolveTangents(10,uint16_t)
                  EvolveTangents(11,uint16_t)
                  EvolveTangents(12,uint16_t)
                  default:
                      __builtin_trap();
                  }

              }
          }
          ));
    }
    for (auto& f : futs)
            f.wait();
    futs.clear();
    auto time2 = std::chrono::high_resolution_clock::now();

    // compute hPsi
    {

        // Eigen::Map<const Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> currentMap(&psi.at(0,0),psi.m_iSize,psi.m_jSize);
        // Eigen::Map<Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> destMap(&hPsi.at(0,0),hPsi.m_iSize,hPsi.m_jSize);
        // destMap.noalias() = currentMap*m_HamEm;

        // m_Ham->apply(psi,hPsi); // This is now done the same time as the Ts
        //psi no longer needed
    }
    auto time3 = std::chrono::high_resolution_clock::now();
    // compute HTs
    {
        Eigen::Map<const Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> TMap(&Ts.at(0,0),Ts.m_iSize,Ts.m_jSize);
        if (TsCopy)
        {
            Eigen::Map<const Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> TMapOnly(&Ts.at(0,0),angles.size(),Ts.m_jSize);
            *TsCopy = TMapOnly.transpose();
        }
        // Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> HTMap(&HTs.at(0,0),HTs.m_iSize,HTs.m_jSize);
        //For speed need to convert the ordering
        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> TMapC = TMap;
        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> T_C;
        T_C.noalias() = m_compressMatrixPsi * TMapC;

        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> HT_C;
        // HT_C.noalias() = T_C*m_HamEm;
        m_Ham->apply(T_C,HT_C);
        Eigen::Map<Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> hPsiMap(&hPsi.at(0,0),hPsi.m_iSize,hPsi.m_jSize);
        //extract hPsi
        hPsiMap.noalias() = HT_C.row(HT_C.rows()-1);

        //We can now compute the <THT> terms
        // Hessian.triangularView<Eigen::Lower>() += 2*TMap * HT_C.transpose();
        //TODO bithacks
        //Need to lie to eigen since .topRows is slow so instead we compute the extra overlap. This generates <psi|hpsi>, along with <psi|T> i.e. energy and derivative
        Eigen::MatrixXd temp;
        temp.noalias() = (2*T_C * HT_C.adjoint()).real();

        THT.resize(temp.rows()-1,temp.cols()-1);
        derivCompressed.resize(temp.rows()-1,false,nullptr);

        for (long i = 0; i < temp.rows()-1; i++)
        {
            for (long j = i; j < temp.cols()-1; j++)
            {
                THT(i,j) = temp(i,j);
                THT(j,i) = temp(i,j);
            }
        }
        for (long i = 0; i < temp.rows()-1; i++)
        {
            derivCompressed[i] = temp(i,temp.cols()-1);
        }
        if (Energy)
            *Energy = temp(temp.rows()-1,temp.cols()-1)/2;
        assert(THT.rows() == m_compressMatrix.rows() && THT.cols() == m_compressMatrix.rows());
    }
    auto time4 = std::chrono::high_resolution_clock::now();
    //Compute <Psi|H|T> = deriv
    // {
    //     Eigen::Map<Eigen::Matrix<realNumType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> derivMap(&deriv.at(0,0),deriv.m_iSize,deriv.m_jSize);

    //     Eigen::Map<const Eigen::Matrix<realNumType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> hPsiMap((realNumType*)&hPsi.at(0,0),hPsi.m_iSize,hPsi.m_jSize*(isComplex ? 2 : 1));
    //     Eigen::Map<const Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> TMap((realNumType*)&Ts.at(0,0),angles.size(),Ts.m_jSize*(isComplex ? 2 : 1));
    //     derivMap.noalias() = 2*hPsiMap*TMap.transpose();
    //     if (Energy)
    //     {
    //         *Energy = std::real(hPsiMap.dot(TMap.row(angles.size())));
    //     }
    // }
    auto time5 = std::chrono::high_resolution_clock::now();
    //Remains to do the backwards evolution of |H|Psi> and |T>
    //This is basically the same as evolveDerivative

    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return -angles[i];});
    size_t stepSize = 1;//std::max(angles.size()/NUM_CORES,1ul);
    for (size_t angleIdxStart = 0; angleIdxStart < angles.size(); angleIdxStart+=stepSize)
    {
        size_t angleIdxEnd = std::min(angleIdxStart+stepSize,angles.size());
        futs.push_back(pool.queueWork([angleIdxStart,angleIdxEnd,&Hessian,&permAngles,&hPsi,this,&Ts]()
          {
              for (size_t angleIdx = angleIdxStart; angleIdx < angleIdxEnd; angleIdx++)
              {
                  std::vector<realNumType> scratchPad;
                  vector<numType> localHpsi;
                  std::vector<realNumType*> derivLocs(permAngles.size());

                  //Negative of the angles because we evolve backwards

                  scratchPad.assign(permAngles.size(),0);

                  localHpsi.copy(hPsi);


                  std::transform(m_excPerm.begin(),m_excPerm.end(),derivLocs.begin(),[&scratchPad](size_t i){return &scratchPad[i];});

                  //We only need to evaluate if m_excPerm(angleIdx) < m_commuteBoundaries[i];
                  //To see this note that we could construct the hessian in permuted indexes
                  //Then we only want to construct if permAngleIDX <= m_commuteBoundaries[i]
                  //This will evaluate each element only once. problem is that diff may go over the boundary. So that needs to be fixed up afterwards.
                  //Further knowing the symmetrical version is annoying so we symmetrise here
                  size_t permAngleIdx = m_excInversePerm[angleIdx];
                  for (size_t i = m_fusedAnsatzes.size(); i > 0; i--)
                  {
                      //            start                     end
                      size_t diff = m_commuteBoundaries[i] -m_commuteBoundaries[i-1];
                      if (!(permAngleIdx <= m_commuteBoundaries[i]))
                          break;
                      switch (m_fusedSizes[i-1])
                      {
                      GenerateHPsiTDiagonal(1,uint8_t)
                      GenerateHPsiTDiagonal(2,uint8_t)
                      GenerateHPsiTDiagonal(3,uint8_t)
                      GenerateHPsiTDiagonal(4,uint8_t)
                      GenerateHPsiTDiagonal(5,uint8_t)
                      GenerateHPsiTDiagonal(6,uint8_t)
                      GenerateHPsiTDiagonal(7,uint8_t)
                      GenerateHPsiTDiagonal(8,uint16_t)
                      GenerateHPsiTDiagonal(9,uint16_t)
                      GenerateHPsiTDiagonal(10,uint16_t)
                      GenerateHPsiTDiagonal(11,uint16_t)
                      GenerateHPsiTDiagonal(12,uint16_t)
                      case 0:
                      {
                          logger().log("Unhandled case 0");
                          __builtin_trap();
                          break;
                      }
                      GenerateHPsiT(1,uint8_t)
                      GenerateHPsiT(2,uint8_t)
                      GenerateHPsiT(3,uint8_t)
                      GenerateHPsiT(4,uint8_t)
                      GenerateHPsiT(5,uint8_t)
                      GenerateHPsiT(6,uint8_t)
                      GenerateHPsiT(7,uint8_t)
                      GenerateHPsiT(8,uint16_t)
                      GenerateHPsiT(9,uint16_t)
                      GenerateHPsiT(10,uint16_t)
                      GenerateHPsiT(11,uint16_t)
                      GenerateHPsiT(12,uint16_t)


                      default:
                          __builtin_trap();
                      }
                      //Store hessian
                      for (size_t n = 0; n < diff; n++)
                      {
                          if (permAngleIdx <= m_commuteBoundaries[i-1] + n)
                          {
                              Hessian(angleIdx,m_excPerm[m_commuteBoundaries[i-1] + n]) += 2**derivLocs[m_commuteBoundaries[i-1] + n];
                              if (permAngleIdx != m_commuteBoundaries[i-1] + n)
                                  Hessian(m_excPerm[m_commuteBoundaries[i-1] + n],angleIdx) += 2**derivLocs[m_commuteBoundaries[i-1] + n];
                          }

                      }
                  }
              }
          }
        ));
    }
    for (auto& f : futs)
        f.wait();

    Hessian = m_compressMatrix * Hessian * m_compressMatrix.transpose();
    Hessian += THT;
    auto time6 = std::chrono::high_resolution_clock::now();
    long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    long duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
    long duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(time4-time3).count();
    long duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(time5-time4).count();
    long duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();
    if constexpr (logTimings)
    {
        logger().log("FusedEvolve Hessian Time taken 1 (ms)",duration1);
        logger().log("FusedEvolve Hessian Time taken 2 (ms)",duration2);
        logger().log("FusedEvolve Hessian Time taken 3 (ms)",duration3);
        logger().log("FusedEvolve Hessian Time taken 4 (ms)",duration4);
        logger().log("FusedEvolve Hessian Time taken 5 (ms)",duration5);
    }
    // Eigen::MatrixXd M = Hessian - Hessian.transpose();
    // logger().log("Symmetric?:",(Hessian - Hessian.transpose()).norm());
}

realNumType FusedEvolve::getEnergy(const vector<numType> &psi)
{

    // Eigen::Map<const Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> currentMap(&psi.at(0,0),psi.m_iSize,psi.m_jSize);
    // Eigen::Matrix<numType,1,-1,Eigen::RowMajor> hPsi;
    auto start = std::chrono::high_resolution_clock::now();
    vector<numType> hPsi;
    // hPsi.noalias() = currentMap*m_HamEm;
    realNumType E =  m_Ham->apply(psi,hPsi).dot(psi);
    auto end = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    if constexpr (logTimings) logger().log("FusedEvolve Energy Time taken 1 (ms)",duration);
    return E;
}

vector<realNumType> FusedEvolve::getEnergies(const Matrix<numType> &psi)
{
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::Map<const Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> currentMap(&psi.at(0,0),psi.m_iSize,psi.m_jSize);
    Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> hPsi(psi.m_iSize,psi.m_jSize);
    Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> hPsiMap(hPsi.data(),hPsi.rows(),hPsi.cols());
    m_Ham->apply(currentMap,hPsiMap);
    vector<realNumType> ret(psi.m_iSize);
    Eigen::Map<Eigen::Matrix<realNumType,1,-1,Eigen::RowMajor>,Eigen::Aligned32>EMap(&ret[0],1,ret.size());
    EMap = (hPsi * currentMap.transpose()).diagonal().real();

    auto end = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    if constexpr (logTimings) logger().log("FusedEvolve Energies Time taken 1 (ms)",duration);
    return ret;
}

void FusedEvolve::evolveDerivativeProj(const vector<numType> &finalVector, vector<realNumType> &deriv, const std::vector<realNumType> &angles, const vector<numType> &projVector, realNumType *Energy)
{
    static_assert(std::is_same_v<realNumType,numType> || std::is_same_v<std::complex<realNumType>,numType>);//we do magic bithacking so need to assure this
    if (!m_excsCached)
        regenCache();
    // regenCache();
    auto startTime = std::chrono::high_resolution_clock::now();
    deriv.resize(angles.size(),false,nullptr); // need memset version


    std::vector<realNumType> permAngles(angles.size());
    std::vector<realNumType*> derivLocs(angles.size());
    //Negative of the angles because we evolve backwards
    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return -angles[i];});
    std::transform(m_excPerm.begin(),m_excPerm.end(),derivLocs.begin(),[&deriv](size_t i){return &deriv[i];});

    //finalVector = |Psi>. we aim to calculate <Psi|H | d/dtheta_i Psi>.
    //Both <Psi|H and Psi> are evolved backwards and the dot product calculated on the fly
    //Note that we go backwards in the fusedAnsatzes. but each fusion is still evolved forwards. This is fine because its commutes.
    vector<numType> dest;
    dest.copy(finalVector);
    // logger().log("dest.dot start Before deriv", dest.dot(m_start)); // this should not be 1

    vector<numType> hPsi;
    hPsi.copy(projVector);
    if (Energy)
        *Energy = hPsi.dot(finalVector);



    for (size_t i = m_fusedAnsatzes.size(); i > 0; i--)
    {
        size_t diff = m_commuteBoundaries[i] -m_commuteBoundaries[i-1];
        switch (m_fusedSizes[i-1])
        {
            EvolveDerDiag(1,uint8_t);
            EvolveDerDiag(2,uint8_t);
            EvolveDerDiag(3,uint8_t);
            EvolveDerDiag(4,uint8_t);
            EvolveDerDiag(5,uint8_t);
            EvolveDerDiag(6,uint8_t);
            EvolveDerDiag(7,uint8_t);
            EvolveDerDiag(8,uint16_t);
            EvolveDerDiag(9,uint16_t);
            EvolveDerDiag(10,uint16_t);
            EvolveDerDiag(11,uint16_t);
            EvolveDerDiag(12,uint16_t);
        case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;
            EvolveDer(1,uint8_t);
            EvolveDer(2,uint8_t);
            EvolveDer(3,uint8_t);
            EvolveDer(4,uint8_t);
            EvolveDer(5,uint8_t);
            EvolveDer(6,uint8_t);
            EvolveDer(7,uint8_t);
            EvolveDer(8,uint16_t);
            EvolveDer(9,uint16_t);
            EvolveDer(10,uint16_t);
            EvolveDer(11,uint16_t);
            EvolveDer(12,uint16_t);
        default:
            __builtin_trap();
        }
    }

    Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> gradVector_mu(&deriv[0],deriv.size(),1);
    Eigen::VectorXd derivCompressed = m_compressMatrix * gradVector_mu;
    deriv.copyFromBuffer(derivCompressed.data(),derivCompressed.rows());
    // logger().log("dest.dot start deriv", dest.dot(m_start)); // this should be 1
    // logger().log("hpsi.dot start deriv", hPsi.dot(m_start)); // this should be E
    auto endTime = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime-startTime).count();
    if constexpr (logTimings) logger().log("DerivProjTimeTaken (ms)", duration);
}

void FusedEvolve::evolveHessianProj(Eigen::MatrixXd &Hessian, vector<realNumType> &derivCompressed, const std::vector<realNumType> &angles, const vector<numType> &projVector,
                                    Eigen::Matrix<numType, -1, -1> *TsCopy, realNumType *Energy)
{
    //TODO
    // constexpr bool isComplex = !std::is_same_v<realNumType,numType>;
    if (!m_excsCached)
        regenCache();
    //There are two types of quantities we need
    // T means a tangent
    // <T|H|T> and <Psi|H|TT>
    //THT is calculated via back evolution of <T|H| and |Psi>
    // <Psi|H|TT> is calculated via back evolution of <Psi|H| and |T>
    //But this means |T> and therefore <T|H| are known at the end. So we can calculate <T|H|T> for `free'

    //A not memory efficient version.
    const vector<numType>& hPsi = projVector;
    // vector<numType> psi;
    Matrix<numType> Ts;
    // Matrix<numType> HTs;
    Ts.resize(angles.size()+1,m_start.size(),m_lieIsCompressed,m_compressor);
    //The last one is psi
    vectorView<Matrix<numType>,Eigen::RowMajor> psi = Ts.getJVectorView(angles.size());
    // HTs.resize(angles.size(),m_start.size(),m_lieIsCompressed,m_compressor);
    psi.copy(m_start);
    // Eigen::MatrixXd THT; // Proj does not have a THT term
    Hessian.resize(angles.size(),angles.size());
    Hessian.setZero();

    //Get the |T>s
    //We could save one evolution by getting psi aswell here

    std::vector<realNumType> permAngles(angles.size());
    std::vector<numType*> TPtrs(angles.size());
    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return angles[i];});
    std::transform(m_excPerm.begin(),m_excPerm.end(),TPtrs.begin(),[&Ts](size_t i){return &Ts.at(i,0);});

    //Do an evolution and gather all the Ts
    //The T ptrs are still at their respective evolutions. They need to be evolved to the end aswell


    auto time1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
    {
        size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
        switch (m_fusedSizes[i])
        {
            GenerateTangentsDiag(1,uint8_t)
        GenerateTangentsDiag(2,uint8_t)
            GenerateTangentsDiag(3,uint8_t)
            GenerateTangentsDiag(4,uint8_t)
            GenerateTangentsDiag(5,uint8_t)
            GenerateTangentsDiag(6,uint8_t)
            GenerateTangentsDiag(7,uint8_t)
            GenerateTangentsDiag(8,uint16_t)
            GenerateTangentsDiag(9,uint16_t)
            GenerateTangentsDiag(10,uint16_t)
            GenerateTangentsDiag(11,uint16_t)
            GenerateTangentsDiag(12,uint16_t)
            case 0:
            logger().log("Unhandled case 0");
            __builtin_trap();
            break;

            GenerateTangents(1,uint8_t)
                GenerateTangents(2,uint8_t)
                GenerateTangents(3,uint8_t)
                GenerateTangents(4,uint8_t)
                GenerateTangents(5,uint8_t)
                GenerateTangents(6,uint8_t)
                GenerateTangents(7,uint8_t)
                GenerateTangents(8,uint16_t)
                GenerateTangents(9,uint16_t)
                GenerateTangents(10,uint16_t)
                GenerateTangents(11,uint16_t)
                GenerateTangents(12,uint16_t)
                default:
                          __builtin_trap();
        }

    }
    std::vector<std::future<void>> futs;
    threadpool& pool = threadpool::getInstance(NUM_CORES);
    for (size_t ang = 0; ang < angles.size(); ang++)
    {
        futs.push_back(pool.queueWork([ang,this,&TPtrs,&permAngles]()
                                      {
                                          for (size_t i = 0; i < m_fusedAnsatzes.size(); i++)
                                          {
                                              size_t diff = m_commuteBoundaries[i+1] -m_commuteBoundaries[i];
                                              if (ang >= m_commuteBoundaries[i+1])
                                                  continue;
                                              switch (m_fusedSizes[i])
                                              {
                                              // EvolveTangentsDiag(1,uint8_t)
                                              case -1:
                                              {
                                                  constexpr uint8_t num = 1;
                                                  fusedDiagonalAnsatzX<num>* FA =  static_cast<fusedDiagonalAnsatzX<num>*>(m_fusedAnsatzes[i]);
                                                  if (ang < m_commuteBoundaries[i])
                                                  {
                                                      RunFuseNDiagonal<uint8_t,num,false,false>(FA->data(),(realNumType*)TPtrs[ang],permAngles.data() + m_commuteBoundaries[i],diff);
                                                  }
                                                  else
                                                  {
                                                      for (size_t currDiff = 0; currDiff < diff; currDiff += num)
                                                      {
                                                          numType** startTPtr = TPtrs.data() + m_commuteBoundaries[i] + currDiff;
                                                          for (numType** TPtr = startTPtr; TPtr < startTPtr + num; TPtr++)
                                                          {
                                                              if (*TPtr == TPtrs[ang])
                                                              {
                                                                  RunFuseNDiagonal<uint8_t,num,false,false>(FA->data() + currDiff/num,(realNumType*)*TPtr,permAngles.data() + m_commuteBoundaries[i] + currDiff,diff-currDiff);
                                                                  break;
                                                              }
                                                          }
                                                      }
                                                  }
                                                  break;
                                              }
                                              EvolveTangentsDiag(2,uint8_t)
                                                  EvolveTangentsDiag(3,uint8_t)
                                                  EvolveTangentsDiag(4,uint8_t)
                                                  EvolveTangentsDiag(5,uint8_t)
                                                  EvolveTangentsDiag(6,uint8_t)
                                                  EvolveTangentsDiag(7,uint8_t)
                                                  EvolveTangentsDiag(8,uint16_t)
                                                  EvolveTangentsDiag(9,uint16_t)
                                                  EvolveTangentsDiag(10,uint16_t)
                                                  EvolveTangentsDiag(11,uint16_t)
                                                  EvolveTangentsDiag(12,uint16_t)
                                                  case 0:
                                                  logger().log("Unhandled case 0");
                                                  __builtin_trap();
                                                  break;
                                                  EvolveTangents(1,uint8_t)
                                                      EvolveTangents(2,uint8_t)
                                                      EvolveTangents(3,uint8_t)
                                                      EvolveTangents(4,uint8_t)
                                                      EvolveTangents(5,uint8_t)
                                                      EvolveTangents(6,uint8_t)
                                                      EvolveTangents(7,uint8_t)
                                                      EvolveTangents(8,uint16_t)
                                                      EvolveTangents(9,uint16_t)
                                                      EvolveTangents(10,uint16_t)
                                                      EvolveTangents(11,uint16_t)
                                                      EvolveTangents(12,uint16_t)
                                                      default:
                                                                __builtin_trap();
                                              }

                                          }
                                      }
                                      ));
    }
    for (auto& f : futs)
        f.wait();
    futs.clear();
    auto time2 = std::chrono::high_resolution_clock::now();
    {
        Eigen::Map<const Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> TMap(&Ts.at(0,0),Ts.m_iSize,Ts.m_jSize);
        Eigen::Map<const Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> hPsiMap(&hPsi[0],1,hPsi.size());
        //For speed need to convert the ordering
        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> TMapC = TMap;
        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> T_C;
        T_C.noalias() = m_compressMatrixPsi * TMapC;

        //TODO bithacks
        //Need to lie to eigen since .topRows is slow so instead we compute the extra overlap. This generates <psi|hpsi>, along with <psi|T> i.e. energy and derivative
        Eigen::MatrixXd temp; // 1xN `matrix'
        temp.noalias() = (hPsiMap * T_C.adjoint()).real();

        derivCompressed.resize(temp.cols()-1,false,nullptr);

        for (long i = 0; i < temp.cols()-1; i++)
        {
            derivCompressed[i] = temp(0,i);
        }
        if (Energy)
            *Energy = temp(0,temp.cols()-1);
    }
    if (TsCopy)
    {
        Eigen::Map<const Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> TMapOnly(&Ts.at(0,0),angles.size(),Ts.m_jSize);
        *TsCopy = TMapOnly.transpose();
    }
    auto time5 = std::chrono::high_resolution_clock::now();
    //Remains to do the backwards evolution of |H|Psi> and |T>
    //This is basically the same as evolveDerivative

    std::transform(m_excPerm.begin(),m_excPerm.end(),permAngles.begin(),[&angles](size_t i){return -angles[i];});
    size_t stepSize = 1;//std::max(angles.size()/NUM_CORES,1ul);
    for (size_t angleIdxStart = 0; angleIdxStart < angles.size(); angleIdxStart+=stepSize)
    {
        size_t angleIdxEnd = std::min(angleIdxStart+stepSize,angles.size());
        futs.push_back(pool.queueWork([angleIdxStart,angleIdxEnd,&Hessian,&permAngles,&hPsi,this,&Ts]()
                                      {
                                          for (size_t angleIdx = angleIdxStart; angleIdx < angleIdxEnd; angleIdx++)
                                          {
                                              std::vector<realNumType> scratchPad;
                                              vector<numType> localHpsi;
                                              std::vector<realNumType*> derivLocs(permAngles.size());

                                              //Negative of the angles because we evolve backwards

                                              scratchPad.assign(permAngles.size(),0);

                                              localHpsi.copy(hPsi);


                                              std::transform(m_excPerm.begin(),m_excPerm.end(),derivLocs.begin(),[&scratchPad](size_t i){return &scratchPad[i];});

                                              //We only need to evaluate if m_excPerm(angleIdx) < m_commuteBoundaries[i];
                                              //To see this note that we could construct the hessian in permuted indexes
                                              //Then we only want to construct if permAngleIDX <= m_commuteBoundaries[i]
                                              //This will evaluate each element only once. problem is that diff may go over the boundary. So that needs to be fixed up afterwards.
                                              //Further knowing the symmetrical version is annoying so we symmetrise here
                                              size_t permAngleIdx = m_excInversePerm[angleIdx];
                                              for (size_t i = m_fusedAnsatzes.size(); i > 0; i--)
                                              {
                                                  //            start                     end
                                                  size_t diff = m_commuteBoundaries[i] -m_commuteBoundaries[i-1];
                                                  if (!(permAngleIdx <= m_commuteBoundaries[i]))
                                                      break;
                                                  switch (m_fusedSizes[i-1])
                                                  {
                                                      GenerateHPsiTDiagonal(1,uint8_t)
                                                  GenerateHPsiTDiagonal(2,uint8_t)
                                                      GenerateHPsiTDiagonal(3,uint8_t)
                                                      GenerateHPsiTDiagonal(4,uint8_t)
                                                      GenerateHPsiTDiagonal(5,uint8_t)
                                                      GenerateHPsiTDiagonal(6,uint8_t)
                                                      GenerateHPsiTDiagonal(7,uint8_t)
                                                      GenerateHPsiTDiagonal(8,uint16_t)
                                                      GenerateHPsiTDiagonal(9,uint16_t)
                                                      GenerateHPsiTDiagonal(10,uint16_t)
                                                      GenerateHPsiTDiagonal(11,uint16_t)
                                                      GenerateHPsiTDiagonal(12,uint16_t)
                                                      case 0:
                                                  {
                                                      logger().log("Unhandled case 0");
                                                      __builtin_trap();
                                                      break;
                                                  }
                                                      GenerateHPsiT(1,uint8_t)
                                                          GenerateHPsiT(2,uint8_t)
                                                          GenerateHPsiT(3,uint8_t)
                                                          GenerateHPsiT(4,uint8_t)
                                                          GenerateHPsiT(5,uint8_t)
                                                          GenerateHPsiT(6,uint8_t)
                                                          GenerateHPsiT(7,uint8_t)
                                                          GenerateHPsiT(8,uint16_t)
                                                          GenerateHPsiT(9,uint16_t)
                                                          GenerateHPsiT(10,uint16_t)
                                                          GenerateHPsiT(11,uint16_t)
                                                          GenerateHPsiT(12,uint16_t)


                                                          default:
                                                                    __builtin_trap();
                                                  }
                                                  //Store hessian
                                                  for (size_t n = 0; n < diff; n++)
                                                  {
                                                      if (permAngleIdx <= m_commuteBoundaries[i-1] + n)
                                                      {
                                                          Hessian(angleIdx,m_excPerm[m_commuteBoundaries[i-1] + n]) += *derivLocs[m_commuteBoundaries[i-1] + n];
                                                          if (permAngleIdx != m_commuteBoundaries[i-1] + n)
                                                              Hessian(m_excPerm[m_commuteBoundaries[i-1] + n],angleIdx) += *derivLocs[m_commuteBoundaries[i-1] + n];
                                                      }

                                                  }
                                              }
                                          }
                                      }
                                      ));
    }
    for (auto& f : futs)
        f.wait();

    Hessian = m_compressMatrix * Hessian * m_compressMatrix.transpose();
    auto time6 = std::chrono::high_resolution_clock::now();
    long duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
    long duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(time5-time2).count();
    long duration5 = std::chrono::duration_cast<std::chrono::milliseconds>(time6-time5).count();
    if constexpr (logTimings)
    {
        logger().log("FusedEvolve HessianProj Time taken 1 (ms)",duration1);
        logger().log("FusedEvolve HessianProj Time taken 4 (ms)",duration4);
        logger().log("FusedEvolve HessianProj Time taken 5 (ms)",duration5);
    }
}
