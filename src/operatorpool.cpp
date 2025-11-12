/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "operatorpool.h"
#include "globals.h"
#include "logger.h"

#include <cassert>

stateRotate::stateRotate(int nQubits, std::shared_ptr<compressor> comp)
{
    m_nQubits = nQubits;
    m_dim = 1ul<<m_nQubits;
    if (comp)
    {
        m_compressStateVectors = true;
        m_compressor = comp;
    }

}

matrixType *stateRotate::getLieAlgebraMatrix(const exc a)
        //format: i j k l -> a^+_i a^+_j a_k a_l  #Note the swapping phase is not kept track of
        // or     i j -> a^+_i a_j
{

    if (m_lieAlgebraMatricesGenerated)
        return &m_lieAlgebra.at(m_lieAlgebraIndexExcReverseMap.at(makeExcHash(a)));
    if (m_lieAlgebraIndexExcReverseMap.find(makeExcHash(a)) != m_lieAlgebraIndexExcReverseMap.end())
        return &m_lieAlgebra.at(m_lieAlgebraIndexExcReverseMap.at(makeExcHash(a)));

    uint64_t* intois = new uint64_t[m_dim];
    uint64_t* intojs = new uint64_t[m_dim];
    numType* vals = new numType[m_dim];



    uint64_t activeBits =  0;
    uint64_t createBits = 0;
    uint64_t annihilateBits = 0;
    uint64_t signMask = 0;
    uint64_t permPhase = 1;


    if (a[2] > -1 && a[3] > -1)
    {
        if (a[0] == a[1] || a[2] == a[3])
            fprintf(stderr,"Wrong order in creation annihilation operators");
        createBits = (1ul<<a[0]) | (1ul<<a[1]);
        annihilateBits = (1ul<<a[2]) | (1ul<<a[3]);
        signMask = ((1ul<<a[0])-1) ^ ((1ul<<a[1])-1) ^((1ul<<a[2])-1) ^((1ul<<a[3])-1);
        signMask = signMask & ~((1ul<<a[0]) | (1ul<<a[1]) | (1ul<<a[2]) | (1ul<<a[3]));
        activeBits = createBits | annihilateBits;
        if (a[0] > a[1]) // see applyExcToBasisState_ in fusedevolve.cpp
            permPhase *= -1;
        if (a[2] > a[3])
            permPhase *= -1;
    }
    else
    {
        createBits = (1ul<<a[0]);
        annihilateBits = (1ul<<a[1]);
        activeBits = createBits | annihilateBits;

        signMask = ((1ul<<a[0])-1) ^ ((1ul<<a[1])-1);
        signMask = signMask & ~((1ul<<a[0]) | (1ul<<a[1]));
    }

    uint64_t* intois_pos = intois;
    uint64_t* intojs_pos = intojs;
    numType* vals_pos = vals;
    uint64_t start = 0;
    uint64_t end = m_dim;


    for (;start != end; start++)
    {
        uint64_t basisState = start;
        uint64_t resultState = basisState;


        numType phase = permPhase;
        phase *= (popcount(basisState & signMask) & 1) ? -1 : 1;
        uint64_t maskedBasisState = basisState & activeBits;

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
                phase *= 0;
            }
        }
        if (phase != numType(0))
        {
            *intois_pos++ = basisState;
            *intojs_pos++ = resultState;
            *vals_pos++ = phase;
        }

    }
    matrixType* ret = nullptr;

    if (intois_pos == intois)
    {
        fprintf(stderr,"element doesnt do anything: %d,%d,%d,%d\n", a[0],a[1],a[2],a[3]);
    }
    else
    {
        //matrixType mat(vals,intois,intojs,intois_pos-intois); //C_ji v_i multiplication expected
        fprintf(stderr,"%zu is `%d,%d,%d,%d' 's effect\n", m_lieAlgebra.size(),a[0],a[1],a[2],a[3]);

        //assert((mat.T() + mat).allClose(0));
        m_lieAlgebraIndexExcMap.emplace(m_lieAlgebra.size(),makeExcHash(a));
        m_lieAlgebraIndexExcReverseMap.emplace(makeExcHash(a),m_lieAlgebra.size());
        if (m_compressStateVectors)
            m_lieAlgebra.insert({m_lieAlgebra.size(),matrixType(vals,intois,intojs,intois_pos-intois,m_compressor,true)});
        else
            m_lieAlgebra.insert({m_lieAlgebra.size(),matrixType(vals,intois,intojs,intois_pos-intois)});

        m_lieOpDim = m_lieAlgebra.size();
        ret = &m_lieAlgebra.at(m_lieAlgebra.size()-1);
    }
    delete[] intois;
    delete[] intojs;
    delete[] vals;
    return ret;
}

matrixType *stateRotate::getLieAlgebraMatrix(size_t idx)
{
    assert(idx < m_lieAlgebra.size());
    return &m_lieAlgebra.at(idx);
}

const std::unordered_map<size_t, matrixType> *stateRotate::getLieAlgebraMatrices()
{
    fprintf(stderr, "Asked for all lie Algebra matrices but idk how to generate them, returning the ones already found\n");
    return &m_lieAlgebra;
}

bool stateRotate::loadOperators(std::string filePath)
{
    std::vector<exc> excs;

    if (!loadOperators(filePath,excs))
        return 0;
    for (auto& e : excs)
        getLieAlgebraMatrix(e);
    return 1;
}

bool stateRotate::loadOperators(std::string filePath, std::vector<stateRotate::exc>& excs)
{
    excs.clear();
    FILE *fp;
    exc Excs;

    fp = fopen(filePath.c_str(), "r");
    if(NULL == fp)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }
    int ret = fscanf(fp, "%hhd %hhd %hhd %hhd \n",&Excs[0],&Excs[1],&Excs[2],&Excs[3] );
    while(EOF != ret)
    {
        //fprintf(stderr,"Read Operator: %hhd %hhd %hhd %hhd\n ", Excs[0],Excs[1],Excs[2],Excs[3]);
        for (int i = 0; i < 4; i++)
            Excs[i] -= 1;
        excs.push_back(Excs);
        ret = fscanf(fp, "%hhd %hhd %hhd %hhd \n",&Excs[0],&Excs[1],&Excs[2],&Excs[3] );
    }
    fclose(fp);
    return 1;
}

SZAndnumberOperatorCompressor::SZAndnumberOperatorCompressor(uint64_t stateVectorSize, int spinUp, int spinDown)
{//Compresses for a specific spin, It assumes that the top N/2 bits are spin up and vice versa
    uint64_t numberOfQubits = 0;
    {
        uint64_t dummy = stateVectorSize-1;
        while(dummy)
        {
            numberOfQubits++;
            dummy = dummy >>1;
        }
    }
    assert(numberOfQubits %2 == 0);
    assert(numberOfQubits < 64);
    m_numberOfQubits = numberOfQubits;
    m_spinDownBitMask = (1ul<<(numberOfQubits/2))-1;
    m_spinUpBitMask = ((1ul<<(numberOfQubits))-1) ^ m_spinDownBitMask;
    m_decompressedSize = stateVectorSize;
    m_spinUp = spinUp;
    m_spinDown = spinDown;

    // compressPerm.resize(stateVectorSize);
    // uint64_t activeCount = 0;
    for (uint64_t i = 0; i < stateVectorSize; i++)
    {
        bool spinUpActive = popcount(i & m_spinUpBitMask) == (char)m_spinUp;
        bool spinDownActive = popcount(i & m_spinDownBitMask) == (char)m_spinDown;
        if (spinUpActive && spinDownActive)
        {
            // compressPerm[i] = activeCount;
            decompressPerm.push_back(i);
            // activeCount++;
        }
        else
        {
            // compressPerm[i] = -1;
        }
    }
}

bool SZAndnumberOperatorCompressor::opDoesSomething(excOp &e)
{
    int spinUpCreate = popcount(e.create & m_spinUpBitMask);
    int spinDownCreate = popcount(e.create & m_spinDownBitMask);

    int spinUpDestroy = popcount(e.destroy & m_spinUpBitMask);
    int spinDownDestroy = popcount(e.destroy & m_spinDownBitMask);
    return (spinUpCreate == spinUpDestroy) && (spinDownCreate == spinDownDestroy);

}
