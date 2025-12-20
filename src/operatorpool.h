/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef OPERATORPOOL_H
#define OPERATORPOOL_H

#include <memory>
#include <vector>
#include <unordered_map>
#include "globals.h"
#include "linalg.h"
#include "sparsematrix.h"



constexpr char bitwiseDot(const uint64_t a, const uint64_t b, int dim)
{
    // uint64_t prod = a&b;
    // char ret = 0; //at most 32
    // for (int i = 0; i < dim; i++)
    // {
    //     ret += (prod & 1);
    //     prod>>=1;
    // }
    // return ret;
    uint64_t i = a & b & (dim < 64 ? ((1ul<<dim) -1) : -1);
    return popcount(i);
}

class operatorPool
{
public:
    virtual matrixType* getLieAlgebraMatrix(void* data) = 0;
    virtual matrixType* getLieAlgebraMatrix(size_t idx) = 0;
    virtual size_t convertDataToIdx(const void* data) = 0;
    virtual const std::unordered_map<size_t,matrixType>* getLieAlgebraMatrices() = 0;
    virtual bool getCompressor(std::shared_ptr<compressor>& ptr) {ptr = nullptr; return false;}
};


class numberOperatorCompressor : public compressor
{
    static const uint64_t allOnes = -1;
    int m_numberOfParticles;
    size_t m_decompressedSize;
public:
    bool compressIndex(uint64_t index, uint64_t& compressedIdx) override
    {
        if (popcount(index) == m_numberOfParticles)
        {
            //The binary search can be avoided same as in SZAndnumberOperatorCompressor using Colexical ordering.
            auto elem = std::lower_bound(decompressPerm.begin(),decompressPerm.end(),index); // binary search.
            if (elem == decompressPerm.end())
            {
                fprintf(stderr,"compressed index not found. numberOperator\n");
                __builtin_trap();
            }
            compressedIdx = elem - decompressPerm.begin();
            return true;
        }
        else
        {
            compressedIdx = -1;
            return false;
        }
    }
    numberOperatorCompressor(int numberOfParticles, uint64_t stateVectorSize)
    {
        assert(numberOfParticles <= 64);
        m_numberOfParticles = numberOfParticles;
        m_decompressedSize = stateVectorSize;
        // compressPerm.resize(stateVectorSize);
        // uint64_t activeCount = 0;
        for (uint64_t i = 0; i < stateVectorSize; i++)
        {
            bool indexActive = bitwiseDot(i,allOnes,64) == (char)numberOfParticles;
            if (indexActive)
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
    size_t getUnCompressedSize() override { return m_decompressedSize; }
    virtual void dummyImplement() override{}
    virtual bool opDoesSomething(excOp&) override{return true;} // Always true for now. All operators are particle conserving
    registeredBaseClasses getSerialisationType() override {return registeredBaseClasses::numberOperatorCompressor;};
    virtual serialDataContainer serialise() override; // derived must serialise this object first, the returned bytes must be stored first in the serial stream
    static std::shared_ptr<compressor> deserialise(char *ptr);
};

class SZAndnumberOperatorCompressor : public compressor
{
    uint64_t m_numberOfQubits;
    uint64_t m_qubitBitMask;
    uint64_t m_spinUpBitMask;
    uint64_t m_spinDownBitMask;
    uint64_t m_spinUpSize = 0;
    uint64_t m_spinDownSize = 0;
    int m_spinUp;
    int m_spinDown;
    size_t m_decompressedSize = 0; // If necessary this can also me compressed in the same way as the m_compressedSpinUpLookup.
    static constexpr uint8_t chooseCacheSize = 33; // because of the + 1 later.
    size_t m_chooseLookup[chooseCacheSize][chooseCacheSize]; // 8712 bytes. Well within L1 cache for most processors. Could be constexpr but why.
    std::vector<uint32_t> m_compressedSpinUpLookup;
    std::vector<uint32_t> m_compressedSpinDownLookup;

    static size_t choose(size_t n, size_t k)
    {
        if (n < k)
            return 0;
        if (n == k)
            return 1;
        if (k == 0)
            return 1;
        return (n * choose(n - 1, k - 1)) / k;
    }
    void setupChooseCache();
    /* Colexicographic ordering. Kinda magic. As an example:
     *  01 = 00011 = 0C1 + 1C2 = 0+0 = 0
     *  02 = 00101 = 0C1 + 2C2 = 0+1 = 1
     *  12 = 00110 = 1C1 + 2C2 = 1+1 = 2
     *  03 = 01001 = 0C1 + 3C2 = 0+3 = 3
     *  13 = 01010 = 1C1 + 3C2 = 1+3 = 4
     *  23 = 01100 = 2C1 + 3C2 = 2+3 = 5
     *  04 = 10001 = 0C1 + 4C2 = 0+6 = 6
     *  14 = 10010 = 1C1 + 4C2 = 1+6 = 7
     *  24 = 10100 = 2C1 + 4C2 = 2+6 = 8
     *  34 = 11000 = 3C1 + 4C2 = 3+6 = 9
     *
     *  This predicts orders the N choose k spins in each spin block.
     *  One could use this directly for compression/decompression but instead we cache it in m_compressedSpinUpLookup and m_compressedSpinDownLookup
     *  for 64 bit statevector this is a max storage of 70GB. A doable number. ~ 3x Speed boost, very similar to the previous single lookup version.
     */
    uint32_t ColexicoOrder(uint32_t index, uint32_t k)
    {
        uint64_t spinDownRank = 0;
        uint32_t spinDownNumberActiveSoFar = 0;

        //spinDown
        for (uint32_t qubit = 0; qubit < m_numberOfQubits/2 && spinDownNumberActiveSoFar < k; qubit++)
        {
            if (index & (1<<qubit))
            {
                spinDownRank += m_chooseLookup[qubit][spinDownNumberActiveSoFar+1];
                spinDownNumberActiveSoFar++;
            }
        }
        return spinDownRank;
    }
    struct serialData
    {
        uint64_t stateVectorSize;
        int spinUp;
        int spinDown;
    };
    friend compressor;
    static size_t deserialise(char *ptr,std::shared_ptr<compressor>& dest);
public:


    bool compressIndex(uint64_t index, uint64_t& compressedIdx) override
    {
        uint32_t spinUpBlock = index >> m_numberOfQubits/2;
        uint32_t spinDownBlock = index & m_spinDownBitMask;
        bool spinUpActive = popcount(spinUpBlock) == (char)m_spinUp;
        bool spinDownActive = popcount(spinDownBlock) == (char)m_spinDown;
        if (spinUpActive && spinDownActive)
        {
            uint64_t spinUpIndex = m_compressedSpinUpLookup[spinUpBlock];
            assert((uint32_t)spinUpIndex != (uint32_t)-1);
            uint64_t spinDownIndex = m_compressedSpinDownLookup[spinDownBlock];
            assert((uint32_t)spinDownIndex != (uint32_t)-1);
            compressedIdx = spinUpIndex*m_spinDownSize + spinDownIndex;

            return true;
        }
        else
        {
            compressedIdx = -1;
            return false;
        }
    }

    bool compressIndex(__m512i index, __m512i& compressedIdx, __mmask8& valid)
    {
        __m512i spinUpBlock = _mm512_srl_epi64(index,_mm_set1_epi64x(m_numberOfQubits/2));
        __m512i spinDownBlock = _mm512_and_epi64(index,_mm512_set1_epi64(m_spinDownBitMask));
        __mmask8 spinUpActive = _mm512_cmpeq_epi64_mask(_mm512_popcnt_epi64(spinUpBlock), _mm512_set1_epi64(m_spinUp));
        __mmask8 spinDownActive = _mm512_cmpeq_epi64_mask(_mm512_popcnt_epi64(spinDownBlock), _mm512_set1_epi64(m_spinDown));
        valid = spinUpActive & spinDownActive;
        __m512i neg1 = _mm512_set1_epi64(-1);
        if (valid)
        {
            __m256i spinUpIndex256 = _mm512_mask_i64gather_epi32(_mm512_castsi512_si256(neg1),valid,spinUpBlock,m_compressedSpinUpLookup.data(),sizeof(uint32_t));//m_compressedSpinUpLookup[spinUpBlock];
            __m512i spinUpIndex = _mm512_cvtepu32_epi64(spinUpIndex256);
            __m256i spinDownIndex256 = _mm512_mask_i64gather_epi32(_mm512_castsi512_si256(neg1),valid,spinDownBlock,m_compressedSpinDownLookup.data(),sizeof(uint32_t));//m_compressedSpinDownLookup[spinDownBlock];
            __m512i spinDownIndex = _mm512_cvtepu32_epi64(spinDownIndex256);

            // compressedIdx = spinUpIndex*m_spinDownSize + spinDownIndex;
            compressedIdx = _mm512_mask_add_epi64(neg1,valid,_mm512_mullo_epi64(spinUpIndex,_mm512_set1_epi64(m_spinDownSize)),spinDownIndex);

            return true;
        }
        else
        {
            compressedIdx = neg1;
            return false;
        }
    }
    SZAndnumberOperatorCompressor(uint64_t stateVectorSize, int spinUp, int spinDown);
    size_t getUnCompressedSize() override { return m_decompressedSize; }
    virtual void dummyImplement() override{}
    virtual bool opDoesSomething(excOp&) override;




    virtual registeredBaseClasses getSerialisationType() override {return registeredBaseClasses::SZAndnumberOperatorCompressor;};
    virtual serialDataContainer serialise() override;


};

class stateRotate : public operatorPool
{
public:
    class exc
    {
        int8_t first;
        int8_t second;
        int8_t third;
        int8_t fourth;
        void setup(); // sets up create,annihilate,signmask,sign
    public:

        bool sign = false; // true = -1, false = 1. This is the builtin sign due to orderings
        uint64_t create;
        uint64_t annihilate;
        uint64_t signMask;

        exc(){}
        exc(int8_t (&arr)[4]){first = arr[0]; second = arr[1]; third = arr[2]; fourth = arr[3]; setup();}
        exc(uint64_t val){first = val >> 24; second = val >> 16; third = val >> 8; fourth = val; setup();}
        operator uint64_t() const {return ((uint8_t)first << 24) + ((uint8_t)second << 16) + ((uint8_t) third << 8) + (uint8_t)fourth;}
        bool isSingleExc() const {return first >= 0 && second >= 0 && third < 0 && fourth < 0;}
        bool isDoubleExc() const {return first >= 0 && second >= 0 && third >= 0 && fourth >= 0;}
        const int8_t& operator [](size_t idx) const
        {
            switch (idx)
            {
            case 0:
                return first;
            case 1:
                return second;
            case 2:
                return third;
            case 3:
                return fourth;
            default:
                __builtin_trap();
                return first;
            }
        }
        bool commutes(const exc& other)
        {
            bool allNonEqual = true;
            uint64_t first = 0;
            uint64_t second = 0;
            for (int8_t i = 0; i < 4; i++)
            {
                if ((*this)[i] != -1)
                    first = first | 1ul<<(*this)[i];
                if (other[i] != -1)
                    second = second | 1ul<<other[i];
            }
            allNonEqual = !(first & second);

            return allNonEqual || (isDiagonal() && other.isDiagonal());
        }
        bool hasDiagonal() const
        {
            bool hasDiagonal = false;
            if (third == -1 && fourth == -1)
            {

                for (int8_t i = 0; i < 2 && !hasDiagonal; i++)
                {
                    for (int8_t j = i+1; j < 2 && !hasDiagonal; j++)
                    {
                        if ((*this)[i] == (*this)[j])
                            hasDiagonal = true;
                    }
                }
            }
            else
            {
                for (int8_t i = 0; i < 4 && !hasDiagonal; i++)
                {
                    for (int8_t j = i+1; j < 4 && !hasDiagonal; j++)
                    {
                        if ((*this)[i] == (*this)[j])
                            hasDiagonal = true;
                    }
                }
            }
            return hasDiagonal;
        }
        bool isDiagonal() const
        {
            //dont care about order
            return (create ^ annihilate) == 0;
        }
    }; // 0 indexed
private:
    typedef uint64_t excHash;

    excHash makeExcHash(const exc e){return e;}
    void makeExcFromHash(exc &e,const excHash eh){ e = exc(eh);}

    std::unordered_map<size_t,matrixType> m_lieAlgebra;
    bool m_lieAlgebraMatricesGenerated = false;
    std::shared_ptr<compressor> m_compressor;
    bool m_compressStateVectors = false;

public:
    std::unordered_map<size_t,excHash> m_lieAlgebraIndexExcMap;
    std::unordered_map<excHash,size_t> m_lieAlgebraIndexExcReverseMap;

    int m_nQubits;
    int m_dim = -1;//number of elements in the pauli vectors
    int m_lieOpDim = 0;

    stateRotate(int nQubits, std::shared_ptr<compressor> comp = nullptr);
    virtual ~stateRotate(){};

    matrixType* getLieAlgebraMatrix(const exc a);
    matrixType* getLieAlgebraMatrix(size_t idx) override;
    matrixType* getLieAlgebraMatrix(void* data) override {return getLieAlgebraMatrix(*(exc*)data);}
    size_t convertDataToIdx(const void* data) override{return m_lieAlgebraIndexExcReverseMap.at(makeExcHash(*(exc*)data));}
    void convertIdxToExc(size_t idx, exc& e){return makeExcFromHash(e,m_lieAlgebraIndexExcMap.at(idx));}
    const std::unordered_map<size_t,matrixType>* getLieAlgebraMatrices() override;

    bool loadOperators(std::string filename);
    static bool loadOperators(std::string filePath, std::vector<stateRotate::exc>& excs);

    bool getCompressor(std::shared_ptr<compressor>& ptr) override {ptr = m_compressor; return m_compressStateVectors;}
};

#endif // OPERATORPOOL_H
