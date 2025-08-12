/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef OPERATORPOOL_H
#define OPERATORPOOL_H

#include <list>
#include <memory>
#include <vector>
#include <unordered_map>
#include "globals.h"
#include "linalg.h"
#include "sparsematrix.h"



constexpr char bitwiseDot(const uint32_t a, const uint32_t b, int dim)
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
public:
    numberOperatorCompressor(uint32_t numberOfParticles, uint32_t stateVectorSize)
    {
        assert(numberOfParticles <= 32);


        compressPerm.resize(stateVectorSize);
        uint32_t activeCount = 0;
        uint32_t allOnes = -1;
        for (uint32_t i = 0; i < stateVectorSize; i++)
        {
            bool indexActive = bitwiseDot(i,allOnes,32) == (char)numberOfParticles;
            if (indexActive)
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
    }
    virtual void dummyImplement(){}
};

class stateRotate : public operatorPool
{
public:
    struct exc
    {
        int8_t first;
        int8_t second;
        int8_t third;
        int8_t fourth;
        exc(){}
        exc(int8_t (&arr)[4]){first = arr[0]; second = arr[1]; third = arr[2]; fourth = arr[3];}
        exc(uint32_t val){first = val >> 24; second = val >> 16; third = val >> 8; fourth = val;}
        operator uint32_t() const {return ((uint8_t)first << 24) + ((uint8_t)second << 16) + ((uint8_t) third << 8) + (uint8_t)fourth;}
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
        int8_t& operator [](size_t idx)
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
            if (third == -1 && fourth == -1 && other.third == -1 && other.fourth == -1)
            {

                for (int8_t i = 0; i < 2 && allNonEqual; i++)
                {
                    for (int8_t j = 0; j < 2 && allNonEqual; j++)
                    {
                        if ((*this)[i] == other[j])
                            allNonEqual = false;
                    }
                }
            }
            else
            {
                for (int8_t i = 0; i < 2 && allNonEqual; i++)
                {
                    for (int8_t j = 0; j < 2 && allNonEqual; j++)
                    {
                        if ((*this)[i] == other[j])
                            allNonEqual = false;
                    }
                }
            }
            return allNonEqual;
        }
    }; // 0 indexed
private:
    typedef uint32_t excHash;

    excHash makeExcHash(const exc e){return e;}
    void makeExcFromHash(exc &e,const excHash eh){ e = exc(eh);}

    std::unordered_map<size_t,matrixType> m_lieAlgebra;
    bool m_lieAlgebraMatricesGenerated = false;
    std::shared_ptr<numberOperatorCompressor> m_compressor;
    bool m_compressStateVectors = false;

public:
    std::unordered_map<size_t,excHash> m_lieAlgebraIndexExcMap;
    std::unordered_map<excHash,size_t> m_lieAlgebraIndexExcReverseMap;

    int m_nQubits;
    int m_dim = -1;//number of elements in the pauli vectors
    int m_lieOpDim = 0;

    stateRotate(int nQubits, bool compressStateVectors = false, int numberOfParticles = 0);
    virtual ~stateRotate(){};

    matrixType* getLieAlgebraMatrix(const exc a);
    matrixType* getLieAlgebraMatrix(size_t idx) override;
    matrixType* getLieAlgebraMatrix(void* data) override {return getLieAlgebraMatrix(*(exc*)data);}
    size_t convertDataToIdx(const void* data) override{return m_lieAlgebraIndexExcReverseMap.at(makeExcHash(*(exc*)data));}
    void convertIdxToExc(size_t idx, exc& e){return makeExcFromHash(e,m_lieAlgebraIndexExcMap.at(idx));}
    const std::unordered_map<size_t,matrixType>* getLieAlgebraMatrices() override;

    bool loadOperators(std::string filename);

    bool getCompressor(std::shared_ptr<compressor>& ptr) override {ptr = m_compressor; return m_compressStateVectors;}
};

#endif // OPERATORPOOL_H
