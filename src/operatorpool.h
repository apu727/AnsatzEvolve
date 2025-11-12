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
    uint64_t i = a & b & (dim < 64 ? ((1<<dim) -1) : -1);
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
};

class SZAndnumberOperatorCompressor : public compressor
{
    uint64_t m_numberOfQubits;
    uint64_t m_qubitBitMask;
    uint64_t m_spinUpBitMask;
    uint64_t m_spinDownBitMask;
    int m_spinUp;
    int m_spinDown;
    size_t m_decompressedSize = 0;
public:
    bool compressIndex(uint64_t index, uint64_t& compressedIdx) override
    {
        bool spinUpActive = popcount(index & m_spinUpBitMask) == (char)m_spinUp;
        bool spinDownActive = popcount(index & m_spinDownBitMask) == (char)m_spinDown;
        if (spinUpActive && spinDownActive)
        {
            auto elem = std::lower_bound(decompressPerm.begin(),decompressPerm.end(),index); // binary search.
            if (elem == decompressPerm.end())
            {
                fprintf(stderr,"compressed index not found. SZ\n");
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
    SZAndnumberOperatorCompressor(uint64_t stateVectorSize, int spinUp, int spinDown);
    size_t getUnCompressedSize() override { return m_decompressedSize; }
    virtual void dummyImplement() override{}
    virtual bool opDoesSomething(excOp&) override;
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
        exc(uint64_t val){first = val >> 24; second = val >> 16; third = val >> 8; fourth = val;}
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
            uint64_t first = 0;
            uint64_t second = 0;
            for (int8_t i = 0; i < 4; i++)
            {
                if ((*this)[i] != -1)
                    first = first | 1<<(*this)[i];
                if (other[i] != -1)
                    second = second | 1<<other[i];
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
            if (third == -1)
            {
                assert(fourth == -1);
                return first == second;
            }
            uint64_t create = (1<<first) | (1<<second);
            uint64_t annihilate = (1<<third) | (1<<fourth);
            assert(third != fourth);//Double destroy
            assert(first != second); //Double create
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
