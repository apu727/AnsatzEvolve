/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "sparsematrix.h"
#include "logger.h"
#include "myComplex.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>

//bool compare1(const std::pair<size_t,size_t> &a, const std::pair<size_t,size_t> &b ) {return a.first < b.first;};
//bool compare2(const std::pair<size_t,size_t> &a, const std::pair<size_t,size_t> &b ) {return a.second < b.second;};



template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::sparseMatrix()
{
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::sparseMatrix(const std::vector<dataType> &value, const std::vector<int> &iIndexes, const std::vector<int> &jIndexes, int)
{
    assert(value.size() == iIndexes.size());
    assert(value.size() == jIndexes.size());

    size_t N = jIndexes.size();

    m_iIndexes.reserve(N);
    m_iIndexes.insert(m_iIndexes.end(),iIndexes.begin(),iIndexes.end());

    m_jIndexes.reserve(N);
    m_jIndexes.insert(m_jIndexes.end(),jIndexes.begin(),jIndexes.end());
    m_data = value;
    unblankAll();
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType, vectorType>::sparseMatrix(dataType* value, uint32_t* iIndex, uint32_t* jIndex,size_t N, std::shared_ptr<compressor> compressor, bool isRotationGenerator)
    : sparseMatrix<dataType,vectorType>(value,iIndex,jIndex,N)
{
    m_compressor = compressor;
    m_isCompressed = true;

    m_isRotationGenerator = isRotationGenerator;

    std::vector<uint32_t> iIndexes = std::move(m_iIndexes);
    std::vector<uint32_t> jIndexes = std::move(m_jIndexes);
    std::vector<dataType> data = std::move(m_data);

    m_iIndexes.clear();
    m_jIndexes.clear();
    m_data.clear();

    for (size_t i = 0; i < N; i++)
    {
        uint32_t compressediIndex;
        uint32_t compressedjIndex;
        if (m_compressor->compressIndex(iIndexes[i],compressediIndex) &&
            m_compressor->compressIndex(jIndexes[i],compressedjIndex))
        {
            if (std::real(data[i]) > 0 || compressediIndex == compressedjIndex /*ComplexRotation*/|| !m_isRotationGenerator)
            {
                if (std::imag(data[i])<0)
                    logger().log("Error");
                m_iIndexes.push_back(compressediIndex);
                m_jIndexes.push_back(compressedjIndex);
                m_data.push_back(data[i]);
            }
        }
    }

    unblankAll();
}
template <typename T>
std::vector<std::size_t> sortPermutation(T* & vec, size_t N)
{
    std::vector<std::size_t> p(N);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](std::size_t i, std::size_t j){ return vec[i] < vec[j]; });
    return p;
}
template <typename T>
void applyPermutationAndStore(const T* vec, const std::vector<std::size_t>& perm, std::vector<T>& dest, size_t N)
{
    dest.resize(N);
    std::transform(perm.begin(), perm.end(), dest.begin(),
                   [&](std::size_t i){ return vec[i]; });
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::sparseMatrix(dataType *value, uint32_t *iIndexes, uint32_t *jIndexes, size_t N)
{
    auto sortPerm = sortPermutation(jIndexes,N);
    applyPermutationAndStore(iIndexes,sortPerm,m_iIndexes,N);
    applyPermutationAndStore(jIndexes,sortPerm,m_jIndexes,N);
    applyPermutationAndStore(value,sortPerm,m_data,N);
    unblankAll();
//    m_iIndexes.reserve(N);
//    m_iIndexes.insert(m_iIndexes.end(),iIndexes,iIndexes+N);
//    m_jIndexes.reserve(N);
//    m_jIndexes.insert(m_jIndexes.end(),jIndexes,jIndexes+N);
//    m_data.reserve(N);
    //    m_data.insert(m_data.end(),value,value+N);
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::sparseMatrix(const Matrix<dataType> &other)
{
    for (size_t i = 0; i < other.m_iSize; i++)
    {
        for (size_t j = 0; j < other.m_jSize; j++)
        {
            const dataType& elem = other.at(i,j);
            if (elem != static_cast<dataType>(0))
            {
                m_iIndexes.push_back(i);
                m_jIndexes.push_back(j);
                m_data.push_back(elem);
            }
        }
    }
    unblankAll();
}

//template<typename dataType, typename vectorType>
//sparseMatrix<dataType,vectorType>::sparseMatrix(const sparseMatrix &other)
//{

//    m_data = other.m_data;
//    m_iIndexes = other.m_iIndexes;
//    m_jIndexes = other.m_jIndexes;
//}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::sparseMatrix(sparseMatrix &&other)
{
    m_data = std::move(other.m_data);
    m_iIndexes = std::move(other.m_iIndexes);
    m_jIndexes = std::move(other.m_jIndexes);
    this->m_blankCount = other.m_blankCount;
    m_isCompressed = other.m_isCompressed;
    m_compressor = std::move(other.m_compressor);
    m_isRotationGenerator = other.m_isRotationGenerator;
    m_iSize = other.m_iSize;
    m_jSize = other.m_jSize;
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType> &sparseMatrix<dataType,vectorType>::copy(const sparseMatrix &other)
{
    m_data = other.m_data;
    m_iIndexes = other.m_iIndexes;
    m_jIndexes = other.m_jIndexes;
    this->m_blankCount = other.m_blankCount;
    m_isCompressed = other.m_isCompressed;
    m_compressor = other.m_compressor;
    m_isRotationGenerator = other.m_isRotationGenerator;
    m_iSize = other.m_iSize;
    m_jSize = other.m_jSize;
    return *this;
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType> &sparseMatrix<dataType,vectorType>::operator*=(dataType scalar)
{
    auto rowData = m_data.begin();

    for (; rowData != m_data.end();)
    {
        (*rowData) *= scalar;
        ++rowData;
    }
    return *this;
}

template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType> &sparseMatrix<dataType,vectorType>::operator=(sparseMatrix&& other)
{
    m_data = std::move(other.m_data);
    m_iIndexes = std::move(other.m_iIndexes);
    m_jIndexes = std::move(other.m_jIndexes);
    m_isCompressed = other.m_isCompressed;
    m_compressor = std::move(other.m_compressor);
    m_isRotationGenerator = other.m_isRotationGenerator;
    m_iSize = other.m_iSize;
    m_jSize = other.m_jSize;
    return *this;
}

template<typename dataType, typename vectorType>
bool sparseMatrix<dataType,vectorType>::allClose(dataType val, realNumType atol)
{
    for (const auto& r: m_data)
        if (std::abs(r-val) > atol)
            return false;
    return true;
}

template<typename dataType, typename vectorType>
const dataType* sparseMatrix<dataType,vectorType>::at(uint32_t i, uint32_t j, bool& success) const
{
    auto d = m_data.begin();
    auto iIdx = m_iIndexes.begin();
    auto jIdx = m_jIndexes.begin();
    success = true;
    while(iIdx < m_iIndexes.end())
    {
        if (*iIdx == i && *jIdx == j)
            return &*d;
        *iIdx++;
        *jIdx++;
        *d++;
    }
    success = false;
    return &def;

}

template<typename dataType, typename vectorType>
bool sparseMatrix<dataType,vectorType>::addBlank()
{
    size_t& blankCount = this->m_blankCount;


    blankCount++;
    if (blankCount > m_iSize)
    {
        blankCount = m_iSize;
        return false;
    }
    return true;
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType,vectorType>::unblankAll()
{
    if (m_isCompressed)
    {
        m_iSize = m_compressor->getCompressedSize();
        m_jSize = m_compressor->getCompressedSize();
    }
    else
    {
        m_iSize = *std::max_element(iItBegin(),iItEnd())+1;
        m_jSize = *std::max_element(jItBegin(),jItEnd())+1;
    }
    if (!m_isRotationGenerator)
        assert(m_iSize == m_jSize);
    this->m_blankCount = m_iSize;
}


template<typename dataType, typename vectorType>
bool sparseMatrix<dataType,vectorType>::loadMatrix(std::string filePath)
{
    return s_loadMatrix(this,filePath);
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::multiply(const vector<vectorType> &other, vector<vectorType> &dest) const
{
    std::shared_ptr<compressor> otherCompressor;
    bool otherIsCompressed = other.getIsCompressed(otherCompressor);
    if (otherIsCompressed != m_isCompressed)
    {
        logger().log("Multiplication between compressed and uncompressed is inefficient, please fix");
        if (otherIsCompressed && !m_isCompressed)
        {
            vector<vectorType> decompressedOther;
            vector<vectorType> decompressedDest;
            compressor::deCompressVector(other,decompressedOther,otherCompressor);
            multiplyDecompressed(decompressedOther,decompressedDest);
            compressor::compressVector(decompressedDest,dest,otherCompressor);
        }
        else if (!otherIsCompressed && m_isCompressed)
        {
            vector<vectorType> compressedOther;
            vector<vectorType> compressedDest;
            compressor::compressVector(other,compressedOther,m_compressor);
            multiplyDecompressed(compressedOther,compressedDest);
            compressor::deCompressVector(compressedDest,dest,m_compressor);
        }
        else
        {
            logger().log("Not possible");
            __builtin_trap();
        }
    }
    else if (otherCompressor.get() != m_compressor.get())
    {
        logger().log("Multiplication between two compressed but differently is super inefficient, please fix");
        vector<vectorType> decompressedOther;
        vector<vectorType> decompressedDest;
        vector<vectorType> compressedOther;
        vector<vectorType> compressedDest;

        compressor::deCompressVector(other,decompressedOther,otherCompressor);

        compressor::compressVector(decompressedOther,compressedOther,m_compressor);

        multiplyDecompressed(compressedOther,compressedDest);
        compressor::deCompressVector(compressedDest,decompressedDest,m_compressor);

        compressor::compressVector(decompressedDest,dest,otherCompressor);
    }
    else
    {
        multiplyDecompressed(other,dest);
    }

}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::rotate(realNumType angle, const vector<vectorType> &other, vector<vectorType> &dest) const
{
    double S = 0;
    double C = 0;
    mysincos(angle,&S,&C);
    rotate(S,C,other,dest);
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType,vectorType>::multiplyDecompressed(const vector<vectorType> &other, vector<vectorType> &dest) const
{
    const sparseMatrix<dataType,vectorType>& lhs = *this;

    dest.resize(other.size(),m_isCompressed,m_compressor); // need memset version

    auto d = lhs.m_data.begin();
    auto dEnd = lhs.m_data.end();
    auto iIdx = lhs.m_iIndexes.begin();
    auto jIdx = lhs.m_jIndexes.begin();
    const bool isrotationGenerator = m_isRotationGenerator;
    if (m_blankingVector.size() > 0)
    {
        while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
        {
            assert(*iIdx < other.size());
            assert(*jIdx < other.size());

            if (isrotationGenerator)
            {
                dest[*iIdx] += other[*jIdx]*vectorType(m_blankingVector[*iIdx]*m_blankingVector[*jIdx]);
                dest[*jIdx] -= other[*iIdx]*vectorType(m_blankingVector[*jIdx]*m_blankingVector[*jIdx]);
            }
            else
                dest[*iIdx] += other[*jIdx]*(*d)*vectorType(m_blankingVector[*iIdx]*m_blankingVector[*jIdx]);
            ++jIdx;
            ++iIdx;
            ++d;
        }
    }
    else
    {
        if (this->m_blankCount >= m_iSize)
        {
            while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
            {
                assert(*iIdx < other.size());
                assert(*jIdx < other.size());

                if (isrotationGenerator)
                {
                    dest[*iIdx] += other[*jIdx];
                    dest[*jIdx] -= other[*iIdx];
                }
                else
                    dest[*iIdx] += other[*jIdx]*(*d);
                ++jIdx;
                ++iIdx;
                ++d;
            }
        }
        else
        {
            while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
            {
                assert(*iIdx < other.size());
                assert(*jIdx < other.size());
                if (*iIdx < this->m_blankCount && *jIdx < this->m_blankCount)
                {
                    if (isrotationGenerator)
                    {
                        dest[*iIdx] += other[*jIdx];
                        dest[*jIdx] -= other[*iIdx];
                    }
                    else
                        dest[*iIdx] += other[*jIdx]*(*d);
                }
                ++jIdx;
                ++iIdx;
                ++d;
            }
        }
    }
}

template<>
void sparseMatrix<std::complex<realNumType>,std::complex<realNumType>>::multiplyDecompressed(const vector<std::complex<realNumType>> &other, vector<std::complex<realNumType>> &dest) const
{
    const sparseMatrix<std::complex<realNumType>,std::complex<realNumType>>& lhs = *this;

    dest.resize(other.size(),m_isCompressed,m_compressor); // need memset version

    auto d = lhs.m_data.begin();
    auto dEnd = lhs.m_data.end();
    auto iIdx = lhs.m_iIndexes.begin();
    auto jIdx = lhs.m_jIndexes.begin();
    const bool isrotationGenerator = m_isRotationGenerator;
    if (m_blankingVector.size() > 0)
    {
        while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
        {
            assert(*iIdx < other.size());
            assert(*jIdx < other.size());

            if (isrotationGenerator)
            {
                if (*iIdx == *jIdx)
                {
                    dest[*iIdx] += other[*iIdx]*iu*std::complex<realNumType>(m_blankingVector[*iIdx]*m_blankingVector[*iIdx]);
                }
                else
                {
                    dest[*iIdx] += other[*jIdx]*std::complex<realNumType>(m_blankingVector[*iIdx]*m_blankingVector[*jIdx]);
                    dest[*jIdx] -= other[*iIdx]*std::complex<realNumType>(m_blankingVector[*jIdx]*m_blankingVector[*jIdx]);
                }
            }
            else
                dest[*iIdx] += other[*jIdx]*(*d)*std::complex<realNumType>(m_blankingVector[*iIdx]*m_blankingVector[*jIdx]);
            ++jIdx;
            ++iIdx;
            ++d;
        }
    }
    else
    {
        if (this->m_blankCount >= m_iSize)
        {
            while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
            {
                assert(*iIdx < other.size());
                assert(*jIdx < other.size());

                if (isrotationGenerator)
                {
                    if (*iIdx == *jIdx)
                    {
                        dest[*iIdx] += other[*iIdx]*iu;
                    }
                    else
                    {
                        dest[*iIdx] += other[*jIdx];
                        dest[*jIdx] -= other[*iIdx];
                    }
                }
                else
                    dest[*iIdx] += other[*jIdx]*(*d);
                ++jIdx;
                ++iIdx;
                ++d;
            }
        }
        else
        {
            while(iIdx != lhs.m_iIndexes.end() && d < dEnd)
            {
                assert(*iIdx < other.size());
                assert(*jIdx < other.size());
                if (*iIdx < this->m_blankCount && *jIdx < this->m_blankCount)
                {
                    if (isrotationGenerator)
                    {
                        if (*iIdx == *jIdx)
                        {
                            dest[*iIdx] += other[*iIdx]*iu;
                        }
                        else
                        {
                            dest[*iIdx] += other[*jIdx];
                            dest[*jIdx] -= other[*iIdx];
                        }
                    }
                    else
                        dest[*iIdx] += other[*jIdx]*(*d);
                }
                ++jIdx;
                ++iIdx;
                ++d;
            }
        }
    }
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::rotate(realNumType S,realNumType C, const vector<vectorType> &other, vector<vectorType> &dest) const
{
    std::shared_ptr<compressor> otherCompressor;
    bool otherIsCompressed = other.getIsCompressed(otherCompressor);
    if (otherIsCompressed != m_isCompressed)
    {
        logger().log("Multiplication between compressed and uncompressed is inefficient, please fix");
        if (otherIsCompressed && !m_isCompressed)
        {
            vector<vectorType> decompressedOther;
            vector<vectorType> decompressedDest;
            compressor::deCompressVector(other,decompressedOther,otherCompressor);
            rotateDecompressed(S,C, decompressedOther,decompressedDest);
            compressor::compressVector(decompressedDest,dest,otherCompressor);
        }
        else if (!otherIsCompressed && m_isCompressed)
        {
            vector<vectorType> compressedOther;
            vector<vectorType> compressedDest;
            compressor::compressVector(other,compressedOther,m_compressor);
            rotateDecompressed(S,C, compressedOther,compressedDest);
            compressor::deCompressVector(compressedDest,dest,m_compressor);
        }
        else
        {
            logger().log("Not possible");
            __builtin_trap();
        }
    }
    else if (otherCompressor.get() != m_compressor.get())
    {
        logger().log("Multiplication between two compressed but differently is super inefficient, please fix");
        vector<vectorType> decompressedOther;
        vector<vectorType> decompressedDest;
        vector<vectorType> compressedOther;
        vector<vectorType> compressedDest;

        compressor::deCompressVector(other,decompressedOther,otherCompressor);

        compressor::compressVector(decompressedOther,compressedOther,m_compressor);

        rotateDecompressed(S,C, compressedOther,compressedDest);

        compressor::deCompressVector(compressedDest,decompressedDest,m_compressor);

        compressor::compressVector(decompressedDest,dest,otherCompressor);
    }
    else
    {
        rotateDecompressed(S,C,other,dest);
    }
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::rotateAndBraketWithTangentOfResult(realNumType S, realNumType C, const vector<vectorType> &other, vector<vectorType> &dest, const vector<vectorType> &toBraket, realNumType &result) const
{
    std::shared_ptr<compressor> otherComp;
    assert(other.getIsCompressed(otherComp) == m_isCompressed);
    assert(otherComp.get() == m_compressor.get());
    assert(toBraket.getIsCompressed(otherComp) == m_isCompressed);
    assert(otherComp.get() == m_compressor.get());

    if (!m_isRotationGenerator)
    {
        logger().log("Rotate called not on a rotation generator, Either implement this method or mark as rotation generator");
        __builtin_trap();
    }
    if (m_blankingVector.size() > 0 || this->m_blankCount < m_iSize)
    {
        logger().log("Rotate called on a blanked sparse matrix, implement this method");
        __builtin_trap();
    }
    result = 0;
    const sparseMatrix<dataType,vectorType>& lhs = *this;

    auto iIdx = lhs.m_iIndexes.begin();
    auto iEnd = lhs.m_iIndexes.end();
    auto jIdx = lhs.m_jIndexes.begin();
    if (&other != &dest)
        dest.copy(other);
    while(iIdx != iEnd)
    {
        vectorType srcI = dest[*iIdx];
        vectorType srcJ = dest[*jIdx];
        vectorType TBI = toBraket[*iIdx];
        vectorType TBJ = toBraket[*jIdx];
        vectorType destI = srcJ*(S) + srcI*C;
        vectorType destJ = -srcI*(S) + srcJ*C;
        dest[*iIdx] = destI;
        dest[*jIdx] = destJ;
        result += std::real(myConj(TBI)*destJ);
        result -= std::real(myConj(TBJ)*destI);

        ++jIdx;
        ++iIdx;
    }
}

template<>
void sparseMatrix<std::complex<realNumType>, std::complex<realNumType>>::rotateAndBraketWithTangentOfResult(realNumType S, realNumType C, const vector<std::complex<realNumType>> &other, vector<std::complex<realNumType>> &dest, const vector<std::complex<realNumType>> &toBraket, realNumType &result) const
{
    std::shared_ptr<compressor> otherComp;
    assert(other.getIsCompressed(otherComp) == m_isCompressed);
    assert(otherComp.get() == m_compressor.get());
    assert(toBraket.getIsCompressed(otherComp) == m_isCompressed);
    assert(otherComp.get() == m_compressor.get());

    if (!m_isRotationGenerator)
    {
        logger().log("Rotate called not on a rotation generator, Either implement this method or mark as rotation generator");
        __builtin_trap();
    }
    if (m_blankingVector.size() > 0 || this->m_blankCount < m_iSize)
    {
        logger().log("Rotate called on a blanked sparse matrix, implement this method");
        __builtin_trap();
    }
    result = 0;
    const sparseMatrix<std::complex<realNumType>,std::complex<realNumType>>& lhs = *this;

    auto iIdx = lhs.m_iIndexes.begin();
    auto iEnd = lhs.m_iIndexes.end();
    auto jIdx = lhs.m_jIndexes.begin();
    if (&other != &dest)
        dest.copy(other);
    while(iIdx != iEnd)
    {
        if (*iIdx == *jIdx)
        {
            std::complex<realNumType> srcI = dest[*iIdx];
            std::complex<realNumType> TBI = toBraket[*iIdx];
            std::complex<realNumType> destI = srcI*iu*(S) + srcI*C;
            dest[*iIdx] = destI;
            result += std::real(myConj(TBI)*destI*iu);
        }
        else
        {
            std::complex<realNumType> srcI = dest[*iIdx];
            std::complex<realNumType> srcJ = dest[*jIdx];
            std::complex<realNumType> TBI = toBraket[*iIdx];
            std::complex<realNumType> TBJ = toBraket[*jIdx];
            std::complex<realNumType> destI = srcJ*(S) + srcI*C;
            std::complex<realNumType> destJ = -srcI*(S) + srcJ*C;
            dest[*iIdx] = destI;
            dest[*jIdx] = destJ;
            result += std::real(myConj(TBI)*destJ);
            result -= std::real(myConj(TBJ)*destI);
        }

        ++jIdx;
        ++iIdx;
    }
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::compress(std::shared_ptr<compressor> comp)
{
    if (m_isCompressed)
    {
        decompress();
    }

    m_compressor = comp;
    m_isCompressed = true;

    std::vector<uint32_t> iIndexes = std::move(m_iIndexes);
    std::vector<uint32_t> jIndexes = std::move(m_jIndexes);
    std::vector<dataType> data = std::move(m_data);
    uint32_t N = jIndexes.size();

    m_iIndexes.clear();
    m_jIndexes.clear();
    m_data.clear();

    for (size_t i = 0; i < N; i++)
    {
        uint32_t compressediIndex;
        uint32_t compressedjIndex;
        if (m_compressor->compressIndex(iIndexes[i],compressediIndex) &&
            m_compressor->compressIndex(jIndexes[i],compressedjIndex))
        {
            m_iIndexes.push_back(compressediIndex);
            m_jIndexes.push_back(compressedjIndex);
            m_data.push_back(data[i]);
        }
    }

    unblankAll();
}

template<typename dataType, typename vectorType>
void sparseMatrix<dataType, vectorType>::decompress()
{


    std::vector<uint32_t> iIndexes = std::move(m_iIndexes);
    std::vector<uint32_t> jIndexes = std::move(m_jIndexes);
    std::vector<dataType> data = std::move(m_data);
    uint32_t N = jIndexes.size();

    m_iIndexes.clear();
    m_jIndexes.clear();
    m_data.clear();

    for (size_t i = 0; i < N; i++)
    {
        uint32_t decompressediIndex;
        uint32_t decompressedjIndex;
        if (m_compressor->deCompressIndex(iIndexes[i],decompressediIndex) &&
            m_compressor->deCompressIndex(jIndexes[i],decompressedjIndex))
        {
            m_iIndexes.push_back(decompressediIndex);
            m_jIndexes.push_back(decompressedjIndex);
            m_data.push_back(data[i]);
        }
    }

    unblankAll();
    m_compressor = nullptr;
    m_isCompressed = false;
}

template<>
void sparseMatrix<realNumType, realNumType>::rotateDecompressed(realNumType S,realNumType C, const vector<realNumType> &other, vector<realNumType> &dest) const
{
    if (!m_isRotationGenerator)
    {
        logger().log("Rotate called not on a rotation generator, Either implement this method or mark as rotation generator");
        __builtin_trap();
    }
    if (m_blankingVector.size() > 0 || this->m_blankCount < m_iSize)
    {
        logger().log("Rotate called on a blanked sparse matrix, implement this method");
        __builtin_trap();
    }
    const sparseMatrix<realNumType,realNumType>& lhs = *this;

    auto iIdx = lhs.m_iIndexes.begin();
    auto iEnd = lhs.m_iIndexes.end();
    auto jIdx = lhs.m_jIndexes.begin();
    if (&other != &dest)
        dest.copy(other);
    while(iIdx != iEnd)
    {
        realNumType srcI = dest[*iIdx];
        realNumType srcJ = dest[*jIdx];
        dest[*iIdx] = srcJ*(S) + srcI*C;
        dest[*jIdx] = -srcI*(S) + srcJ*C;

        ++jIdx;
        ++iIdx;
    }
}

template<>
void sparseMatrix<realNumType, std::complex<realNumType>>::rotateDecompressed(realNumType S,realNumType C, const vector<std::complex<realNumType>> &other, vector<std::complex<realNumType>> &dest) const
{
    if (!m_isRotationGenerator)
    {
        logger().log("Rotate called not on a rotation generator, Either implement this method or mark as rotation generator");
        __builtin_trap();
    }
    if (m_blankingVector.size() > 0 || this->m_blankCount < m_iSize)
    {
        logger().log("Rotate called on a blanked sparse matrix, implement this method");
        __builtin_trap();
    }
    const sparseMatrix<realNumType,std::complex<realNumType>>& lhs = *this;

    auto iIdx = lhs.m_iIndexes.begin();
    auto iEnd = lhs.m_iIndexes.end();
    auto jIdx = lhs.m_jIndexes.begin();
    if (&other != &dest)
        dest.copy(other);
    while(iIdx != iEnd)
    {
        std::complex<realNumType> srcI = dest[*iIdx];
        std::complex<realNumType> srcJ = dest[*jIdx];
        dest[*iIdx] = srcJ*(S) + srcI*C;
        dest[*jIdx] = -srcI*(S) + srcJ*C;

        ++jIdx;
        ++iIdx;
    }
}

template<>
void sparseMatrix<std::complex<realNumType>, std::complex<realNumType>>::rotateDecompressed(realNumType S,realNumType C, const vector<std::complex<realNumType>> &other, vector<std::complex<realNumType>> &dest) const
{
    if (!m_isRotationGenerator)
    {
        logger().log("Rotate called not on a rotation generator, Either implement this method or mark as rotation generator");
        __builtin_trap();
    }
    if (m_blankingVector.size() > 0 || this->m_blankCount < m_iSize)
    {
        logger().log("Rotate called on a blanked sparse matrix, implement this method");
        __builtin_trap();
    }
    const sparseMatrix<std::complex<realNumType>,std::complex<realNumType>>& lhs = *this;

    auto iIdx = lhs.m_iIndexes.begin();
    auto iEnd = lhs.m_iIndexes.end();
    auto jIdx = lhs.m_jIndexes.begin();
    if (&other != &dest)
        dest.copy(other);
    while(iIdx != iEnd)
    {
        if (*iIdx == *jIdx)
        {
            std::complex<realNumType> srcI = dest[*iIdx];
            dest[*iIdx] = srcI*iu*(S) + srcI*C;
        }
        else
        {
            std::complex<realNumType> srcI = dest[*iIdx];
            std::complex<realNumType> srcJ = dest[*jIdx];
            dest[*iIdx] = srcJ*(S) + srcI*C;
            dest[*jIdx] = -srcI*(S) + srcJ*C;
        }

        ++jIdx;
        ++iIdx;
    }
}


template<typename dataType, typename vectorType>
sparseMatrix<dataType,vectorType>::operator sparseMatrix<dataType,vectorType>::EigenSparseMatrix() const
{
    uint32_t rowNumber = m_iSize;
    uint32_t columnNumber = m_jSize;
    EigenSparseMatrix ret(rowNumber,columnNumber);
    auto iBegin = iItBegin();
    auto iEnd = iItEnd();
    auto jBegin = jItBegin();
    auto dataBegin = begin();

    typedef Eigen::Triplet<dataType> T;
    std::vector<T> tripletList;
    tripletList.reserve(m_data.size());
    while(iBegin != iEnd)
    {
        tripletList.push_back(T(*iBegin,*jBegin,*dataBegin));
        iBegin++;
        jBegin++;
        dataBegin++;

    }
    ret.setFromTriplets(tripletList.begin(), tripletList.end());

    return ret;
}



template<typename dataType>
projectionMatrix<dataType>::projectionMatrix(std::vector<vector<dataType>>&& basisVectors)
{
    m_basisVectors = std::move(basisVectors);
    for (vector<dataType>& v : m_basisVectors)
    {
        realNumType MagSq = v.dot(v);
        v *= 1/std::sqrt(MagSq);
    }
    unblankAll();
}

template<typename dataType>
bool projectionMatrix<dataType>::addBlank()
{
    size_t& blankCount = this->m_blankCount;

    blankCount++;
    if (blankCount > m_basisVectors.size())
    {
        blankCount = m_basisVectors.size();
        return false;
    }
    return true;
}

template<typename dataType>
void projectionMatrix<dataType>::multiply(const vector<dataType> &other, vector<dataType> &dest) const
{
    dest.resize(m_basisVectors[0].size(),false,nullptr);
    vector<dataType> temp;
    size_t count = 0;
    for (const vector<dataType>& v : m_basisVectors)
    {
        if (count >= this->m_blankCount)
            break;
        realNumType proj = v.dot(other);
        mul(proj,v,temp);
        dest += temp;
        count++;
    }
}

#ifdef useComplex
template class sparseMatrix<realNumType,realNumType>;
template class sparseMatrix<realNumType,numType>;
template class sparseMatrix<numType,numType>;

template class projectionMatrix<realNumType>;
template class projectionMatrix<numType>;
#else
template class sparseMatrix<numType,numType>;
template class projectionMatrix<numType>;
#endif
