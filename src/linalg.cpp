/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "linalg.h"
#include "globals.h"
#include "threadpool.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <future>
static const int alignment = 32;


template<typename dataType>
void Matrix<dataType>::cleanup()
{
    if (m_data == nullptr)
            return;
    operator delete[](m_data,std::align_val_t(alignment));
    m_data = nullptr;
    m_isCompressed = false;
    m_compressor = nullptr;
}

template<typename dataType>
void Matrix<dataType>::setZero()
{
    dataType* ptr = m_data;
    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            *ptr++ = dataType(0);
        }
    }
}

template<typename dataType>
Matrix<dataType>::Matrix(size_t iSize, size_t jSize)
{
    this->m_iSize = iSize;
    this->m_jSize = jSize;
    if (iSize*jSize == 0)
        return;
    m_data = new(std::align_val_t(alignment)) dataType[m_iSize*m_jSize];
    setZero();
}

template<typename dataType>
Matrix<dataType>::Matrix(int value, const std::vector<uint32_t>& iIndex, const std::vector<uint32_t>& jIndex, int size)
{
    m_iSize = m_jSize = size;
    assert(iIndex.size() == jIndex.size());
    assert(iIndex.size() <= (size_t)m_iSize);
    if (size == 0)
        return;

    m_data = new(std::align_val_t(alignment)) dataType[size*size];
    setZero();

    for (size_t i = 0; i < iIndex.size(); i++)
    {
        assert(iIndex[i] < m_iSize);
        assert(jIndex[i] < m_jSize);
        m_data[iIndex[i]*m_jSize + jIndex[i]] = value;
    }
}

template<typename dataType>
Matrix<dataType>::Matrix(const dataType *buffer, size_t iSize, size_t jSize)
{
    copyFromBuffer(buffer,iSize,jSize);
}

//template<typename dataType>
//Matrix<dataType>::Matrix(const Matrix &other) : Matrix(other.m_iSize,other.m_jSize)
//{
//    if (other.m_iSize*other.m_jSize != 0)
//        memcpy(m_data,other.m_data,sizeof(dataType)*m_iSize*m_jSize);
//}

template<typename dataType>
Matrix<dataType>::Matrix(Matrix &&other) noexcept
{
    m_iSize = other.m_iSize;
    m_jSize = other.m_jSize;


    m_data = other.m_data;
    other.m_data = nullptr;
    m_isCompressed = other.m_isCompressed;
    m_compressor = std::move(other.m_compressor);
}


template<typename dataType>
Matrix<dataType> &Matrix<dataType>::operator+=(const Matrix &other)
{
    assert(m_iSize == other.m_iSize && m_jSize == other.m_jSize);
    assert(m_isCompressed == other.m_isCompressed && (m_isCompressed ? other.m_compressor.get() == m_compressor.get() : true));


    dataType* dataPtr = m_data;
    dataType* otherDataPtr = other.m_data;

    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            *dataPtr++ += *otherDataPtr++;
        }
    }
    return *this;
}

template<>
Matrix<double> &Matrix<double>::addEquals(const Matrix &otherA, const Matrix &otherB)
{
    assert(m_iSize == otherA.m_iSize && m_jSize == otherA.m_jSize);
    assert(m_iSize == otherB.m_iSize && m_jSize == otherB.m_jSize);
    assert(m_isCompressed == otherA.m_isCompressed && (m_isCompressed ? otherA.m_compressor.get() == m_compressor.get() : true));
    assert(m_isCompressed == otherB.m_isCompressed && (m_isCompressed ? otherB.m_compressor.get() == m_compressor.get() : true));

    double* dataPtr = m_data;
    double* otherADataPtr = otherA.m_data;
    double* otherBDataPtr = otherB.m_data;
#ifdef vectoriseScalarMul
    int i = m_iSize*m_jSize;
    for ( ;i > 4; i-=4)
    {
        auto Double4a = _mm256_load_pd(otherADataPtr);
        auto Double4b = _mm256_load_pd(otherBDataPtr);
        auto Double4c = _mm256_load_pd(dataPtr);

        Double4a = _mm256_add_pd(Double4a,Double4b);
        Double4a = _mm256_add_pd(Double4a,Double4c);
        _mm256_store_pd(dataPtr,Double4a);
        otherADataPtr+=4;
        otherBDataPtr+=4;
        dataPtr+=4;
    }
    for (;i>0; i--)
    {
        *dataPtr++ += *otherADataPtr++ + *otherBDataPtr++;
    }
#else
    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            *dataPtr++ += *otherADataPtr++ + *otherBDataPtr++;
        }
    }
#endif
    return *this;
}

template<typename dataType>
Matrix<dataType> &Matrix<dataType>::addEquals(const Matrix &otherA, const Matrix &otherB)
{
    assert(m_iSize == otherA.m_iSize && m_jSize == otherA.m_jSize);
    assert(m_iSize == otherB.m_iSize && m_jSize == otherB.m_jSize);
    assert(m_isCompressed == otherA.m_isCompressed && (m_isCompressed ? otherA.m_compressor.get() == m_compressor.get() : true));
    assert(m_isCompressed == otherB.m_isCompressed && (m_isCompressed ? otherB.m_compressor.get() == m_compressor.get() : true));

    dataType* dataPtr = m_data;
    dataType* otherADataPtr = otherA.m_data;
    dataType* otherBDataPtr = otherB.m_data;

    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            *dataPtr++ += *otherADataPtr++ + *otherBDataPtr++;
        }
    }
    return *this;
}

template<typename dataType>
Matrix<dataType> &Matrix<dataType>::operator*=(dataType scalar)
{
    this->mul(scalar,*this);
    return *this;
}

template<typename dataType>
Matrix<dataType> &Matrix<dataType>::copy(const Matrix &other)
{
    if (m_iSize == other.m_iSize && m_jSize == other.m_jSize && m_iSize*m_jSize != 0)
    {
        memmove(m_data,other.m_data,sizeof(dataType)*m_iSize*m_jSize);
    }
    else
    {
        cleanup();
        this->m_iSize = other.m_iSize;
        this->m_jSize = other.m_jSize;
        if (m_iSize*m_jSize != 0)
        {
            m_data = new(std::align_val_t(alignment)) dataType[m_iSize*m_jSize];
            memcpy(m_data,other.m_data, sizeof(dataType)*m_iSize*m_jSize);
        }
    }
    m_isCompressed = other.m_isCompressed;
    m_compressor = other.m_compressor;
    return *this;
}

template<typename dataType>
Matrix<dataType> &Matrix<dataType>::copyFromBuffer(const dataType *buffer, size_t iSize, size_t jSize)
{
    cleanup();
    this->m_iSize = iSize;
    this->m_jSize = jSize;
    if (iSize*jSize == 0)
        return *this;
    m_data = new(std::align_val_t(alignment)) dataType[m_iSize*m_jSize];
    memcpy(m_data,buffer,sizeof(dataType)*m_iSize*m_jSize);
    return *this;
}


template<typename dataType>
Matrix<dataType> &Matrix<dataType>::operator=(Matrix&& other) noexcept
{
    std::swap(m_iSize, other.m_iSize);
    std::swap(m_jSize, other.m_jSize);
    std::swap(m_data, other.m_data);
    m_isCompressed = other.m_isCompressed;
    m_compressor = other.m_compressor;
    return *this;
}

template<typename dataType>
Matrix<dataType> Matrix<dataType>::T()
{
    Matrix ret(m_jSize,m_iSize);
    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            ret.m_data[j*ret.m_jSize + i] = m_data[i*m_jSize + j];
        }
    }
    return ret;
}



template<>
void Matrix<double>::mul(double scalar, Matrix<double> &dest) const
{
    const Matrix<double>& mat = *this;
    dest.resize(mat.m_iSize,mat.m_jSize,m_isCompressed,m_compressor,true);

#ifdef vectoriseScalarMul
    double* destPtr = dest.m_data;
    double* srcPtr = mat.m_data;
    auto scalar4 = _mm256_broadcast_sd(&scalar);
    int i = mat.m_iSize*mat.m_jSize;
    for ( ;i > 4; i-=4)
    {
        auto Double4 = _mm256_load_pd(srcPtr);
        Double4 = _mm256_mul_pd(Double4,scalar4);
        _mm256_store_pd(destPtr,Double4);
        srcPtr+=4;
        destPtr+=4;
    }
    for (;i>0; i--)
    {
        *destPtr++ = scalar*(*srcPtr);
        srcPtr++;
    }

#else
    for (size_t i = 0; i < mat.m_iSize; i++)
    {
        for (size_t j = 0; j < mat.m_jSize; j++)
        {
            dest.m_data[i*dest.m_jSize + j] = scalar*mat.m_data[i*dest.m_jSize + j];
        }
    }
#endif
}

template<typename dataType>
void Matrix<dataType>::mul(dataType scalar, Matrix<dataType> &dest) const
{
    dest.resize(m_iSize,m_jSize,m_isCompressed,m_compressor,true);
    for (size_t i = 0; i < m_iSize; i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            dest.m_data[i*dest.m_jSize + j] = scalar*m_data[i*dest.m_jSize + j];
        }
    }
}

template<typename dataType>
Matrix<realNumType> Matrix<dataType>::real()
{
    Matrix<realNumType> ret;
    ret.resize(m_iSize,m_jSize,m_isCompressed,m_compressor);
    for (size_t i = 0; i < m_iSize;i++)
    {
        for (size_t j = 0; j < m_jSize; j++)
        {
            ret(i,j) = std::real((*this)(i,j));
        }
    }
    return ret;
}

template<typename dataType>
Matrix<dataType>::~Matrix()
{
    cleanup();
}

template<typename dataType>
bool Matrix<dataType>::allClose(dataType val, realNumType atol)
{
    for (size_t i = 0; i < m_iSize*m_jSize; i++)
        if (std::abs(m_data[i]-val) > atol)
            return false;
    return true;
}

template<typename dataType>
void Matrix<dataType>::resize(size_t iSize, size_t jSize, bool isCompressed, std::shared_ptr<compressor> compressor)
{
    if (m_iSize != iSize || m_jSize != jSize)
    {
        cleanup();
        m_iSize = iSize;
        m_jSize = jSize;
        if (iSize*jSize == 0)
            return;
        m_data = new(std::align_val_t(alignment)) dataType[m_iSize*m_jSize];
    }
    m_isCompressed = isCompressed;
    m_compressor = compressor;
    //    numType* destPtr = m_data;
    //    numType zero = 0;
    //    auto scalar4 = _mm256_broadcast_sd(&zero);
    //    int i = m_iSize*m_jSize;
    //    for ( ;i > 4; i-=4)
    //    {
    //        _mm256_store_pd(destPtr,scalar4);
    //        destPtr+=4;
    //    }
    //    for (;i>0; i--)
    //    {
    //        *destPtr++ = 0;
    //    }
    setZero();
}

template<typename dataType>
void Matrix<dataType>::resize(size_t iSize, size_t jSize, bool isCompressed, std::shared_ptr<compressor> compressor, bool)
//no memset
{
    if (m_iSize != iSize || m_jSize != jSize)
    {
        cleanup();
        m_iSize = iSize;
        m_jSize = jSize;
        if (iSize*jSize == 0)
            return;
        m_data = new(std::align_val_t(alignment)) dataType[m_iSize*m_jSize];
    }
    m_isCompressed = isCompressed;
    m_compressor = compressor;
}

template<typename dataType>
void Matrix<dataType>::swap(Matrix &other)
{
    std::swap(m_iSize, other.m_iSize);
    std::swap(m_jSize, other.m_jSize);
    std::swap(m_data, other.m_data);
    std::swap(m_isCompressed,other.m_isCompressed);
    std::swap(m_compressor,other.m_compressor);
}

template<typename dataType>
void asyncProd(dataType* dest, dataType* srcA, dataType* srcB, size_t N)
{
    std::atomic_int finishCount = 0;
    auto multiply = [&](dataType* destStart, dataType* destEnd, dataType* srcA, dataType* srcB)
    {
        while (destStart < destEnd)
            *destStart++ = *srcA++ * *srcB++;
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
        return true;
    };

    const int stepSize = std::max(N/NUM_CORES,1ul);;
    std::vector<std::future<void>> futures;
    threadpool& pool = threadpool::getInstance(NUM_CORES);

    for (size_t i = 0; i < N; i += stepSize)
        futures.push_back(pool.queueWork([&,i](){multiply(dest + i, dest + std::min(i+stepSize, N ), srcA + i, srcB + i );}));

    for (auto &f: futures)
        f.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

template<typename dataType>
Matrix<dataType>& Matrix<dataType>::prod(const Matrix &other)
{
    assert(m_iSize == other.m_iSize && m_jSize == other.m_jSize);
    assert(m_isCompressed == other.m_isCompressed && (m_isCompressed ? other.m_compressor.get() == m_compressor.get() : true));

    dataType* dataPtr = m_data;
    dataType* otherDataPtr = other.m_data;
    asyncProd(dataPtr,dataPtr,otherDataPtr,m_iSize*m_jSize);
    return *this;
}

template<typename dataType>
vector<dataType>::vector(const EigenVector &vec):Matrix<dataType>(1,vec.rows())
{
    for (size_t i = 0; i < size(); i++)
    {
        (*this)[i] = vec(i);
    }
}

template<typename dataType>
vector<dataType>::vector(const std::vector<dataType>& data) : vector(data.size())
{
    std::copy(data.begin(),data.end(),this->m_data);
}

template<typename dataType>
vector<dataType>::operator vector<dataType>::EigenVector() const
{
    vector<dataType>::EigenVector ret(this->m_jSize);
    Eigen::Map<Eigen::Matrix<dataType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> thisMap(this->m_data,this->m_iSize,this->m_jSize);
    ret = thisMap;
    return ret;
}






template<typename dataType> // NOTE this returns the REAL part of the complex vector product!!!!!
realNumType vector<dataType>::dot(const vector &other) const
{
    //Ugly code:
    if constexpr (std::is_same_v<typename Eigen::NumTraits<dataType>::Real,dataType>)
    {
        dataType ret;
        Eigen::Map<Eigen::Matrix<dataType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> thisMap(this->m_data,this->m_iSize,this->m_jSize);
        Eigen::Map<Eigen::Matrix<dataType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> otherMap(other.m_data,other.m_iSize,other.m_jSize);
        ret = thisMap.dot(otherMap);
        return ret;
    }
    else if constexpr(std::is_same_v<std::complex<typename Eigen::NumTraits<dataType>::Real>,dataType>)
    {//complex
        typedef typename Eigen::NumTraits<dataType>::Real realT;
        realT  ret;
        Eigen::Map<Eigen::Matrix<realT,1,-1,Eigen::RowMajor>,Eigen::Aligned32> thisMap((realT*)this->m_data,this->m_iSize,this->m_jSize*2);
        Eigen::Map<Eigen::Matrix<realT,1,-1,Eigen::RowMajor>,Eigen::Aligned32> otherMap((realT*)other.m_data,other.m_iSize,other.m_jSize*2);
        ret = thisMap.dot(otherMap);
        return ret;
    }
    else
        static_assert(std::is_same_v<typename Eigen::NumTraits<dataType>::Real,dataType> || std::is_same_v<std::complex<typename Eigen::NumTraits<dataType>::Real>,dataType>,"Unknown type?");



    //Nice code:
    // realNumType ret = 0;
    // for (size_t i = 0; i < other.size(); i++)
    // {//TODO something like float16 and std::complex<float> may break this? IF so it breaks into assuming it is complex so this is fine.
    //     if constexpr (std::is_same_v<dataType, decltype(std::real(dataType()))>)
    //         ret += (*this)[i] * other[i];
    //     else
    //         ret += std::real(std::conj((*this)[i]) * other[i]);
    // }
    // return ret;

}


template<typename dataType>
std::complex<realNumType> vector<dataType>::cdot(const vector &other) const
{
    if constexpr(std::is_same_v<dataType, decltype(std::real(dataType()))>)
        return this->dot(other);
    else
    {
        assert(size() == other.size());

        std::complex<realNumType> ret = 0;
        for (size_t i = 0; i < other.size(); i++)
            ret += std::conj((*this)[i]) * other[i];
        return ret;
    }
}

template<typename dataType>
void vector<dataType>::partialScalarMul(size_t startIndex, size_t endIndex,dataType scalar)
{

    while (startIndex < endIndex && (size_t)&this->m_data[startIndex] % 32 != 0)
    {//fix alignment
        this->m_data[startIndex]*=scalar;
        startIndex++;
    }
    if (startIndex >= endIndex)
        return;

    //TODO use the new view class
    vector<dataType> view;
    view.m_jSize = endIndex-startIndex;
    view.m_iSize = 1;
    view.m_data = &this->m_data[startIndex]; //ugly pointers
    view*=scalar;
    view.m_data = nullptr;
    view.m_jSize = 0;
}


template<typename dataType>
void vector<dataType>::conj()
{
    if constexpr(std::is_same_v<dataType, decltype(std::real(dataType()))>)
        return;
    else
    {
        for (size_t i = 0; i < size(); i++)
        {
            this->m_data[i] = std::conj(this->m_data[i]);
        }
    }
}

template<typename dataType>
vector<realNumType> vector<dataType>::real()
{
    vector<realNumType> ret;
    ret.resize(size(),this->m_isCompressed,this->m_compressor);

    for (size_t j = 0; j < size(); j++)
    {
        ret[j] = std::real((*this)[j]);
    }

    return ret;
}

template<typename dataType>
void vector<dataType>::normalize()
{
    Eigen::Map<Eigen::Matrix<dataType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> thisMap(this->m_data,this->m_iSize,this->m_jSize);
    thisMap.normalize();
}

Matrix<numType>::EigenMatrix convert(const std::vector<vector<numType> > &AnsatzTangentSpace)
{
    Matrix<numType>::EigenMatrix ret(AnsatzTangentSpace.size(),AnsatzTangentSpace[0].size());
    for (size_t i = 0; i < AnsatzTangentSpace.size();i++)
        for (size_t j = 0; j < AnsatzTangentSpace[0].size(); j++)
            ret(i,j) = AnsatzTangentSpace[i][j];
    return ret;
}

template class Matrix<std::complex<realNumType>>;
template class Matrix<realNumType>;
template class vector<std::complex<realNumType>>;
template class vector<realNumType>;

template class vectorView<Matrix<std::complex<realNumType>>>;
template class vectorView<Matrix<realNumType>>;
template class vectorView<const Matrix<std::complex<realNumType>>>;
template class vectorView<const Matrix<realNumType>>;

template class vectorView<Matrix<std::complex<realNumType>>, Eigen::ColMajor>;
template class vectorView<Matrix<realNumType>, Eigen::ColMajor>;
template class vectorView<const Matrix<std::complex<realNumType>>, Eigen::ColMajor>;
template class vectorView<const Matrix<realNumType>, Eigen::ColMajor>;
