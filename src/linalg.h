/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef LINALG_H
#define LINALG_H

#include "globals.h"

#include "Eigen/Eigenvalues"


#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#ifdef __AVX__
#define vectoriseScalarMul
#else
#warning not using AVX
#endif


#ifdef vectoriseScalarMul
#include <immintrin.h>
#endif

#ifdef APPLECLANG
#define mysincos __sincos
#else
#define mysincos sincos
#endif


class compressor;

template<typename dataType>
class Matrix
{
protected:
    void cleanup();
    dataType* m_data=nullptr;
    void setZero();
    bool m_isCompressed = false;
    std::shared_ptr<compressor> m_compressor;
public:
    typedef Eigen::Matrix<dataType, Eigen::Dynamic, Eigen::Dynamic> EigenMatrix;
    size_t m_iSize;
    size_t m_jSize;

    Matrix():Matrix(0,0){};
    Matrix(size_t iSize, size_t jSize);
    Matrix(int value, const std::vector<uint32_t>& iIndex, const std::vector<uint32_t>& jIndex,int size);
    Matrix(const dataType* buffer, size_t iSize, size_t jSize);




    virtual ~Matrix();



    Matrix& operator+= (const Matrix& other);
    Matrix& addEquals (const Matrix& otherA, const Matrix& otherB);
    Matrix& operator*= (dataType scalar);

    Matrix(const Matrix& other) = delete;
    Matrix(Matrix&& other) noexcept;
    Matrix& operator= (const Matrix& other) = delete;
    Matrix& copy (const Matrix& other);
    Matrix& copyFromBuffer(const dataType* buffer, size_t iSize, size_t jSize);

    Matrix& operator= (Matrix&& other) noexcept;

    inline dataType& at(size_t i,size_t j){assert(i < m_iSize && j < m_jSize); return m_data[i*m_jSize + j];}
    inline const dataType& at(size_t i,size_t j)const{assert(i < m_iSize && j < m_jSize); return m_data[i*m_jSize + j];}


    bool allClose(dataType val,realNumType atol=1e-10);
    void resize(size_t iSize, size_t jSize,bool isCompressed, std::shared_ptr<compressor> compressor);
    void resize(size_t iSize, size_t jSize,bool isCompressed, std::shared_ptr<compressor> compressor,bool); // no memset 0
    void swap(Matrix& other);

    Matrix& prod(const Matrix& other);


    friend void prod(const Matrix<dataType>& first, const Matrix<dataType>& other, Matrix<dataType>& dest)
    {
        assert(first.m_iSize == other.m_iSize && first.m_jSize == other.m_jSize);

        dest.resize(first.m_iSize,first.m_jSize,true);

        dataType* firstDataPtr = first.m_data;
        dataType* otherDataPtr = other.m_data;
        dataType* destDataPtr = dest.m_data;

        asyncProd(destDataPtr,firstDataPtr,otherDataPtr,first.m_iSize*first.m_jSize);
    }

    virtual Matrix T();

    friend void mul (const Matrix<dataType>& lhs, const Matrix<dataType>& other, Matrix<dataType>& dest)
    {
        assert(lhs.m_jSize == other.m_iSize);

        dest.resize(lhs.m_iSize,other.m_jSize,true);

        for (int i = 0; i < lhs.m_iSize; i++)
        {
            for (int j = 0; j < other.m_jSize; j++)
            {
                dest.m_data[i*dest.m_jSize + j] = 0;
                for (int k = 0; k < lhs.m_jSize; k++)
                    dest.m_data[i*dest.m_jSize + j] += lhs.m_data[i*lhs.m_jSize + k]*other.m_data[k*other.m_jSize + j];
            }
        }
    }

    void mul(dataType scalar, Matrix<dataType>& dest) const;
    friend void mul(dataType scalar, const Matrix<dataType>& mat, Matrix<dataType>& dest)
    {
        mat.mul(scalar,dest);
    }
    friend void add (const Matrix<dataType>& first, const Matrix<dataType>& other, Matrix<dataType>& dest)
    {
        assert(first.m_iSize == other.m_iSize && first.m_jSize == other.m_jSize);
        assert(first.m_isCompressed == other.m_isCompressed && (first.m_isCompressed ? other.m_compressor.get() == first.m_compressor.get() : true));

        dest.resize(first.m_iSize,first.m_jSize,first.m_isCompressed,first.m_compressor,true);

        dataType* dataPtr = first.m_data;
        dataType* otherDataPtr = other.m_data;
        dataType* destPtr = dest.m_data;

        for (size_t i = 0; i < first.m_iSize; i++)
        {
            for (size_t j = 0; j < first.m_jSize; j++)
            {
                *destPtr++ = *dataPtr++ + *otherDataPtr++;
            }
        }
    }
    friend void sub (const Matrix<dataType>& first, const Matrix<dataType>& other, Matrix<dataType>& dest)
    {
        assert(first.m_iSize == other.m_iSize && first.m_jSize == other.m_jSize);
        assert(first.m_isCompressed == other.m_isCompressed && (first.m_isCompressed ? other.m_compressor.get() == first.m_compressor.get() : true));

        dest.resize(first.m_iSize,first.m_jSize,first.m_isCompressed,first.m_compressor,true);

        dataType* dataPtr = first.m_data;
        dataType* otherDataPtr = other.m_data;
        dataType* destPtr = dest.m_data;

        for (size_t i = 0; i < first.m_iSize; i++)
        {
            for (size_t j = 0; j < first.m_jSize; j++)
            {
                *destPtr++ = *dataPtr++ - *otherDataPtr++;
            }
        }
    }
    bool getIsCompressed(std::shared_ptr<compressor>& ptr) const {ptr = m_compressor; return m_isCompressed; }
    void setIsCompressed(bool val, std::shared_ptr<compressor> compressor){m_isCompressed = val; m_compressor = compressor;}
};



template<typename dataType>
class vector : public Matrix<dataType>
{
    vector(const vector &other) = delete;
    vector& operator= (const vector& other) = delete;

public:
    typedef Eigen::Vector<dataType,Eigen::Dynamic> EigenVector;
    vector():vector(0){}
    vector(int size):Matrix<dataType>(size,1){}
    vector(const EigenVector& vec);
    vector(const std::vector<dataType>& data);

    vector(const Matrix<dataType>& other) = delete;
    vector(dataType* buffer, size_t size) : Matrix<dataType> (buffer,size,1){}

    vector(Matrix<dataType>&& other) noexcept : Matrix<dataType>(std::move(other))
    {
        assert(this->m_jSize == 1);
    }
    vector(vector&& other) noexcept: Matrix<dataType>(std::move(other))
    {
        assert(this->m_jSize == 1);
    }






    dataType& operator[](int size){return this->at(size,0);}
    const dataType& operator[](int size)const{return this->at(size,0);}
    operator EigenVector () const
    {
        EigenVector ret(this->m_iSize);
        for (size_t i = 0; i < this->m_iSize; i++)
            ret[i] = (*this)[i];
        return ret;
    }

    realNumType dot(const vector &other) const;
    std::complex<realNumType> cdot(const vector &other) const; //always returns a complex value.
    void partialScalarMul(size_t startIndex, size_t endIndex, dataType scalar);
    //vector& operator *= (const vector &other);
    size_t size() const {return this->m_iSize;}
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor)
    {this->Matrix<dataType>::resize(size,1,isCompressed,compressor);}
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor,bool)
    {this->Matrix<dataType>::resize(size,1,isCompressed,compressor,true);} // no memset 0
    vector& copyFromBuffer(dataType* buffer, size_t size) {this->Matrix<dataType>::copyFromBuffer(buffer,size,1); return *this;}
    void conj();


    const dataType* begin() const {return this->m_data;}
    const dataType* end() const {return this->m_data + size();}


};

//Convert from the represenation used in Ansatz to the Eigen format. Not sure where to put this TODO
Matrix<numType>::EigenMatrix convert(const std::vector<vector<numType>>& AnsatzTangentSpace);

#endif // LINALG_H
