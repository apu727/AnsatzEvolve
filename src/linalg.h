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
#if defined __has_builtin
#  if __has_builtin (__builtin_popcount)
#    define popcount(i) __builtin_popcountl(i) // TODO __builtin_popcountg
#  endif
#endif
#ifndef popcount
#    define popcount(i) explicitPopcount(i)
//popcount for machines without it
constexpr char explicitPopcount(uint32_t i)
{
#warning NOT USING BUILTIN POPCOUNT
    //Magic code that compiles to popcnt https://stackoverflow.com/questions/109023/count-the-number-of-set-bits-in-a-32-bit-integer
    i = i - ((i >> 1) & 0x55555555);        // add pairs of bits
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);  // quads
    i = (i + (i >> 4)) & 0x0F0F0F0F;        // groups of 8
    i *= 0x01010101;                        // horizontal sum of bytes
    return  i >> 24;               // return just that top byte (after truncating to 32-bit even when int is wider than uint64_t)
}
#endif


// template <class derived>
class compressor;

template<typename dataType>
class Matrix;

template<typename dataType>
class vector;

template<typename derived, Eigen::StorageOptions storageOrder = Eigen::RowMajor>
class vectorView
{
    static_assert(storageOrder == Eigen::RowMajor || storageOrder == Eigen::ColMajor);
    // vectorView(const vectorView &other) = delete;
    // vectorView& operator= (const vectorView& other) = delete;
    typedef typename derived::static_type dataType;
    derived* m_derived;
    size_t m_otherIndexOffset;

    template<typename,Eigen::StorageOptions>
    friend class vectorView;
    bool m_allowResize = false;


public:
    vectorView(size_t otherIndexOffset, derived* derivedObj, bool allowResize)
    {
        m_otherIndexOffset = otherIndexOffset;
        m_derived = derivedObj;
        m_allowResize = allowResize;
    }
    typedef Eigen::Vector<dataType,Eigen::Dynamic> EigenVector;

    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    dataType& operator[](int size)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        return at(size);
    }
    dataType operator[](int size)const{return at(size);}
    operator EigenVector () const
    {
        size_t mySize = size();

        EigenVector ret(mySize);
        for (size_t i = 0; i < mySize; i++)
            ret[i] = (*this)[i];
        return ret;
    }
    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    dataType& at(size_t idx)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        assert(idx < size());
        if constexpr (storageOrder == Eigen::RowMajor)
            return m_derived->at(m_otherIndexOffset,idx);
        if constexpr (storageOrder == Eigen::ColMajor)
            return m_derived->at(idx,m_otherIndexOffset);
    }

    dataType at (size_t idx) const
    {
        assert(idx < size());
        if constexpr (storageOrder == Eigen::RowMajor)
            return m_derived->at(m_otherIndexOffset,idx);
        if constexpr (storageOrder == Eigen::ColMajor)
            return m_derived->at(idx,m_otherIndexOffset);
    }


    size_t size() const {return storageOrder == Eigen::RowMajor ? m_derived->m_jSize : m_derived->m_iSize;}
    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        if (!m_allowResize && size != this->size())
        {
            // if this is true then it is not actually a resize. Lots of places call resize in order to set compressors and whatnot. Basically we need to prevent reallocation
            fprintf(stderr,"Cannot resize this vector view");
            __builtin_trap();
        }

        if constexpr (storageOrder == Eigen::RowMajor)
            return m_derived->resize(m_derived->m_iSize,size,isCompressed,compressor);
        if constexpr (storageOrder == Eigen::ColMajor)
            return m_derived->resize(size,m_derived->m_jSize,size,isCompressed,compressor);
    }
    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor,bool) // no memset 0
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        if (!m_allowResize && size != this->size())
        {
            // if this is true then it is not actually a resize. Lots of places call resize in order to set compressors and whatnot. Basically we need to prevent reallocation
            fprintf(stderr,"Cannot resize this vector view");
            __builtin_trap();
        }

        if constexpr (storageOrder == Eigen::RowMajor)
            return m_derived->resize(m_derived->m_iSize,size,isCompressed,compressor,false);
        if constexpr (storageOrder == Eigen::ColMajor)
            return m_derived->resize(size,m_derived->m_jSize,size,isCompressed,compressor,false);
    }
    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    void copy(const vector<typename derived::static_type>& other)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        copy(other.getView());
    }

    template<typename otherDerived, Eigen::StorageOptions otherstorageOrder, typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    void copy(const vectorView<otherDerived,otherstorageOrder>& other)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        if (size() != other.size())
        {
            if (m_allowResize)
            {
                std::shared_ptr<compressor> comp;
                bool isComp = other.getIsCompressed(comp);
                resize(other.size(),isComp,comp);
            }
            else
            {
                fprintf(stderr,"cannot copy from a vector of a different size into a vectorView without allowing resize");
                __builtin_trap();
            }
        }


        for (size_t idx = 0; idx < size(); idx++)
        {
            (*this)[idx] = other[idx];
        }
    }
    template<typename otherDerived, Eigen::StorageOptions otherstorageOrder>
    bool isSame(const vectorView<otherDerived,otherstorageOrder>& other) const
    {
        if (storageOrder != otherstorageOrder)
            return false;
        if (!std::is_same_v<std::remove_cv_t<derived>,std::remove_cv_t<otherDerived>>)
            return false;
        if (m_derived == other.m_derived && m_otherIndexOffset == other.m_otherIndexOffset)
            return true;
        return false;
    }

    template<typename newDerived>
    operator const vectorView<newDerived,storageOrder> ()
    {
        static_assert(std::is_same_v<newDerived,derived> || std::is_same_v<newDerived,std::add_const_t<derived>>, "Can only add const not remove it");
        return vectorView<newDerived,storageOrder>(m_otherIndexOffset, m_derived,m_allowResize);
    }
    bool getIsCompressed(std::shared_ptr<compressor>& ptr) const {return m_derived->getIsCompressed(ptr);}
    template <typename d=derived, std::enable_if_t<!std::is_const_v<d>,bool> = 1>
    void setIsCompressed(bool val, std::shared_ptr<compressor> compressor)
    {
        static_assert(std::is_same_v<d,derived>, "Dont overwrite the default tempalte parameter in sfinae");
        m_derived->setIsCompressed(val,compressor);
    }

    template<Eigen::StorageOptions otherOrder>
    realNumType dot(const vectorView<derived,otherOrder> &other) const
    {
        assert(size() == other.size());

        realNumType ret = 0;
        for (size_t i = 0; i < other.size(); i++)
        {
            if constexpr(std::is_same_v<typename derived::static_type,realNumType>)
                ret += (*this)[i] * other[i];
            else
                ret += std::real(std::conj((*this)[i]) * other[i]);
        }
        return ret;
    }
    template<Eigen::StorageOptions otherOrder>
    std::complex<realNumType> cdot(const vectorView<derived,otherOrder> &other) const //always returns a complex value.
    {
        assert(size() == other.size());

        std::complex<realNumType> ret = 0;
        for (size_t i = 0; i < other.size(); i++)
            ret += std::conj((*this)[i]) * other[i];
        return ret;
    }
    template<Eigen::StorageOptions otherOrder>
    vectorView operator += (const vectorView<std::add_const_t<derived>,otherOrder>& other)
    {
        assert(size() == other.size());
        for (size_t i = 0; i < size(); i++)
        {
            (*this)[i] += other[i];
        }
        return *this;
    }

};

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
    typedef dataType static_type;
    size_t m_iSize;
    size_t m_jSize;

    Matrix():Matrix(0,0){};
    Matrix(size_t iSize, size_t jSize);
    Matrix(int value, const std::vector<uint64_t>& iIndex, const std::vector<uint64_t>& jIndex,int size);
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
    inline dataType& operator () (size_t i, size_t j){return at(i,j);}
    inline const dataType& operator () (size_t i, size_t j)const{return at(i,j);}

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
    //Returns a vector that has Indices i. i.e. getIVector(j)[i] = (*this)(i,j)
    vectorView<Matrix,Eigen::ColMajor> getIVectorView(size_t j){return vectorView<Matrix,Eigen::ColMajor>(j,this,false);}
    //Returns a vector that has Indices j. i.e. getJVector(i)[j] = (*this)(i,j)
    vectorView<Matrix,Eigen::RowMajor> getJVectorView(size_t i){return vectorView<Matrix,Eigen::RowMajor>(i,this,false);}
    vectorView<const Matrix,Eigen::ColMajor> getIVectorView(size_t j) const {return vectorView<const Matrix,Eigen::ColMajor>(j,this,false);}
    //Returns a vector that has Indices j. i.e. getJVector(i)[j] = (*this)(i,j)
    vectorView<const Matrix,Eigen::RowMajor> getJVectorView(size_t i) const {return vectorView<const Matrix,Eigen::RowMajor>(i,this,false);}
    Matrix<realNumType> real();
};

template<>    Matrix<double> &Matrix<double>::addEquals(const Matrix &otherA, const Matrix &otherB);
template<>    void Matrix<double>::mul(double scalar, Matrix<double> &dest) const;

template<typename dataType>
class vector : public Matrix<dataType>
{
    vector(const vector &other) = delete;
    vector& operator= (const vector& other) = delete;

public:
    typedef Eigen::Vector<dataType,Eigen::Dynamic> EigenVector;
    vector():vector(0){}
    vector(int size):Matrix<dataType>(1,size){}
    vector(const EigenVector& vec);
    vector(const std::vector<dataType>& data);

    vector(const Matrix<dataType>& other) = delete;
    vector(dataType* buffer, size_t size) : Matrix<dataType>(buffer,size,1){}

    vector(Matrix<dataType>&& other) noexcept : Matrix<dataType>(std::move(other))
    {
        assert(this->m_iSize == 1);
    }
    vector(vector&& other) noexcept: Matrix<dataType>(std::move(other))
    {
        assert(this->m_iSize == 1);
    }






    dataType& operator[](int size){assert(this->m_iSize == 1 && this->at(0,size) == this->m_data[size]); return this->m_data[size];}
    const dataType& operator[](int size)const{assert(this->m_iSize == 1 && this->at(0,size) == this->m_data[size]); return this->m_data[size];}
    operator EigenVector () const;

    realNumType dot(const vector &other) const;
    std::complex<realNumType> cdot(const vector &other) const; //always returns a complex value.
    void partialScalarMul(size_t startIndex, size_t endIndex, dataType scalar);
    //vector& operator *= (const vector &other);
    size_t size() const {return this->m_jSize;}
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor)
    {this->Matrix<dataType>::resize(1,size,isCompressed,compressor);}
    void resize(size_t size,bool isCompressed, std::shared_ptr<compressor> compressor,bool)
    {this->Matrix<dataType>::resize(1,size,isCompressed,compressor,true);} // no memset 0
    vector& copyFromBuffer(dataType* buffer, size_t size) {this->Matrix<dataType>::copyFromBuffer(buffer,1,size); return *this;}
    void conj();


    const dataType* begin() const {return this->m_data;}
    const dataType* end() const {return this->m_data + size();}

    operator vectorView<Matrix<dataType>,Eigen::RowMajor> () {return vectorView<Matrix<dataType>,Eigen::RowMajor>(0,(Matrix<dataType>*)this,true);}
    operator vectorView<Matrix<dataType>,Eigen::RowMajor> ()const {return vectorView<Matrix<dataType>,Eigen::RowMajor>(0,(Matrix<dataType>*)this,false);}
    operator vectorView<const Matrix<dataType>> () const {return this->getJVectorView(0);}
    vectorView<Matrix<dataType>,Eigen::RowMajor> getView(){return *this;}
    vectorView<const Matrix<dataType>,Eigen::RowMajor> getView()const {return *this;}
    vector<realNumType> real();
    void normalize();


};



//Convert from the represenation used in Ansatz to the Eigen format. Not sure where to put this TODO
Matrix<numType>::EigenMatrix convert(const std::vector<vector<numType>>& AnsatzTangentSpace);

extern template class Matrix<std::complex<realNumType>>;
extern template class Matrix<realNumType>;
extern template class vector<std::complex<realNumType>>;
extern template class vector<realNumType>;

extern template class vectorView<Matrix<std::complex<realNumType>>>;
extern template class vectorView<Matrix<realNumType>>;
extern template class vectorView<const Matrix<std::complex<realNumType>>>;
extern template class vectorView<const Matrix<realNumType>>;

extern template class vectorView<Matrix<std::complex<realNumType>>, Eigen::ColMajor>;
extern template class vectorView<Matrix<realNumType>, Eigen::ColMajor>;
extern template class vectorView<const Matrix<std::complex<realNumType>>, Eigen::ColMajor>;
extern template class vectorView<const Matrix<realNumType>, Eigen::ColMajor>;



#endif // LINALG_H
