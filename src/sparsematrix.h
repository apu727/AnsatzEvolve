/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H
#include "globals.h"
#include "hamiltonianmatrix.h"
#include "linalg.h"

#include <Eigen/Sparse>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>



//typedef bool (*compareType)(const std::pair<size_t,size_t>&, const std::pair<size_t,size_t> &);

template<typename dataType, typename vectorType>
class targetMatrix
{
protected:
    size_t m_blankCount = 0;
public:
    virtual ~targetMatrix(){};

    virtual void multiply(const vectorView<const Matrix<vectorType>> &other, vectorView<Matrix<vectorType>> dest) const = 0;

    friend void mul (const targetMatrix<dataType,vectorType>& lhs, const vector<vectorType>& other, vector<vectorType>& dest)
    {
        lhs.multiply(other,dest); // allows for virtual lookups while remaining compatible
    }

    dataType braket(const vector<vectorType> &lhs, const vector<vectorType> &rhs, vector<vectorType> *tempStorage = nullptr) const
    {
        bool toDelete = false;
        if (tempStorage == nullptr)
        {
            tempStorage = new vector<vectorType>;
            toDelete = true;
        }
        this->multiply(rhs,*tempStorage);
        dataType ret = lhs.dot(*tempStorage);
        if (toDelete)
            delete tempStorage;

        return ret;
    }
    friend dataType braket (const vector<dataType>& lhs, const targetMatrix<dataType,vectorType>& middle, const vector<dataType>& rhs, vector<dataType>* tempStorage = nullptr)
    {
        return middle.braket(lhs,rhs,tempStorage); // allows for virtual lookups while remaining compatible
    }

    virtual bool addBlank() = 0;
    virtual size_t getBlankCount() const = 0;
    virtual void unblankAll() = 0;
    virtual void blankAllButOne() = 0;
    virtual size_t getMaxBlankCount() const = 0;
};
//template <class derived> // avoids virtual dispatch, Eigen style
class compressor
{
protected:
    // std::vector<int64_t> compressPerm;
    std::vector<int64_t> decompressPerm;
public:
    virtual bool compressIndex(uint64_t index, uint64_t& compressedIdx) = 0;

    bool deCompressIndex(uint64_t index, uint64_t& decompressedIdx)
    {
        assert(index < decompressPerm.size());
        decompressedIdx = decompressPerm[index];
        return true;
    }
    template<typename VectorType>
    static void compressVector(const vectorView<const Matrix<VectorType>,Eigen::RowMajor>& src, vectorView<Matrix<VectorType>,Eigen::RowMajor> dst,
                               std::shared_ptr<compressor> thisPtr)
    {
        std::shared_ptr<compressor> srcCompressor;
        if (src.getIsCompressed(srcCompressor)) // this comes up often enough that its better to special case it than throw an error
        {
            if (srcCompressor.get() == thisPtr.get())
            {
                dst.copy(src);
                return;
            }
            __builtin_trap();
        }

        dst.resize(thisPtr->decompressPerm.size(),true,thisPtr,false);
        for (size_t i = 0; i < src.size(); i++)
        {
            uint64_t newIdx;
            if (thisPtr->compressIndex(i,newIdx))
                dst[newIdx] = src[i];
        }
    }

    template<typename VectorType>
    static void compressVector(std::vector<VectorType> coeffs, std::vector<uint64_t> indexes, vectorView<Matrix<VectorType>,Eigen::RowMajor> dst,
                               std::shared_ptr<compressor> thisPtr)
    {
        assert(indexes.size() == coeffs.size());
        std::shared_ptr<compressor> srcCompressor;

        dst.resize(thisPtr->decompressPerm.size(),true,thisPtr);
        for (size_t i = 0; i < coeffs.size(); i++)
        {
            uint64_t newIdx;
            if (thisPtr->compressIndex(indexes[i],newIdx))
                dst[newIdx] = coeffs[i];
        }
    }

    template<typename VectorType>
    static void deCompressVector(const vectorView<const Matrix<VectorType>>& src, vectorView<Matrix<VectorType>,Eigen::RowMajor> dst, std::shared_ptr<compressor> thisPtr)
    {
        std::shared_ptr<compressor> srcCompressor;
        assert(src.getIsCompressed(srcCompressor) == true); // decompressing a decompressed vector is an error. This is to catch bugs
        assert(srcCompressor.get() == thisPtr.get());

        dst.resize(thisPtr->getUnCompressedSize(),false,thisPtr);
        for (size_t i = 0; i < thisPtr->decompressPerm.size(); i++)
        {
            int64_t newIdx = thisPtr->decompressPerm[i];
            dst[newIdx] = src[i];
        }
    }
    compressor(){}
    virtual ~compressor(){};

    size_t getCompressedSize(){return decompressPerm.size();}
    virtual size_t getUnCompressedSize() = 0;
    virtual void dummyImplement() = 0; // To make this an abstract base class. Derived class needs to implement the construction of compressPerm and decompressPerm
    virtual bool opDoesSomething(excOp& op) = 0;

    enum class registeredBaseClasses // only needed if you wish to be serialisable
    {
        numberOperatorCompressor = 1,
        SZAndnumberOperatorCompressor = 2,
    };
    virtual registeredBaseClasses getSerialisationType() = 0;
    virtual serialDataContainer serialise() // derived must serialise this object first, the returned bytes must be stored first in the serial stream
    {
        int type = static_cast<int>(getSerialisationType());
        std::shared_ptr<char[]> ret (new char[sizeof(type)]);
        std::memcpy(ret.get(),&type,sizeof(type));
        return {.ptr = ret, .size = sizeof(type), .alignment = alignof(char)};
    }
    static size_t deserialise(char *ptr,std::shared_ptr<compressor>& dest);// derived class must override, UB if not, probably crash.
};
template<typename vectorType>
inline bool s_loadMatrix(sparseMatrix<std::complex<realNumType>,vectorType>* me,std::string filePath);
template<typename vectorType>
inline bool s_loadMatrix(sparseMatrix<realNumType,vectorType>* me,std::string filePath);
template<typename dataType, typename vectorType>
bool s_loadOneAndTwoElectronsIntegrals(sparseMatrix<dataType,vectorType>* me,std::string filePath,size_t numberOfQubits, std::shared_ptr<compressor> comp);

template<typename dataType, typename vectorType>
class sparseMatrix : public targetMatrix<dataType,vectorType>
{

    std::vector<uint64_t> m_iIndexes;
    std::vector<uint64_t> m_jIndexes;
    std::vector<dataType> m_data;
    dataType def = dataType();
    uint64_t m_iSize = 0;
    uint64_t m_jSize = 0;
    std::vector<uint64_t> m_blankingVector; // vector that multiplies by 1 or 0 depending of if the column/row is allowed.

    friend bool s_loadMatrix <>(sparseMatrix<dataType,vectorType>* me,std::string filePath);
    friend bool s_loadOneAndTwoElectronsIntegrals <>(sparseMatrix<dataType,vectorType>* me,std::string filePath,size_t numberOfQubits, std::shared_ptr<compressor> comp);
    std::shared_ptr<compressor> m_compressor;
    bool m_isCompressed = false;
    bool m_isRotationGenerator = false;// allows certain optimisations

    void multiplyDecompressed(const vectorView<const Matrix<vectorType>>& other, vectorView<Matrix<vectorType>> dest) const;
    void rotateDecompressed(realNumType S,realNumType C, const vectorView<const Matrix<vectorType>>& other, vectorView<Matrix<vectorType>> dest) const;
public:
    sparseMatrix();
    sparseMatrix(const std::vector<dataType>& value, const std::vector<int>& iIndex, const std::vector<int>& jIndex,int);
    sparseMatrix(dataType* value, uint64_t* iIndex, uint64_t* jIndex,size_t N,
                 std::shared_ptr<compressor> compressor, bool isRotationGenerator = false);
    sparseMatrix(dataType* value, uint64_t* iIndex, uint64_t* jIndex,size_t N);
    sparseMatrix(const Matrix<dataType>& other);

    virtual ~sparseMatrix(){};

    sparseMatrix& operator*= (dataType scalar);

    sparseMatrix(const sparseMatrix& other) = delete;
    sparseMatrix& operator= (const sparseMatrix& other) = delete;
    sparseMatrix& copy (const sparseMatrix& other);
    sparseMatrix(sparseMatrix&& other);
    sparseMatrix& operator= (sparseMatrix&& other);

    bool allClose(dataType val,realNumType atol=1e-10);
    const dataType* at(uint64_t i, uint64_t j, bool& success) const;

    dataType* at(uint64_t i, uint64_t j, bool& success){return const_cast<dataType*>(static_cast<const sparseMatrix*>(this)->at(i,j,success));}
    const dataType* at(uint64_t i, uint64_t j) const{bool temp; return this->at(i,j,temp);}
    dataType* at(uint64_t i, uint64_t j){bool temp; return this->at(i,j,temp);}

    virtual bool addBlank() override;
    virtual size_t getBlankCount() const override {return this->m_blankCount;}
    virtual void unblankAll() override;
    virtual void blankAllButOne() override{this->m_blankCount = 1;}
    virtual size_t getMaxBlankCount() const override {return m_iSize;}
    void setBlankingVector(const std::vector<uint64_t> &v){m_blankingVector = v;}
    std::vector<uint64_t> &getBlankingVector(){return m_blankingVector;}
    virtual bool loadMatrix(std::string filename,size_t numberOfQubits,std::shared_ptr<compressor> comp);
    bool dumpMatrix(const std::string& filePath);

    void multiply(const vectorView<const Matrix<vectorType>>& other, vectorView<Matrix<vectorType>> dest) const override;
    void rotate(realNumType angle, const vectorView<const Matrix<vectorType>>& other, vectorView<Matrix<vectorType>> dest) const;
    void rotate(realNumType S,realNumType C, const vectorView<const Matrix<vectorType>>& other, vectorView<Matrix<vectorType>> dest) const;
    void rotateAndBraketWithTangentOfResult(realNumType S,realNumType C, const vectorView<const Matrix<vectorType>>& other,vectorView<Matrix<vectorType>> dest,
                                    const vectorView<const Matrix<vectorType>>& toBraket, realNumType& result) const;
    void compress(std::shared_ptr<compressor> comp);
    void decompress();




    auto begin() const {return m_data.cbegin();}
    auto end() const {return m_data.cend();}
    auto begin() {return m_data.begin();}
    auto end() {return m_data.end();}

    auto iItBegin() const {return m_iIndexes.cbegin();}
    auto jItBegin() const {return m_jIndexes.cbegin();}
    auto iItEnd() const {return m_iIndexes.cend();}
    auto jItEnd() const {return m_jIndexes.cend();}

    size_t size() const{return m_jIndexes.size();}
    uint64_t getiSize(){return m_iSize;}
    uint64_t getjSize(){return m_jSize;}

    typedef Eigen::SparseMatrix<dataType,Eigen::RowMajor> EigenSparseMatrix;

    operator EigenSparseMatrix () const;


};
template<typename dataType>
class projectionMatrix : public targetMatrix<dataType,dataType>
{//Encodes a target matrix as a representation of basis vectors. Computes projections efficiently. even when not diagonal
//Also allows for blanking by calling .blank()
    std::vector<vector<dataType>> m_basisVectors;
public:
    projectionMatrix(std::vector<vector<dataType>>&& basisVectors);
    virtual ~projectionMatrix(){};

    virtual bool addBlank() override;
    virtual size_t getBlankCount()const override {return this->m_blankCount;};
    virtual void unblankAll()override{this->m_blankCount = m_basisVectors.size();};
    virtual void blankAllButOne()override{this->m_blankCount = 1;};
    virtual size_t getMaxBlankCount() const override {return m_basisVectors.size();};

    void multiply(const vectorView<const Matrix<dataType>>& other, vectorView<Matrix<dataType>> dest) const override;

    const vector<dataType>& getTargetVector()const{return m_basisVectors[this->m_blankCount-1];}
};



#endif // SPARSEMATRIX_H
