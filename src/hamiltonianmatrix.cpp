/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "hamiltonianmatrix.h"
#include "logger.h"
#include "threadpool.h"
#include <chrono>
static size_t choose(size_t n, size_t k)
{
    if (k == 0)
        return 1;
    return (n * choose(n - 1, k - 1)) / k;
}

template<typename dataType>
inline bool s_loadMatrix(Eigen::SparseMatrix<dataType, Eigen::ColMajor>& destMat,std::string filePath, std::shared_ptr<compressor> comp, size_t linearSize)
{
    //works for loading from some random sparse matrix format. Index file is 1 Indexed!!
    FILE *fpCoeff;

    fpCoeff = fopen((filePath+"_Ham_Coeff.dat").c_str(), "r");
    if(NULL == fpCoeff)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }

    FILE *fpIndex;

    fpIndex = fopen((filePath+"_Ham_Index.dat").c_str(), "r");
    if(NULL == fpIndex)
    {
        fclose(fpCoeff);
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }
    //There is no way to know the code so this will have to do.
    double coeffReal = 0;
    double coeffImag = 0;

    uint32_t idxs[2] = {};

    int ret = fscanf(fpCoeff, "%lg,  %lg  \n", &coeffReal, &coeffImag);
    int ret2 = fscanf(fpIndex, "%u %u\n",&(idxs[0]),&(idxs[1]));

    std::vector<Eigen::Triplet<dataType>> tripletList;
    // tripletList.reserve(m_data.size()); //TODO estimate for size
    while(EOF != ret && EOF != ret2)
    {
        // fprintf(stderr,"Read Coeff: %lf \n ", coeff);
        // fprintf(stderr,"Read Index: %u,%u \n ", idxs[0]-1,idxs[1]-1);
        uint32_t compi = idxs[0]-1;
        uint32_t compj = idxs[1]-1;
        if (comp)
        {
            comp->compressIndex(compi,compi);
            comp->compressIndex(compj,compj);
        }
        if (compi != (uint32_t)-1 && compj != (uint32_t)-1)
        {
            if constexpr (std::is_same_v<dataType,typename Eigen::NumTraits<dataType>::Real>)
                tripletList.push_back(Eigen::Triplet<dataType>(compi,compj,coeffReal));
            else
                tripletList.push_back(Eigen::Triplet<dataType>(compi,compj,dataType(coeffReal,coeffImag)));
        }
        ret = fscanf(fpCoeff, realNumTypeCode ", " realNumTypeCode " \n", &coeffReal, &coeffImag);
        ret2 = fscanf(fpIndex, "%u %u\n",&(idxs[0]),&(idxs[1]));
    }
    destMat.resize(linearSize,linearSize);
    destMat.setFromTriplets(tripletList.begin(),tripletList.end());
    logger().log("Full sparse size", tripletList.size());
    fclose(fpIndex);
    fclose(fpCoeff);
    return 1;
}


static unsigned long ReadFile(FILE *fp, unsigned char *Buffer, unsigned long BufferSize)
{
    return(fread(Buffer, 1, BufferSize, fp));
}

static size_t CalculateFileSize(FILE *fp)
{
    size_t size;
    fseek (fp,0,SEEK_END);
    size= ftell (fp);
    fseek (fp,0,SEEK_SET);
    if (size != (size_t)-1)
    {
        return size;
    }
    else
        return 0;
}

template<typename dataType>
bool s_loadOneAndTwoElectronsIntegrals_Check(Eigen::SparseMatrix<dataType, Eigen::ColMajor> __attribute__ ((unused))CheckWith, std::string filePath,size_t numberOfQubits, std::shared_ptr<compressor> comp)
{
    FILE *fponeEInts;

    fponeEInts = fopen((filePath+"_oneEInts.bin").c_str(), "rb");
    if(!fponeEInts)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath + +"_oneEInts.bin").c_str());
        return 0;
    }
    FILE *fptwoEInts;
    fptwoEInts = fopen((filePath+"_twoEInts.bin").c_str(), "rb");
    if(!fptwoEInts)
    {
        fclose(fponeEInts);
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath + +"_twoEInts.bin").c_str());
        return 0;
    }
    size_t oneEIntsBufferSize = CalculateFileSize(fponeEInts);//Calculate total size of file
    if (oneEIntsBufferSize != (numberOfQubits/2) * (numberOfQubits/2)*sizeof(double))
    {
        fprintf(stderr,"oneEInts Binary file has wrong amount of data. Hamiltonians should be in Molecular orbital format and not adapted for permutational symmetry. i.e. S1\n");
        fclose(fponeEInts);
        fclose(fptwoEInts);
        return false;
    }
    size_t twoEIntsBufferSize = CalculateFileSize(fptwoEInts);//Calculate total size of file
    if (twoEIntsBufferSize != (numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*sizeof(double))
    {
        fprintf(stderr,"twoEInts Binary file has wrong amount of data. Hamiltonians should be in Molecular orbital format and not adapted for permutational symmetry. i.e. S1\n");
        fclose(fponeEInts);
        fclose(fptwoEInts);
        return false;
    }

    Eigen::Matrix<double,-1,-1,Eigen::RowMajor> oneEInts((numberOfQubits/2),(numberOfQubits/2));
    double* twoEInts = new double[(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)];

    unsigned long __attribute__ ((unused))RetValue = ReadFile(fponeEInts, (unsigned char*)oneEInts.data(), oneEIntsBufferSize);
    assert(RetValue == oneEIntsBufferSize);
    RetValue = ReadFile(fptwoEInts, (unsigned char*)twoEInts, twoEIntsBufferSize);
    assert(RetValue == twoEIntsBufferSize);

    auto twoEIntsLookup = [numberOfQubits](size_t i, size_t j, size_t k, size_t l)
    {
        size_t numAO = numberOfQubits/2;
        // return (((i*numAO)+j)*numAO + k)*numAO + l;
        return (((i*numAO)+l)*numAO + j)*numAO + k; // pyscf ordering, (il|jk) = <ij|kl>
    };
    // auto getSpatialFromSpin = [numberOfQubits](size_t i){return i % (numberOfQubits/2);};

    auto getTwoElectronEnergy = [&twoEIntsLookup,twoEInts](const std::pair<size_t,bool> (&idxs)[4])
    {//Note the conventions on destroying in the same order you created.
        realNumType Energy = 0;
        if (idxs[0].second == idxs[2].second && idxs[1].second == idxs[3].second)
        {
            size_t twoEIntIdx = twoEIntsLookup(idxs[1].first,idxs[0].first,idxs[2].first,idxs[3].first);
            Energy += twoEInts[twoEIntIdx];
        }
        if (idxs[0].second == idxs[3].second && idxs[1].second == idxs[2].second)
        {
            size_t twoEIntIdx = twoEIntsLookup(idxs[1].first,idxs[0].first,idxs[3].first,idxs[2].first);
            Energy -= twoEInts[twoEIntIdx];
        }
        return Energy;
    };

    auto getFockMatrixElem = [&getTwoElectronEnergy,&oneEInts,numberOfQubits]
        (std::pair<size_t,bool> (&idxs)[4], size_t jBasisState)
    {
        assert(idxs[0] != idxs[2]);
        size_t trueIdx0 = idxs[0].first + (idxs[0].second ? numberOfQubits/2 : 0);
        size_t trueIdx2 = idxs[2].first + (idxs[2].second ? numberOfQubits/2 : 0);
        realNumType Energy = 0;
        Energy += oneEInts(idxs[0].first,idxs[2].first);
        for (size_t k = 0; k < numberOfQubits; k++)
        {
            std::pair kIdx = {k % (numberOfQubits/2),k >= (numberOfQubits/2)};
            if (jBasisState & (1<<k))
            {
                if (k != trueIdx0 && k != trueIdx2)
                    Energy += getTwoElectronEnergy({idxs[0],kIdx,idxs[2],kIdx});
            }
        }
        return Energy;
    };

    auto getEnergy = [numberOfQubits,&oneEInts,&getTwoElectronEnergy](size_t jBasisState)
    {
        realNumType Energy = 0;
        for (size_t k = 0; k < numberOfQubits; k++)
        {
            if (!(jBasisState & (1<<k)))
                continue;
            std::pair kIdx = {k % (numberOfQubits/2),k >= (numberOfQubits/2)};
            //h_ii
            Energy += oneEInts(kIdx.first,kIdx.first);

            for (size_t l = k+1; l < numberOfQubits; l++)
            {
                if (!(jBasisState & (1<<l)))
                    continue;
                std::pair lIdx = {l % (numberOfQubits/2),l >= (numberOfQubits/2)};
                Energy += getTwoElectronEnergy({kIdx,lIdx,kIdx,lIdx});
            }
        }
        return Energy;
    };

    auto getElement = [numberOfQubits,&getEnergy,&getTwoElectronEnergy,getFockMatrixElem](uint32_t iBasisState, uint32_t jBasisState)
    {
        std::pair<size_t,bool> idxs[4];// a^\dagger a^\dagger a a
        int8_t annihilatePos = 2;
        int8_t createPos = 0;
        bool eveniElecSoFar = 0;
        bool evenjElecSoFar = 0;

        bool sign = true; //True => positive
        realNumType Energy = 0;

        for (size_t k = 0; k < numberOfQubits; k++)
        {
            bool isSet = false;
            bool jsSet = false;
            if (iBasisState & (1<< k))
            {
                isSet = true;
                eveniElecSoFar = !eveniElecSoFar;
            }
            if (jBasisState & (1<<k))
            {
                jsSet = true;
                evenjElecSoFar = !evenjElecSoFar;
            }
            if (isSet == jsSet)
                continue;
            if (isSet)
            {
                if (createPos > 1)
                    __builtin_trap();
                idxs[createPos++] = {k % (numberOfQubits/2),k >= (numberOfQubits/2)};
                if (eveniElecSoFar)
                    sign = !sign;
            }
            if (jsSet)
            {
                if (annihilatePos > 3)
                    __builtin_trap();
                idxs[annihilatePos++] = {k % (numberOfQubits/2),k >= (numberOfQubits/2)};
                if (evenjElecSoFar)
                    sign = !sign;
            }
        }

        if(annihilatePos == 2 && createPos == 0)
        {
            Energy = getEnergy(jBasisState);
        }
        else if (annihilatePos == 3 && createPos == 1)
        {
            Energy = (sign ? 1 : -1)*getFockMatrixElem(idxs,jBasisState);
        }
        else if (annihilatePos == 4 && createPos == 2)
        {
            Energy = (sign ? 1 : -1)*getTwoElectronEnergy(idxs);
        }
        else
        {
            logger().log("Not handled case construct Ham");
            __builtin_trap();
        }
        return Energy;
    };



    //only construct the compressed elements, but in the decompressed format
    size_t compressedSize;
    if (comp)
    {
        compressedSize = comp->getCompressedSize();
        logger().log("Matrix linear Size",compressedSize);
    }
    else
    {
        compressedSize = 1<<numberOfQubits;
    }

    for (size_t i = 0; i < compressedSize; i++)
    {
        uint32_t jBasisState;
        if (comp)
            comp->deCompressIndex(i,jBasisState);
        else
            jBasisState = i;
        assert(numberOfQubits < 256);
        for (std::uint_fast8_t a = 0; a < numberOfQubits; a++)
        {
            if(jBasisState & (1<<a))
                continue;
            for (std::uint_fast8_t b = a+1; b < numberOfQubits; b++)
            {
                if(jBasisState & (1<<b))
                    continue;
                for (std::uint_fast8_t c = 0; c < numberOfQubits; c++)
                {
                    if (!(jBasisState & (1<<c)))
                        continue;
                    if (c == a || c == b)
                        continue;
                    for (std::uint_fast8_t d = c+1; d < numberOfQubits; d++)
                    {
                        if (d == b || d == a)
                            continue;
                        if (!(jBasisState & (1<<d)))
                            continue;

                        uint32_t iBasisState = ((1<<a) | (1<<b)) ^ ((1<<c | 1<< d) ^ jBasisState);
                        if (iBasisState < jBasisState)
                            continue;
                        assert(iBasisState != jBasisState);
                        if (comp)
                        {
                            uint32_t temp;
                            comp->compressIndex(iBasisState,temp);
                            if (temp == (uint32_t)-1)
                                continue;
                        }

                        realNumType __attribute__ ((unused))Energy = getElement(iBasisState,jBasisState);
                        uint32_t compi = iBasisState;
                        uint32_t compj = jBasisState;
                        if (comp)
                        {
                            comp->compressIndex(compi,compi);
                            comp->compressIndex(compj,compj);
                        }

                        assert(std::abs(CheckWith.coeff(compi,compj) - Energy) < 1e-13);
                        assert(std::abs(CheckWith.coeff(compj,compi) - Energy) < 1e-13);

                    }
                }
            }
        }

        for (std::uint_fast8_t a = 0; a < numberOfQubits; a++)
        {
            if ((jBasisState & (1<<a)))
                continue;
            for (std::uint_fast8_t c = 0; c < numberOfQubits; c++)
            {
                if (c == a)
                    continue;
                if (!(jBasisState & (1<<c)))
                    continue;

                uint32_t iBasisState = ((1<<a)) ^ ((1<<c) ^ jBasisState);
                if (iBasisState < jBasisState)
                    continue;
                assert(iBasisState != jBasisState);
                if (comp)
                {
                    uint32_t temp;
                    comp->compressIndex(iBasisState,temp);
                    if (temp == (uint32_t)-1)
                        continue;
                }

                realNumType __attribute__ ((unused))Energy = getElement(iBasisState,jBasisState);
                uint32_t compi = iBasisState;
                uint32_t compj = jBasisState;
                if (comp)
                {
                    comp->compressIndex(compi,compi);
                    comp->compressIndex(compj,compj);
                }

                assert(std::abs(CheckWith.coeff(compi,compj) - Energy) < 1e-13);
                assert(std::abs(CheckWith.coeff(compj,compi) - Energy) < 1e-13);
            }

        }

        realNumType __attribute__ ((unused))Energy = getElement(jBasisState,jBasisState);
        uint32_t compj = jBasisState;
        if (comp)
            comp->compressIndex(compj,compj);

        assert(std::abs(CheckWith.coeff(compj,compj) - Energy) < 1e-13);
    }

    delete[] twoEInts;
    fclose(fponeEInts);
    fclose(fptwoEInts);
    return true;
}

template<typename dataType>
bool s_loadOneAndTwoElectronsIntegrals(std::vector<excOp>& operators,
                                       std::vector<dataType>& vals,const std::string& filePath, const size_t numberOfQubits)
{
    FILE *fponeEInts;

    fponeEInts = fopen((filePath+"_oneEInts.bin").c_str(), "rb");
    if(!fponeEInts)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath + +"_oneEInts.bin").c_str());
        return 0;
    }
    FILE *fptwoEInts;
    fptwoEInts = fopen((filePath+"_twoEInts.bin").c_str(), "rb");
    if(!fponeEInts)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath + +"_twoEInts.bin").c_str());
        return 0;
    }
    size_t oneEIntsBufferSize = CalculateFileSize(fponeEInts);//Calculate total size of file
    if (oneEIntsBufferSize != (numberOfQubits/2) * (numberOfQubits/2)*sizeof(double))
    {
        fprintf(stderr,"oneEInts Binary file has wrong amount of data. Hamiltonians should be in Molecular orbital format");
        fclose(fponeEInts);
        fclose(fptwoEInts);
        return false;
    }
    size_t twoEIntsBufferSize = CalculateFileSize(fptwoEInts);//Calculate total size of file
    if (twoEIntsBufferSize != (numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*sizeof(double))
    {
        fprintf(stderr,"twoEInts Binary file has wrong amount of data. Hamiltonians should be in Molecular orbital format");
        fclose(fponeEInts);
        fclose(fptwoEInts);
        return false;
    }

    Eigen::Matrix<double,-1,-1,Eigen::RowMajor> oneEInts((numberOfQubits/2),(numberOfQubits/2));
    double* twoEInts = new double[(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)*(numberOfQubits/2)];

    unsigned long __attribute__ ((unused))RetValue = ReadFile(fponeEInts, (unsigned char*)oneEInts.data(), oneEIntsBufferSize);
    assert(RetValue == oneEIntsBufferSize);
    RetValue = ReadFile(fptwoEInts, (unsigned char*)twoEInts, twoEIntsBufferSize);
    assert(RetValue == twoEIntsBufferSize);

    auto twoEIntsLookup = [numberOfQubits](size_t i, size_t j, size_t k, size_t l)
    {
        size_t numAO = numberOfQubits/2;
        // return (((i*numAO)+j)*numAO + k)*numAO + l;
        return (((i*numAO)+l)*numAO + j)*numAO + k; // pyscf ordering, (il|jk) = <ij|kl>
    };
    // auto getSpatialFromSpin = [numberOfQubits](size_t i){return i % (numberOfQubits/2);};

    auto getTwoElectronEnergy = [&twoEIntsLookup,twoEInts](const std::pair<size_t,bool> (&idxs)[4])
    {//Note the conventions on destroying in the same order you created.
        realNumType Energy = 0;
        if (idxs[0].second == idxs[2].second && idxs[1].second == idxs[3].second)
        {
            size_t twoEIntIdx = twoEIntsLookup(idxs[1].first,idxs[0].first,idxs[2].first,idxs[3].first);
            Energy += twoEInts[twoEIntIdx];
        }
        if (idxs[0].second == idxs[3].second && idxs[1].second == idxs[2].second)
        {
            size_t twoEIntIdx = twoEIntsLookup(idxs[1].first,idxs[0].first,idxs[3].first,idxs[2].first);
            Energy -= twoEInts[twoEIntIdx];
        }
        return Energy;
    };

    operators.clear();
    vals.clear();

    //Overestimates but not too bad. 32^4 ~ 100,000 so not expensive
    operators.reserve(numberOfQubits*numberOfQubits*numberOfQubits*numberOfQubits);
    vals.reserve(numberOfQubits*numberOfQubits*numberOfQubits*numberOfQubits);


    auto start = std::chrono::high_resolution_clock::now();


    constexpr double tol = 1e-13;
    assert(numberOfQubits < 32);
    for (std::uint_fast8_t a = 0; a < numberOfQubits; a++) //create
    {
        for (std::uint_fast8_t b = a+1; b < numberOfQubits; b++) //create
        {
            // uint32_t iBasisState = (1<<a) | (1<<b);
            for (std::uint_fast8_t c = 0; c < numberOfQubits; c++) //annihilate
            {
                for (std::uint_fast8_t d = c+1; d < numberOfQubits; d++) //annihilate
                {
                    //Note we include the permutation abcd,abdc, bacd, badc as the same thing. This is guaranteed by a < b, c < d

                    // uint32_t jBasisState = (1<<c) | (1<<d);

                    // realNumType Energy = getElement(iBasisState,jBasisState); //TODO optimise since we know these are
                    std::pair<size_t,bool> idxs[4];
                    idxs[0] = {a % (numberOfQubits/2), a >= numberOfQubits/2};
                    idxs[1] = {b % (numberOfQubits/2), b >= numberOfQubits/2};
                    idxs[2] = {c % (numberOfQubits/2), c >= numberOfQubits/2};
                    idxs[3] = {d % (numberOfQubits/2), d >= numberOfQubits/2};

                    //Formally H = a^\dagger_p a_q + 1/2 a^\dagger_p a^\dagger_q a_r a_s.
                    //The factor of two goes due to pqrs = qpsr. Exchange takes care of the pqrs->pqsr perm

                    double Energy = getTwoElectronEnergy(idxs); // Both fock and exchange
                    if (std::abs(Energy) > tol)//TODO threshold
                    {
                        // operators.push_back({a,b,c,d});
                        uint32_t create = (1<<a) | (1<<b);
                        uint32_t destroy = (1<<c) | (1<<d);
                        uint32_t signMask = ((1<<a)-1) ^ ((1<<b)-1) ^((1<<c)-1) ^((1<<d)-1);
                        signMask = signMask & ~((1<<a) | (1<<b) | (1<<c) | (1<<d));
                        operators.push_back({create,destroy,signMask});
                        vals.push_back(Energy);
                    }
                }
            }
        }
    }

    for (std::uint_fast8_t a = 0; a < numberOfQubits; a++)
    {
         // uint32_t iBasisState = (1<<a);
        for (std::uint_fast8_t c = 0; c < numberOfQubits; c++)
        {
            // uint32_t jBasisState = (1<<c);

            realNumType Energy = oneEInts(a % (numberOfQubits/2), c % (numberOfQubits/2));
            if (std::abs(Energy) > tol)//TODO threshold
            {
                // operators.push_back({a,a,c,c});
                uint32_t create = (1<<a);
                uint32_t destroy = (1<<c);
                uint32_t signMask = ((1<<a)-1) ^ ((1<<c)-1);
                signMask = signMask & ~((1<<a) | (1<<c));
                operators.push_back({create,destroy,signMask});
                vals.push_back(Energy);
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("Ham Time taken (ms)",duration);
    operators.shrink_to_fit();
    vals.shrink_to_fit();
    //Dont care about compression
    delete[] twoEInts;
    fclose(fponeEInts);
    fclose(fptwoEInts);
    return true;
}

template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::constructFromSecQuant()
{
    std::vector<Eigen::Triplet<dataType>> tripletList;

    // tripletList.reserve(m_data.size()); //TODO estimate for size
    for (uint32_t k = 0; k < m_linearSize; k++)
    {//basis states
        uint32_t iBasisState = k;
        if (m_isCompressed)
            m_compressor->deCompressIndex(iBasisState,iBasisState);

        auto opIt = m_operators.cbegin();
        auto valIt = m_vals.cbegin();
        const auto valItEnd = m_vals.cend();
        //The j loop
        for (;valIt != valItEnd; ++valIt,++opIt)
        {
            // fprintf(stderr,"Read Coeff: %lf \n ", coeff);
            // fprintf(stderr,"Read Index: %u,%u \n ", idxs[0]-1,idxs[1]-1);
            //Can apply to this basisState;.
            uint32_t j;
            bool sign;

            uint32_t destroy = iBasisState ^ opIt->destroy;
            bool canDestroy = (destroy & opIt->destroy) == 0;
            if (!canDestroy)
                continue;
            bool canCreate =  (destroy & opIt->create) == 0;
            if (!canCreate)
                continue;
            j  = destroy | opIt->create;

            sign = popcount(iBasisState & opIt->signBitMask) & 1;
            if (m_isCompressed)
                m_compressor->compressIndex(j,j);
            if (j == (uint32_t)-1)
                continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one
            tripletList.push_back(Eigen::Triplet<dataType>(k,j, (sign ? -*valIt : *valIt)));
        }
    }
    m_fullyConstructedMatrix.resize(m_linearSize,m_linearSize);
    m_fullyConstructedMatrix.setFromTriplets(tripletList.begin(),tripletList.end());
    logger().log("Element count", tripletList.size());

}

template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::postProcessOperators()
{
    size_t opSize = m_operators.size();
    std::vector<excOp> newOps;
    std::vector<dataType> newVals;
    logger().log("Ops before",opSize);
    newOps.reserve(opSize);
    newVals.reserve(opSize);

    for (size_t i = 0; i < opSize; i++)
    {
        if (m_compressor->opDoesSomething(m_operators[i]))
        {
            newOps.push_back(m_operators[i]);
            newVals.push_back(m_vals[i]);
        }
    }
    logger().log("Ops after", newOps.size());
    newOps.shrink_to_fit();
    newVals.shrink_to_fit();
    m_operators = std::move(newOps);
    m_vals = std::move(newVals);
}

template<typename dataType, typename vectorType>
HamiltonianMatrix<dataType, vectorType>::HamiltonianMatrix(const std::vector<dataType> &value, const std::vector<int> &iIndex, const std::vector<int> &jIndex, std::shared_ptr<compressor> comp)
{
    m_isCompressed = !!comp;
    m_compressor = comp;
    assert(value.size() == iIndex.size() && value.size() == jIndex.size());
    std::vector<Eigen::Triplet<dataType>> triplets;
    triplets.reserve(value.size());
    uint32_t maxI = 0;
    uint32_t maxJ = 0;
    for (size_t it = 0; it < value.size(); it++)
    {
        assert(iIndex[it] > 0 && jIndex[it] > 0);
        uint32_t i = iIndex[it];
        uint32_t j = jIndex[it];
        if (comp)
        {
            comp->compressIndex(i,i);
            comp->compressIndex(j,j);
        }
        if (i == (uint32_t)-1 || j == (uint32_t)-1)
            continue;
        triplets.push_back(Eigen::Triplet<dataType>(i,j,value[it]));
        maxI = std::max(i,maxI);
        maxJ = std::max(j,maxJ);
    }
    assert(maxI == maxJ);
    m_fullyConstructedMatrix.resize(maxI+1,maxJ+1);
    m_fullyConstructedMatrix.setFromTriplets(triplets.begin(),triplets.end());
    m_isFullyConstructed = true;

}

template<typename dataType, typename vectorType>
HamiltonianMatrix<dataType, vectorType>::HamiltonianMatrix(const std::string &filePath, size_t numberOfQubits, std::shared_ptr<compressor> comp)
{
    if (comp)
    {
        m_compressor = comp;
        m_isCompressed = true;
        m_linearSize = m_compressor->getCompressedSize();
    }
    else
    {
        m_linearSize = 1<<numberOfQubits;
    }
    logger().log("Matrix linear size",m_linearSize);
    bool success  = s_loadMatrix(m_fullyConstructedMatrix,filePath,comp,m_linearSize);
    if (success)
    {
        m_isFullyConstructed = true;
        logger().log("Loaded full");
        return;
    }

    success = s_loadOneAndTwoElectronsIntegrals<dataType>(m_operators,m_vals,filePath,numberOfQubits);
    m_isSecQuantConstructed = success;
    size_t sizeEstimate = 0.3*choose(numberOfQubits/2+2,2)*choose(numberOfQubits/2,2)*(1<<numberOfQubits);
    if (comp)
    {
        sizeEstimate = 0.3*comp->getCompressedSize() * choose(numberOfQubits/2+2,2)*choose(numberOfQubits/2,2);
        //This is an overestimate but we can always add more complicated estimate functions later
    }
    if (m_isSecQuantConstructed && sizeEstimate < maxSizeForFullConstruction && false)
    {
        //Construct Fully
        logger().log("Constructing fully, size estimate", sizeEstimate);
        constructFromSecQuant();
        m_isFullyConstructed = true;
        s_loadOneAndTwoElectronsIntegrals_Check(m_fullyConstructedMatrix,filePath,numberOfQubits,m_compressor);
    }
    else if (m_isSecQuantConstructed)
    {
        postProcessOperators();
        logger().log("Loaded secQuant, size estimate", sizeEstimate);
    }
    if (!success)
        logger().log("Could not construct Hamiltonian");
}
// void AVXFMA(double* destPtr,const double* srcPtr, double scalar, long numberOfRows)
// {
//     auto scalar4 = _mm256_broadcast_sd(&scalar);
//     int i = numberOfRows;
//     for ( ;i > 4; i-=4)
//     {
//         auto Double4A = _mm256_loadu_pd(srcPtr);
//         auto Double4destA = _mm256_loadu_pd(destPtr);
//         auto Double4FMAA = _mm256_fmadd_pd(Double4A,scalar4,Double4destA);
//         _mm256_storeu_pd(destPtr,Double4FMAA);
//         srcPtr+=4;
//         destPtr+=4;
//     }
//     for (;i>0; i--)
//     {
//         *destPtr++ += scalar*(*srcPtr);
//         srcPtr++;
//     }
// }
template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::apply(const Eigen::Matrix<vectorType,-1, -1, Eigen::ColMajor> &src,
                                                    Eigen::Matrix<vectorType, -1, -1, Eigen::ColMajor> &dest) const
{
    //The witchcraft and wizadry
    constexpr bool EigenMultiply = true;
    if (m_isFullyConstructed && EigenMultiply)
    {
        dest.noalias() = src*m_fullyConstructedMatrix;
        return;
    }
    else if (m_isFullyConstructed && !EigenMultiply)
    {
        //D_{ij} = src_{ik} H_{kj}
        long numberOfCols = src.cols();
        long numberOfRows = src.rows();
        dest.resize(numberOfRows,numberOfCols);
        dest.setZero();
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        long stepSize = std::max(numberOfCols/NUM_CORES,1ul);
        std::vector<std::future<void>> futs;
        for (long startj = 0; startj < numberOfCols; startj+= stepSize)
        {
            long endj = std::min(startj + stepSize,numberOfCols);
            futs.push_back(pool.queueWork([this,&src,&dest,startj,endj,numberOfRows](){
                for (long j = startj; j < endj; j++)
                {
                    //k axis
                    for (typename Eigen::SparseMatrix<dataType, Eigen::ColMajor>::InnerIterator it(m_fullyConstructedMatrix,j); it; ++it)
                    {
                        // dest.col(j) += src.col(it.row())*it.value();
#pragma GCC unroll 8
#pragma GCC ivdep
                        for (long i = 0; i < numberOfRows; i++)
                        {
                            dest(i,j) += src(i,it.row()) * it.value();
                        }
                        // AVXFMA(&dest(0,j),&src(0,it.row()),it.value(),numberOfRows);
                    }
                }}));
        }
        for (auto& f : futs)
            f.wait();

    }
    else if (m_isSecQuantConstructed)
    {
        //Need to go over the operators and apply them to the basisState one by one;
        //For now no unrolling and no SIMD

        long numberOfCols = src.cols();
        long numberOfRows = src.rows();
        dest.resize(numberOfRows,numberOfCols);
        dest.setZero();
        assert(m_compressor->getCompressedSize() == (size_t)numberOfCols);

        //Ready
        // T_{ik} H_{kj}
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        long stepSize = std::max(numberOfCols/NUM_CORES,1ul);
        std::vector<std::future<void>> futs;
        for (long startj = 0; startj < numberOfCols; startj+= stepSize)
        {
            long endj = std::min(startj + stepSize,numberOfCols);
            futs.push_back(pool.queueWork([this,&src,&dest,startj,endj,numberOfRows](){
                for (long j = startj; j < endj; j++)
                {//across
                    uint32_t jBasisState = j;
                    if (m_isCompressed)
                        m_compressor->deCompressIndex(jBasisState,jBasisState);

                    auto opIt = m_operators.cbegin();
                    auto valIt = m_vals.cbegin();
                    const auto valItEnd = m_vals.cend();
                    uint32_t badCreate = 0;
                    uint32_t goodCreate = 0;
                    uint32_t destroy;
                    bool canDestroy;
                    if (valIt != valItEnd)
                    {
                        goodCreate = opIt->create;
                        destroy = jBasisState ^ opIt->create;
                        canDestroy = (destroy & opIt->create) == 0;
                        if (!canDestroy)
                        {
                            badCreate = opIt->create;
                        }
                    }
                    //The j loop
                    for (;valIt != valItEnd; ++valIt,++opIt)
                    {
                        if (opIt->create == badCreate)
                            continue;
                        badCreate = 0;

                        uint32_t i;
                        bool sign;


                        if (opIt->create != goodCreate)
                        {
                            goodCreate = opIt->create;
                            destroy = jBasisState ^ opIt->create;
                            canDestroy = (destroy & opIt->create) == 0;
                            if (!canDestroy)
                            {
                                badCreate = opIt->create;
                                continue;
                            }
                        }

                        bool canCreate =  (destroy & opIt->destroy) == 0;
                        if (!canCreate)
                            continue;
                        i  = destroy | opIt->destroy;

                        sign = popcount(jBasisState & opIt->signBitMask) & 1;
                        if (m_isCompressed)
                            m_compressor->compressIndex(i,i);
                        if (i == (uint32_t)-1)
                            continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one

                        // for (long i = 0; i < numberOfRows; i++)
                        // {//down
                        //     HT_C(i,j) += T_C(i,k) * valIt;
                        // }
                        // dest.col(j) += src.col(i) * ((sign ? -1 : 1)* *valIt);
                        dataType v = (sign ? -*valIt : *valIt);
#pragma GCC unroll 8
#pragma GCC ivdep
                        for (long r = 0; r < numberOfRows; r++)
                        {
                            dest(r,j) += src(r,i) * v;
                        }
                        // AVXFMA(&dest(0,j),&src(0,i),v,numberOfRows);
                    }
                }
            }));
        }
        for (auto& f : futs)
            f.wait();
        return;
    }
    else
    {
        logger().log("Hamiltonian is in error state");
        __builtin_trap();
    }
}

template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::apply(const Eigen::Map<const Eigen::Matrix<vectorType, 1, -1, Eigen::RowMajor>, Eigen::Aligned32> &src,
                                                    Eigen::Map<Eigen::Matrix<vectorType, 1, -1, Eigen::RowMajor>, Eigen::Aligned32> &dest) const
{//This is a copy paste from above. TODO work with Dense base?
    //The witchcraft and wizadry
    constexpr bool EigenMultiply = true;
    if (m_isFullyConstructed && EigenMultiply)
    {
        dest.noalias() = src*m_fullyConstructedMatrix;
        return;
    }
    else if (m_isFullyConstructed && !EigenMultiply)
    {
        //D_{ij} = src_{ik} H_{kj}
        long numberOfCols = src.cols();
        assert(dest.cols() == numberOfCols);
        dest.setZero();
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        long stepSize = std::max(numberOfCols/NUM_CORES,1ul);
        std::vector<std::future<void>> futs;
        for (long startj = 0; startj < numberOfCols; startj+= stepSize)
        {
            long endj = std::min(startj + stepSize,numberOfCols);
            futs.push_back(pool.queueWork([this,&src,&dest,startj,endj](){
                for (long j = startj; j < endj; j++)
                {
                    //k axis
                    for (typename Eigen::SparseMatrix<dataType, Eigen::ColMajor>::InnerIterator it(m_fullyConstructedMatrix,j); it; ++it)
                    {
                        // dest.col(j) += src.col(it.row())*it.value();
                        dest(0,j) += src(0,it.row()) * it.value();

                    }
                }}));
        }
        for (auto& f : futs)
            f.wait();

    }
    else if (m_isSecQuantConstructed)
    {
        //Need to go over the operators and apply them to the basisState one by one;
        //For now no unrolling and no SIMD

        long numberOfCols = src.cols();
        long numberOfRows = src.rows();
        dest.resize(numberOfRows,numberOfCols);
        dest.setZero();
        assert(m_compressor->getCompressedSize() == (size_t)numberOfCols);

        //Ready
        // T_{ik} H_{kj}
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        long stepSize = std::max(numberOfCols/NUM_CORES,1ul);
        std::vector<std::future<void>> futs;
        for (long startj = 0; startj < numberOfCols; startj+= stepSize)
        {
            long endj = std::min(startj + stepSize,numberOfCols);
            futs.push_back(pool.queueWork([this,src = src.data(),dest = dest.data(),startj,endj](){
                long endJ8 = ((endj-startj)/8)*8 + startj;
                assert((endJ8-startj)%8 == 0);
                for (long j = startj; j < endJ8; j+=8)
                {//across
                    //Do 8 JBasisStates at a time
                    uint32_t jBasisStates[8];
                    jBasisStates[0] = j;
                    jBasisStates[1] = j+1;
                    jBasisStates[2] = j+2;
                    jBasisStates[3] = j+3;
                    jBasisStates[4] = j+4;
                    jBasisStates[5] = j+5;
                    jBasisStates[6] = j+6;
                    jBasisStates[7] = j+7;

                    if (m_isCompressed)
                        //Find the bit strings for the JBasisStates
                        m_compressor->deCompressIndex(jBasisStates,jBasisStates);

                    auto opIt = m_operators.cbegin();
                    auto valIt = m_vals.cbegin();
                    const auto valItEnd = m_vals.cend();
                    uint32_t badCreate = 0;
                    uint32_t goodCreate = 0;

                    __m256i destroys;
                    __m256i BCastCreate = _mm256_broadcastd_epi32(_mm_loadu_si32(&opIt->create));
                    __m256i JBasisStatesVec = _mm256_loadu_epi32(jBasisStates);

                    __mmask8 canDestroys;
                    if (valIt != valItEnd)
                    {
                        //Yes yes these are backwards to the way it was defined however for real hamiltonians it doesnt matter because theyre Hermitian. see comments on a,b,c,d
                        goodCreate = opIt->create;
                        //Apply the Creation operator on the right, (Destroy)
                        destroys = JBasisStatesVec ^ BCastCreate;
                        //Check that the bits have been destroyed, Same as checking they were 1 before
                        canDestroys = _mm256_testn_epi32_mask(destroys, BCastCreate);
                        //Check that this did something
                        if (canDestroys == 0)
                            badCreate = opIt->create;
                    }
                    //The j loop
                    for (;valIt != valItEnd; ++valIt,++opIt)
                    {
                        //Skip if known that it does nothing
                        if (opIt->create == badCreate)
                            continue;
                        badCreate = 0;

                        __mmask8 signs;

                        //createOp

                        if (opIt->create != goodCreate)
                        {
                            //Broadcast the create op to the whole vector
                            BCastCreate = _mm256_set1_epi32(opIt->create);
                            goodCreate = opIt->create;
                            //Apply the Creation operator on the right, (Destroy)
                            destroys = JBasisStatesVec ^ BCastCreate;
                            //Check that the bits have been destroyed, Same as checking they were 1 before
                            canDestroys = _mm256_testn_epi32_mask(destroys, BCastCreate);
                            //Check that this did something
                            if (canDestroys == 0)
                            {
                                badCreate = opIt->create;
                                continue;
                            }
                        }

                        //destroyOp
                        __mmask8 canCreates;
                        __m256i BCastDestroy = _mm256_set1_epi32(opIt->destroy);
                        //Check that the bits can be created
                        canCreates =  _mm256_testn_epi32_mask(destroys, BCastDestroy);
                        //Check that this did something
                        if (canCreates == 0)
                            continue;

                        __m256i is;
                        //Setup the signmask as a vector
                        __m256i BCastSignBitMask = _mm256_set1_epi32(opIt->signBitMask);
                        //Create the bits
                        is = destroys | BCastDestroy;
                        __m256i BCast1s = _mm256_set1_epi32(1);
                        //Compute the signs. bool signs =  popcount(JBasisStatesVec & signBitMask) & 1. sign = 1 => negative
                        signs = _mm256_test_epi32_mask(_mm256_popcnt_epi32(JBasisStatesVec & BCastSignBitMask),BCast1s);


                        __mmask8 valid = -1;
                        //Compress the indices if needed
                        if (m_isCompressed)
                            m_compressor->compressIndex(is,is,valid);
                        //check that we need to do something
                        valid = valid & canCreates & canDestroys;
                        if (valid == 0)
                            continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one

                        //Load the value vector
                        __m512d vs = _mm512_set1_pd(*valIt);
                        __m512d negZero = _mm512_set1_pd(-0.0);
                        //Negate the terms indicated by signs
                        vs = _mm512_mask_xor_pd(vs,signs,vs,negZero);

                        //Load the dest vector for FMA
                        __m512d destVec = _mm512_loadu_pd(dest + j);
                        //Load the src vector for FMA, Mask out any invalids with negZero. (exact value doesnt matter)
                        __m512d srcVec = _mm512_mask_i32gather_pd(negZero,valid,is,src,sizeof(double));
                        //FMA, Mask means that if valid == 0 then FMA copies from destVec. i.e. does nothing
                        destVec = _mm512_mask3_fmadd_pd(srcVec,vs,destVec,valid);
                        //Store the result. Take care of unaligned accesses potentially.
                        _mm512_storeu_pd(dest + j,destVec);

                    }
                }
                for (long j = endJ8; j < endj; j++)
                {//across
                    uint32_t jBasisState = j;
                    if (m_isCompressed)
                    {
                        m_compressor->deCompressIndex(jBasisState,jBasisState);
                    }

                    auto opIt = m_operators.cbegin();
                    auto valIt = m_vals.cbegin();
                    const auto valItEnd = m_vals.cend();
                    uint32_t badCreate = 0;
                    uint32_t goodCreate = 0;

                    uint32_t destroy;

                    bool canDestroy;
                    if (valIt != valItEnd)
                    {
                        //Yes yes these are backwards to the way it was defined however for real hamiltonians it doesnt matter because theyre Hermitian. see comments on a,b,c,d
                        goodCreate = opIt->create;
                        destroy = jBasisState ^ opIt->create;
                        canDestroy = (destroy & opIt->create) == 0;
                        if (!canDestroy)
                        {
                            badCreate = opIt->create;
                        }
                    }
                    //The j loop
                    for (;valIt != valItEnd; ++valIt,++opIt)
                    {
                        if (opIt->create == badCreate)
                            continue;
                        badCreate = 0;

                        uint32_t i;
                        bool sign;


                        if (opIt->create != goodCreate)
                        {
                            goodCreate = opIt->create;
                            destroy = jBasisState ^ opIt->create;
                            canDestroy = (destroy & opIt->create) == 0;
                            if (!canDestroy)
                            {
                                badCreate = opIt->create;
                                continue;
                            }
                        }

                        bool canCreate =  (destroy & opIt->destroy) == 0;

                        if (!canCreate)
                            continue;

                        i  = destroy | opIt->destroy;
                        sign = popcount(jBasisState & opIt->signBitMask) & 1;

                        bool OutOfSpace = false;
                        if (m_isCompressed)
                            OutOfSpace = !m_compressor->compressIndex(i,i);

                        if (m_isCompressed && OutOfSpace)
                            continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one

                        dataType v;
                        v = (sign ? -*valIt : *valIt);
                        dest[j] += src[i] * v;

                    }
                }
            }));
        }
        for (auto& f : futs)
            f.wait();
        return;
    }
    else
    {
        logger().log("Hamiltonian is in error state");
        __builtin_trap();
    }
}




template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::apply(const Eigen::Map<const Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> &src,
                                                    Eigen::Map<Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> &dest,
                                                    const Eigen::SparseMatrix<realNumType, Eigen::RowMajor>* compress) const
{
    //For speed need to convert the ordering, We accept RowMajor since that is most convenient for expected users. Converting is an implementation detail
    Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> srcC;
    Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> T_C;
    if (compress)
    {
        srcC = src;
        T_C.noalias() = *compress * srcC;

    }
    else
        T_C = src;

    Eigen::Matrix<vectorType,-1,-1,Eigen::ColMajor> HT_C;
    apply(T_C,HT_C);
    dest = HT_C;
    return;
}

template<typename dataType, typename vectorType>
vector<vectorType> & HamiltonianMatrix<dataType, vectorType>::apply(const vector<vectorType> &src, vector<vectorType> &dest) const
{
    {
        std::shared_ptr<compressor> c;
        assert(src.getIsCompressed(c) == m_isCompressed && c == m_compressor);
    }

    dest.resize(src.size(),m_isCompressed,m_compressor);
    Eigen::Map<const Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> srcMap(&src[0],1,src.size());
    Eigen::Map<Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> retMap(&dest[0],1,dest.size());
    apply(srcMap,retMap);
    return dest;
}
template<typename dataType, typename vectorType>
vector<vectorType> HamiltonianMatrix<dataType, vectorType>::apply(const vector<vectorType> &src) const
{
    {
        std::shared_ptr<compressor> c;
        assert(src.getIsCompressed(c) == m_isCompressed && c == m_compressor);
    }
    vector<vectorType> ret;
    ret.resize(src.size(),m_isCompressed,m_compressor);
    Eigen::Map<const Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> srcMap(&src[0],1,src.size());
    Eigen::Map<Eigen::Matrix<vectorType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> retMap(&ret[0],1,ret.size());
    apply(srcMap,retMap);
    return ret;
}








#ifdef useComplex
template class HamiltonianMatrix<realNumType,realNumType>;
template class HamiltonianMatrix<realNumType,numType>;
template class HamiltonianMatrix<numType,numType>;
#else
template class HamiltonianMatrix<numType,numType>;
#endif


