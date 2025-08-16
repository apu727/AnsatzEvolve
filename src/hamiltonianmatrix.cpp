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
        if constexpr (std::is_same_v<dataType,typename Eigen::NumTraits<dataType>::Real>)
            tripletList.push_back(Eigen::Triplet<dataType>(compi,compj,coeffReal));
        else
            tripletList.push_back(Eigen::Triplet<dataType>(compi,compj,dataType(coeffReal,coeffImag)));

        ret = fscanf(fpCoeff, realNumTypeCode ", " realNumTypeCode " \n", &coeffReal, &coeffImag);
        ret2 = fscanf(fpIndex, "%u %u\n",&(idxs[0]),&(idxs[1]));
    }
    destMat.resize(linearSize,linearSize);
    destMat.setFromTriplets(tripletList.begin(),tripletList.end());
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
bool s_loadOneAndTwoElectronsIntegrals_Check(Eigen::SparseMatrix<dataType, Eigen::ColMajor> CheckWith, std::string filePath,size_t numberOfQubits, std::shared_ptr<compressor> comp)
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

    unsigned long RetValue = ReadFile(fponeEInts, (unsigned char*)oneEInts.data(), oneEIntsBufferSize);
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
        logger().log("MatrixCompressedSize",compressedSize);
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

                        realNumType Energy = getElement(iBasisState,jBasisState);
                        uint32_t compi = iBasisState;
                        uint32_t compj = jBasisState;
                        if (comp)
                        {
                            comp->compressIndex(compi,compi);
                            comp->compressIndex(compj,compj);
                        }

                        assert(abs(CheckWith.coeff(compi,compj) - Energy) < 1e-13);
                        assert(abs(CheckWith.coeff(compj,compi) - Energy) < 1e-13);

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

                realNumType Energy = getElement(iBasisState,jBasisState);
                uint32_t compi = iBasisState;
                uint32_t compj = jBasisState;
                if (comp)
                {
                    comp->compressIndex(compi,compi);
                    comp->compressIndex(compj,compj);
                }

                assert(abs(CheckWith.coeff(compi,compj) - Energy) < 1e-13);
                assert(abs(CheckWith.coeff(compj,compi) - Energy) < 1e-13);
            }

        }

        realNumType Energy = getElement(jBasisState,jBasisState);
        uint32_t compj = jBasisState;
        if (comp)
            comp->compressIndex(compj,compj);

        assert(abs(CheckWith.coeff(compj,compj) - Energy) < 1e-13);
    }

    delete[] twoEInts;
    fclose(fponeEInts);
    fclose(fptwoEInts);
    return true;
}

template<typename dataType,typename vectorType>
bool s_loadOneAndTwoElectronsIntegrals(std::vector<typename HamiltonianMatrix<dataType,vectorType>::excOp>& operators,
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

    unsigned long RetValue = ReadFile(fponeEInts, (unsigned char*)oneEInts.data(), oneEIntsBufferSize);
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

    auto getElement = [numberOfQubits,&getEnergy,&getTwoElectronEnergy,getFockMatrixElem,&oneEInts](uint32_t iBasisState, uint32_t jBasisState, bool onlyTwoElectron)
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
            // if (isSet == jsSet)
            //     continue;
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

        // if(annihilatePos == 2 && createPos == 0)
        // {
        //     Energy = getEnergy(jBasisState);
        // }
        // else if (annihilatePos == 3 && createPos == 1)
        // {
        //     Energy = (sign ? 1 : -1)*getFockMatrixElem(idxs,jBasisState);
        // }
        // else if (annihilatePos == 4 && createPos == 2)
        // {
        //     Energy = (sign ? 1 : -1)*getTwoElectronEnergy(idxs);
        // }
        // else
        // {
        //     logger().log("Not handled case construct Ham");
        //     __builtin_trap();
        // }
        if (onlyTwoElectron)
        {
            Energy = (sign ? 1 : -1)*getTwoElectronEnergy(idxs);
        }
        if (!onlyTwoElectron)
        {
           Energy = oneEInts(idxs[0].first,idxs[2].first);
        }
        return Energy;
    };


    operators.clear();
    vals.clear();

    //Overestimates but not too bad. 32^4 ~ 100,000 so not expensive
    operators.reserve(numberOfQubits*numberOfQubits*numberOfQubits*numberOfQubits);
    vals.reserve(numberOfQubits*numberOfQubits*numberOfQubits*numberOfQubits);


    auto start = std::chrono::high_resolution_clock::now();



    assert(numberOfQubits < 32);
    for (std::uint_fast8_t a = 0; a < numberOfQubits; a++) //create
    {
        for (std::uint_fast8_t b = a+1; b < numberOfQubits; b++) //create
        {
            uint32_t iBasisState = (1<<a) | (1<<b);
            for (std::uint_fast8_t c = 0; c < numberOfQubits; c++) //annihilate
            {
                for (std::uint_fast8_t d = c+1; d < numberOfQubits; d++) //annihilate
                {
                    //Note we include the permutation abcd,abdc, bacd, badc as the same thing. This is guaranteed by a < b, c < d

                    uint32_t jBasisState = (1<<c) | (1<<d);

                    // realNumType Energy = getElement(iBasisState,jBasisState); //TODO optimise since we know these are
                    std::pair<size_t,bool> idxs[4];
                    idxs[0] = {a % (numberOfQubits/2), a >= numberOfQubits/2};
                    idxs[1] = {b % (numberOfQubits/2), b >= numberOfQubits/2};
                    idxs[2] = {c % (numberOfQubits/2), c >= numberOfQubits/2};
                    idxs[3] = {d % (numberOfQubits/2), d >= numberOfQubits/2};

                    //Formally H = a^\dagger_p a_q + 1/2 a^\dagger_p a^\dagger_q a_r a_s.
                    //The factor of two goes due to pqrs = qpsr. Exchange takes care of the pqrs->pqsr perm

                    double Energy = getTwoElectronEnergy(idxs); // Both fock and exchange
                    double Energy2 = getElement(iBasisState,jBasisState,true);
                    assert(Energy == Energy2);
                    if (abs(Energy) > 1e-15)//TODO threshold
                    {
                        operators.push_back({a,b,c,d});
                        vals.push_back(Energy);
                    }
                }
            }
        }
    }

    for (std::uint_fast8_t a = 0; a < numberOfQubits; a++)
    {
         uint32_t iBasisState = (1<<a);
        for (std::uint_fast8_t c = 0; c < numberOfQubits; c++)
        {
            uint32_t jBasisState = (1<<c);

            realNumType Energy = oneEInts(a % (numberOfQubits/2), c % (numberOfQubits/2));
            double Energy2 = getElement(iBasisState,jBasisState,false);
            assert(Energy == Energy2);
            if (abs(Energy) > 1e-15)//TODO threshold
            {
                operators.push_back({a,a,c,c});
                vals.push_back(Energy);
            }
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
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
            if (opIt->a == opIt->b)
            {
                assert(opIt->c == opIt->d);
                uint32_t destroyOp = (1<<opIt->a);
                uint32_t createOp = (1<<opIt->c);
                uint32_t destroy = iBasisState ^ destroyOp;
                bool canDestroy = ((iBasisState & destroyOp) ^ destroyOp) == 0;
                if (!canDestroy)
                    continue;
                bool canCreate =  (destroy & createOp) == 0;
                if (!canCreate)
                    continue;
                j  = destroy | createOp;

                sign = (bitwiseDot(iBasisState,-1,opIt->a) & 1);

                if (bitwiseDot(j,-1,opIt->c) & 1)
                    sign = !sign;
                if (m_isCompressed)
                    m_compressor->compressIndex(j,j);
                if (j == (uint32_t)-1)
                    continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one
            }
            else
            {
                assert(opIt->c != opIt->d);
                uint32_t destroyOp = ((1<<opIt->a) | (1<<opIt->b));
                uint32_t createOp = (1<<opIt->c) | (1<<opIt->d);
                uint32_t destroy = iBasisState ^ destroyOp;
                bool canDestroy = ((iBasisState & destroyOp) ^ destroyOp) == 0;
                if (!canDestroy)
                    continue;
                bool canCreate =  (destroy & createOp) == 0;
                if (!canCreate)
                    continue;
                j  = destroy | createOp;

                sign = (bitwiseDot(iBasisState,-1,opIt->a) & 1);
                if ((bitwiseDot(iBasisState,-1,opIt->b) & 1))
                    sign = !sign;

                if (bitwiseDot(j,-1,opIt->c) & 1)
                    sign = !sign;
                if ((bitwiseDot(j,-1,opIt->d) & 1))
                    sign = !sign;

                if (m_isCompressed)
                    m_compressor->compressIndex(j,j);
                if (j == (uint32_t)-1)
                    continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one
            }
            tripletList.push_back(Eigen::Triplet<dataType>(k,j,(sign ? -1 : 1)* *valIt));
        }
    }
    m_fullyConstructedMatrix.resize(m_linearSize,m_linearSize);
    m_fullyConstructedMatrix.setFromTriplets(tripletList.begin(),tripletList.end());
    logger().log("Element count", tripletList.size());

}

template<typename dataType, typename vectorType>
HamiltonianMatrix<dataType, vectorType>::HamiltonianMatrix(const std::vector<dataType> &value, const std::vector<int> &iIndex, const std::vector<int> &jIndex)
{
    assert(value.size() == iIndex.size() && value.size() == jIndex.size());
    std::vector<Eigen::Triplet<dataType>> triplets;
    triplets.reserve(value.size());
    for (size_t i = 0; i < value.size(); i++)
    {
        triplets.push_back(Eigen::Triplet<dataType>(iIndex[i],jIndex[i],value[i]));
    }
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
    bool success  = s_loadMatrix(m_fullyConstructedMatrix,filePath,comp,m_linearSize);
    if (success)
    {
        m_isFullyConstructed = true;
        logger().log("Loaded full");
        return;
    }

    success = s_loadOneAndTwoElectronsIntegrals<dataType,vectorType>(m_operators,m_vals,filePath,numberOfQubits);
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
    else
    {
        logger().log("Loaded secQuant, size estimate", sizeEstimate);
    }
    if (!success)
        logger().log("Could not construct Hamiltonian");
}

template<typename dataType, typename vectorType>
void HamiltonianMatrix<dataType, vectorType>::apply(const Eigen::Matrix<vectorType, -1, -1, Eigen::ColMajor> &src, Eigen::Matrix<vectorType, -1, -1, Eigen::ColMajor> &dest) const
{
    //The witchcraft and wizadry
    if (m_isFullyConstructed)
    {
        dest.noalias() = src*m_fullyConstructedMatrix;
        return;
    }
    else if (m_isSecQuantConstructed)
    {
        //Need to go over the operators and apply them to the basisState one by one;
        //For now no unrolling and no SIMD

        long numberOfCols = src.cols();
        long numberOfRows = src.rows();
        dest.resize(numberOfRows,numberOfCols);
        dest.setZero();

        //Ready
        // T_{ik} H_{kj}
        threadpool& pool = threadpool::getInstance(NUM_CORES);
        long stepSize = std::min(numberOfCols/NUM_CORES,1ul);
        std::vector<std::future<void>> futs;
        for (long startk = 0; startk < numberOfCols; startk+= stepSize)
        {
            long endK = std::min(startk + stepSize,numberOfCols);
            futs.push_back(pool.queueWork([this,&src,&dest,startk,endK](){
        for (long k = startk; k < endK; k++)
        {//across
            uint32_t iBasisState = k;
            if (m_isCompressed)
                m_compressor->deCompressIndex(iBasisState,iBasisState);

            auto opIt = m_operators.cbegin();
            auto valIt = m_vals.cbegin();
            const auto valItEnd = m_vals.cend();
            //The j loop
            for (;valIt != valItEnd; ++valIt,++opIt)
            {
                uint32_t j;
                bool sign;
                if (opIt->a == opIt->b)
                {
                    assert(opIt->c == opIt->d);
                    uint32_t destroyOp = (1<<opIt->a);
                    uint32_t createOp = (1<<opIt->c);
                    uint32_t destroy = iBasisState ^ destroyOp;
                    bool canDestroy = ((iBasisState & destroyOp) ^ destroyOp) == 0;
                    if (!canDestroy)
                        continue;
                    bool canCreate =  (destroy & createOp) == 0;
                    if (!canCreate)
                        continue;
                    j  = destroy | createOp;

                    sign = (bitwiseDot(iBasisState,-1,opIt->a) & 1);

                    if (bitwiseDot(j,-1,opIt->c) & 1)
                        sign = !sign;
                    if (m_isCompressed)
                        m_compressor->compressIndex(j,j);
                    if (j == (uint32_t)-1)
                        continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one
                }
                else
                {
                    assert(opIt->c != opIt->d);
                    uint32_t destroyOp = ((1<<opIt->a) | (1<<opIt->b));
                    uint32_t createOp = (1<<opIt->c) | (1<<opIt->d);
                    uint32_t destroy = iBasisState ^ destroyOp;
                    bool canDestroy = ((iBasisState & destroyOp) ^ destroyOp) == 0;
                    if (!canDestroy)
                        continue;
                    bool canCreate =  (destroy & createOp) == 0;
                    if (!canCreate)
                        continue;
                    j  = destroy | createOp;

                    sign = (bitwiseDot(iBasisState,-1,opIt->a) & 1);
                    if ((bitwiseDot(iBasisState,-1,opIt->b) & 1))
                        sign = !sign;

                    if (bitwiseDot(j,-1,opIt->c) & 1)
                        sign = !sign;
                    if ((bitwiseDot(j,-1,opIt->d) & 1))
                        sign = !sign;

                    if (m_isCompressed)
                        m_compressor->compressIndex(j,j);
                    if (j == (uint32_t)-1)
                        continue;//Out of the space. Probably would cancel somwhere else assuming the symmetry is a valid one
                }

                // for (long i = 0; i < numberOfRows; i++)
                // {//down
                //     HT_C(i,j) += T_C(i,k) * valIt;
                // }
                dest.col(j) += src.col(k) * ((sign ? -1 : 1)* *valIt); // Should be nicely SIMD but eigen may betray me for small vectors
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
    Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> srcC;
    Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> T_C;
    if (compress)
    {
        srcC = src;
        T_C.noalias() = *compress * srcC;
    }
    else
        T_C = src;

    Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> HT_C;
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
    Eigen::Map<const Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> srcMap(&src[0],1,src.size());
    Eigen::Map<Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> retMap(&dest[0],1,dest.size());
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
    vector<numType> ret;
    ret.resize(src.size(),m_isCompressed,m_compressor);
    Eigen::Map<const Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> srcMap(&src[0],1,src.size());
    Eigen::Map<Eigen::Matrix<vectorType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32> retMap(&ret[0],1,ret.size());
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








