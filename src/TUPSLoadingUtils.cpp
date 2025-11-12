/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "TUPSLoadingUtils.h"
#include "logger.h"


bool loadPath(std::shared_ptr<stateRotate> sr, std::string filePath, std::vector<ansatz::rotationElement>& rotationPath)
{
    FILE *fp;
    stateRotate::exc Excs;

    fp = fopen(filePath.c_str(), "r");
    if(NULL == fp)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }
    int ret = fscanf(fp, "%hhd %hhd %hhd %hhd \n",&Excs[0],&Excs[1],&Excs[2],&Excs[3] );
    while(EOF != ret)
    {
        // fprintf(stderr,"Read Operator: %hhd %hhd %hhd %hhd\n ", Excs[0],Excs[1],Excs[2],Excs[3]);
        for (int i = 0; i < 4; i++)
            Excs[i] -= 1;
        if (sr)
            rotationPath.push_back({sr->convertDataToIdx(&Excs),0});
        else
            rotationPath.push_back({0,0});
        ret = fscanf(fp, "%hhd %hhd %hhd %hhd \n",&Excs[0],&Excs[1],&Excs[2],&Excs[3] );
    }
    fclose(fp);
    return 1;
}

bool loadParameters(std::string filePath,
                    std::vector<ansatz::rotationElement>& rotationPath,
                    std::vector<std::vector<ansatz::rotationElement>> &rotationPaths,
                    std::vector<std::pair<int,realNumType>> &orders/*the fixed relation between parameters*/,
                    int& numberOfUniqueParameters)
{
    FILE *fp;
    FILE *fp2;

    fp = fopen((filePath+"_Parameters.dat").c_str(), "r");
    if(fp == nullptr)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath+"_Parameters.dat").c_str());
        return 0;
    }

    fp2 = fopen((filePath+"_Order.dat").c_str(), "r");
    if(fp2 == nullptr)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",(filePath+"_Order.dat").c_str());
        return 0;
    }




    /* Hardcoded parsing of a file of the format
     *
           9
Energy of minimum      1=  -2.844887240192929 first found at step        5 after                  245 function calls
        0.182710385127496
        1.657072388288146
        0.110549543159596
        0.878483796766535
        0.225437898626115
        1.095597930656243
       -0.000000000027978
        0.797060942289907
       -1.570796326773916
           9
Energy of minimum      2=  -2.809247486912495 first found at step       12 after                  681 function calls
       -0.869149415717867
       -0.702644479156458
        2.760870118766821
       -1.013671622127060
        1.314899820469964
       -1.097762093300513
        1.570796326889411
       -0.087609819576141
       -0.000000000090040
...etc
*/
    int ret = 0;
    int numberOfParameters = 0;
    realNumType parameter = realNumType();
    std::vector<std::vector<realNumType>> parameters;

    while(EOF != ret)
    {
        ret = fscanf(fp, "%i", &numberOfParameters);

        if (EOF == ret)
            break;

        if (ret != 1)
            fprintf(stderr,"Format Invalid, Number of Parameters");

        realNumType Energy = realNumType();
        ret = fscanf(fp,"%*[^=]= " realNumTypeCode " %*[^\n]",&Energy);
        // fprintf(stderr,"Path: %zu has energy " realNumTypeCode "\n",parameters.size()+1,Energy);

        parameters.push_back(std::vector<realNumType>());
        for (int i = 0; i < numberOfParameters; i++)
        {
            ret = fscanf(fp,  realNumTypeCode,&parameter);
            parameters.back().push_back(parameter);
            // fprintf(stderr,"Read Angle: " numTypeCode "\n ", parameter);
        }
    }
    numberOfUniqueParameters = numberOfParameters;

    int order = 0;
    realNumType scaleFactor = 1;
    int ret2 = fscanf(fp2,  "%i," realNumTypeCode,&order,&scaleFactor);
    order = order-1;


    size_t pathIndex = 0; // which element in the path
    rotationPaths.assign(parameters.size()+1,rotationPath); // adding in the 0 angle path
    int maxOrder = 0;

    while(ret2 != EOF)
    {
        // fill all the paths with the different parameter sequences according
        // to the parameter -> position mapping given
        if (order < 0)
            fprintf(stderr,"Order cannot be negative\n");
        orders.push_back({order,scaleFactor});
        maxOrder = std::max(maxOrder,order);
        if (order < numberOfParameters && pathIndex < rotationPath.size())
        {
            for (size_t i = 0; i < parameters.size(); i++)
            {
                rotationPaths[i+1][pathIndex].second = parameters[i][order]*scaleFactor;
            }
        }
        pathIndex++;
        scaleFactor = 1;
        ret2 = fscanf(fp2,  "%i," realNumTypeCode ,&order,&scaleFactor);
        order = order-1;
    }
    numberOfUniqueParameters = std::max(maxOrder+1,numberOfUniqueParameters);
    fclose(fp2);
    fclose(fp);
    if (pathIndex != rotationPath.size())
    {
        fprintf(stderr, "More operators in path than relations given\n");
        return 0;
    }
    return 1;
}


int readCsvState(std::vector<std::complex<realNumType>>& Coeffs, const std::string& filePath)
{
    FILE *fp;
    char str1[32];
    realNumType coeffReal = 0;
    realNumType coeffImag = 0;

    fp = fopen(filePath.c_str(), "r");
    if(NULL == fp)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }

    int ret = 0;

    while(EOF != ret )
    {
        coeffReal = 0;
        coeffImag = 0;
        ret = fscanf(fp, " %31[10], " realNumTypeCode " , " realNumTypeCode " ,\n", str1, &coeffReal,&coeffImag);
        if (ret == EOF)
            break;
        if (ret < 2)
        {
            if (ret == 1 && coeffReal == 0)
            {
                fprintf(stderr,"warning assuming initial state starts with magnitude 1\n");
                coeffReal = 1;
            }
            else
                fprintf(stderr,"Unable to read initial state, missing bitstring?\n");
        }
        //fprintf(stderr,"Read: %s, " numTypeCode, str1, coeff);
        std::string bitstring(str1);
        uint64_t bitstring_Int = 0;
        for (size_t i=0; i < bitstring.length(); i++)
        {
            bitstring_Int |= ((1 << i) * (bitstring[bitstring.length()-i-1] == '1'));
        }
        size_t vectorSizeNeeded = 1<<bitstring.length();
        if (Coeffs.size() < vectorSizeNeeded)
            Coeffs.resize(vectorSizeNeeded,0);

        Coeffs[bitstring_Int] = std::complex<double>(coeffReal,coeffImag);
    }
    fclose(fp);
    return 1;
}

int readCsvState(std::vector<realNumType>& Coeffs, std::vector<uint64_t>& indexes, const std::string& filePath, int &numQubits)
{
    FILE *fp;
    char str1[32];
    realNumType coeff = 0;
    realNumType imag = 0;


    fp = fopen(filePath.c_str(), "r");
    if(NULL == fp)
    {
        fprintf(stderr,"\nError in opening file.");
        fprintf(stderr,"fileGiven: %s\n",filePath.c_str());
        return 0;
    }

    int ret = 0;
    numQubits = -1;
    Coeffs.clear();
    indexes.clear();

    while(EOF != ret )
    {
        coeff = 0;
        ret = fscanf(fp, " %31[10], " realNumTypeCode " , " realNumTypeCode " ,\n", str1, &coeff,&imag);
        if (ret == EOF)
            break;
        if (ret < 2)
        {
            if (ret == 1 && coeff == 0)
            {
                fprintf(stderr,"warning assuming initial state starts with magnitude 1\n");
                coeff = 1;
            }
            else
                fprintf(stderr,"Unable to read initial state, missing bitstring?\n");
        }
        if (imag != 0)
        {
            logger().log("Imaginary coeff not zero: ",str1);
        }
        //fprintf(stderr,"Read: %s, " numTypeCode, str1, coeff);
        std::string bitstring(str1);
        uint64_t bitstring_Int = 0;
        for (size_t i=0; i < bitstring.length(); i++)
        {
            bitstring_Int |= ((1 << i) * (bitstring[bitstring.length()-i-1] == '1'));
        }
        if (numQubits == -1)
        {
            numQubits = bitstring.length();
        }
        else
        {
            releaseAssert(numQubits ==  (int)bitstring.length(),"numQubits ==  bitstring.length()");
        }
        Coeffs.push_back(coeff);
        indexes.push_back(bitstring_Int);
    }
    fclose(fp);
    if (numQubits == -1)
    {
        fprintf(stderr,"unable to determine number of qubits\n");
        return 0;
    }
    return 1;
}



void LoadNuclearEnergy(realNumType& NuclearEnergy, std::string filePath)
{
    FILE* fp = fopen((filePath + "_Nuclear_Energy.dat").c_str(),"r");
    if(NULL == fp)
    {
        fprintf(stderr,"\nError in opening file. ");
        fprintf(stderr,"fileGiven: %s\n",(filePath + "_Nuclear_Energy.dat").c_str());
        NuclearEnergy = 0;
        return;
    }
    else
    {
        int ret = fscanf(fp, realNumTypeCode ,&NuclearEnergy);
        if (ret <=0)
            fprintf(stderr, "Error reading Nuclear Energy\n");
    }
    fclose(fp);
}

