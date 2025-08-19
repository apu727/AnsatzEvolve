/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "tupsquantities.h"
#include "diis.h"
#include "fusedevolve.h"
#include "logger.h"
#include "threadpool.h"
#include <iostream>

char countActiveBits(uint32_t state)
{
    char ret = 0;
    while(state != 0)
    {
        if (state & 1)
            ret +=1;
        state >>=1;
    }
    return ret;
}

void writeMatrix(std::string filename, Matrix<double>::EigenMatrix &Mat)
{
    FILE* fp = fopen((filename + ".Matbin").c_str(),"wb");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fwrite(&(Mat(i,j)),sizeof(Mat(0,0)),1,fp);
        }
    }
    fclose(fp);
    fp = fopen((filename + ".Matcsv").c_str(),"w");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fprintf(fp,"%.10lg,",Mat(i,j));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

}
void writeMatrix(std::string filename, Matrix<long double>::EigenMatrix &Mat)
{
    FILE* fp = fopen((filename + ".LMatbin").c_str(),"wb");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fwrite(&(Mat(i,j)),sizeof(Mat(0,0)),1,fp);
        }
    }
    fclose(fp);
    fp = fopen((filename + ".LMatcsv").c_str(),"w");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fprintf(fp,"%.10Lg,",Mat(i,j));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);

}

void writeMatrix(std::string filename, Matrix<std::complex<double>>::EigenMatrix &Mat)
{
    FILE* fp = fopen((filename + ".CMatbin").c_str(),"wb");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fwrite(&(Mat(i,j)),sizeof(Mat(0,0)),1,fp);
        }
    }
    fclose(fp);
    fp = fopen((filename + ".CMatcsv").c_str(),"w");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fprintf(fp,"%.10lg+%.10lgj,",real(Mat(i,j)),imag(Mat(i,j)));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}
void writeMatrix(std::string filename, Matrix<std::complex<long double>>::EigenMatrix &Mat)
{
    FILE* fp = fopen((filename + ".LCMatbin").c_str(),"wb");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fwrite(&(Mat(i,j)),sizeof(Mat(0,0)),1,fp);
        }
    }
    fclose(fp);
    fp = fopen((filename + ".LCMatcsv").c_str(),"w");
    if (!fp)
        return;
    for (long int i = 0; i < Mat.rows(); i++)
    {
        for (long int j = 0; j < Mat.cols(); j++)
        {
            fprintf(fp,"%.10Lg+%.10Lgj,",real(Mat(i,j)),imag(Mat(i,j)));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

TUPSQuantities::TUPSQuantities(std::shared_ptr<HamiltonianMatrix<realNumType,numType>> Ham, std::vector<std::pair<int,realNumType>> order,
                               int numberOfUniqueParameters, realNumType NuclearEnergy, std::string runPath,  FILE* logfile)
{
    m_file = logfile;
    if (m_file == nullptr)
        m_file = stdout;

    m_Ham = Ham;
    // m_Ham.copy(Ham);
    // m_HamEm = Ham;
    m_NuclearEnergy = NuclearEnergy;
    m_numberOfUniqueParameters = numberOfUniqueParameters;
    m_runPath = runPath;



    buildCompressionMatrices(numberOfUniqueParameters, order, m_deCompressMatrix,m_compressMatrix);
}

void TUPSQuantities::writeProperties(std::shared_ptr<stateAnsatz> myAnsatz, std::shared_ptr<FusedEvolve> FE, std::vector<std::vector<ansatz::rotationElement>>& rotationPaths)
{
    bool useFusedEvolve = false;
    if (!myAnsatz)
        useFusedEvolve = true;
    // m_HamEm = m_Ham;
    /* calculate the energy for all rotation paths*/
    std::vector<realNumType> Energies(rotationPaths.size());
    std::vector<realNumType> RealEnergies(rotationPaths.size());
    std::vector<realNumType> EnergiesAndNucEnergy(rotationPaths.size());

    std::vector<realNumType> NumberOfNegativeHessianEValues(rotationPaths.size());
    std::vector<realNumType> NumberOfNearZeroHessianEValues(rotationPaths.size());
    std::vector<realNumType> NumberOfPositiveHessianEValues(rotationPaths.size());

    std::vector<realNumType> NumberOfNegativeHessianDiagValues(rotationPaths.size());
    std::vector<realNumType> NumberOfNearZeroHessianDiagValues(rotationPaths.size());
    std::vector<realNumType> NumberOfPositiveHessianDiagValues(rotationPaths.size());

    std::vector<realNumType> NumberOfPositiveMetricEValues(rotationPaths.size());
    std::vector<realNumType> NumberOfZeroMetricEValues(rotationPaths.size());
    std::vector<realNumType> NumberOfZeroMetricDiagonalValues(rotationPaths.size());

    std::vector<realNumType> NormOfGradVector(rotationPaths.size());
    Matrix<realNumType>::EigenMatrix FrechetDistance(rotationPaths.size(),rotationPaths.size());
    //Jacobian?

    vector<numType> temp; //temporary
    vector<numType> dest;



    for (size_t rpIndex = 0; rpIndex < rotationPaths.size(); rpIndex++)
    {
        fprintf(stderr,"On Path%zu\n",rpIndex);
        // for (size_t rpIndex2 = 0; rpIndex2 <= rpIndex; rpIndex2++)
        // {
        //     FrechetDistance(rpIndex,rpIndex2) = computeFrechetDistanceBetweenPaths(myAnsatz,FE,rotationPaths[rpIndex],rotationPaths[rpIndex2]);
        //     FrechetDistance(rpIndex2,rpIndex) = FrechetDistance(rpIndex,rpIndex2);
        // }
        if (!useFusedEvolve)
        {
            myAnsatz->setCalculateFirstDerivatives(true);
            myAnsatz->setCalculateSecondDerivatives(false);
            myAnsatz->resetPath();
            const std::vector<ansatz::rotationElement> &rp = rotationPaths[rpIndex];
            for (auto rpe : rp)
            {
                myAnsatz->addRotation(rpe.first,rpe.second);
                //todo calculate quantities along path?
            }
        }
        //myAnsatz->calcRotationAlongPath(rp,dest,start);
        //writeVector("dest.csv",dest,*lie);

        std::vector<realNumType> anglesV(rotationPaths[rpIndex].size());
        if (useFusedEvolve)
        {
            std::transform(rotationPaths[rpIndex].begin(),rotationPaths[rpIndex].end(),anglesV.begin(),[](const ansatz::rotationElement& r){return r.second;});
            FE->evolve(dest,anglesV);
        }
        else
            dest.copy(myAnsatz->getVec());
        std::vector<vector<numType>> derivTangentSpace;

        if (!useFusedEvolve)
        {
            const std::vector<vector<numType>>&  ref = myAnsatz->getDerivTangentSpace();
            for (const auto& r : ref)
            {
                derivTangentSpace.emplace_back();
                derivTangentSpace.back().copy(r);
            }
        }

        /* some notation:
         * quantities in `compressed' notation after taking into account which angles are equivalent are denoted by the greek subscripts \mu \nu
         * in uncompressed notation with every exponential having a free angle: latin subscripts i,j
         * coefficients of a ket in a basis e.g. \ket{\Psi} in the statevector basis have the subscripts a,b
         *
         * So \frac{d}{d \theta_\mu} is a derivative with respect to restricted angles.
         * H_{ij} is the Hessian with each angle acting independently.
         * v_a = \ket{\Psi} is also valid
         */

        // derivTangentSpaceEM_{ai} = \frac{d}{d\theta_i}\ket{\psi}

        vector<numType>::EigenVector destEM = dest;
        //Debugging if it is a phase away from the real eigenvector
        // vector<realNumType>::EigenVector destArg(destEM.rows());
        // for (long i = 0; i < destArg.rows(); i++ )
        // {
        //     destArg[i] = atan2(destEM[i].imag(),destEM[i].real());
        // }

        /* g_{\mu\nu} = \braket{\eta_\mu | \eta_\nu}
         * Where \ket{\eta_\mu} = \frac{d}{d \theta_\mu} \ket{\Psi}
         *
         * Some \theta_mu are fixed with respect to each other. we have f(x_1,x_2,x_3...etc)
         * and derivTangentSpaceEM_{ij} =  \frac{d}{d\x_j} f_i(x_1,x_2,x_3...etc)
         * Therefore \frac{d}{d\theta_1}f(x_1,x_2,x_3...etc) = \frac{d}{d\x_1}f(x_1,x_2...etc) + \frac{d}{d\x_2}f(x_1,x_2...etc).
         * We have used \frac{dx_1}{d\theta_1} = 1
         * i.e. chain rule.
        */

        //derivTangentSpaceEMCondensed_{a\mu} = compressMatrix_{\mu,i} derivTangentSpaceEM_{a,i}


        //g_{\mu\nu} = derivTangentSpaceEMCondensed_{a\mu} derivTangentSpaceEMCondensed_{a\nu}


        //gradVector_{\mu} = 2 * \braket{\psi | H | \frac{d}{d\theta_\mu}\psi}
        vector<realNumType> gradVectorCalc;



        // H_{\mu\nu}: = \frac{dx_j}{d\theta_nu} \frac{dx_i}{d\theta_mu} H_{ij} (for real vectors)
        // H_{ij}      = \braket{\psi | H | \frac{d^2}{dx_i d x_j} \psi } + \braket{\frac{d^2}{dx_i d x_j} \psi | H | \psi} +
        //             + \braket{\frac{d}{d x_i}\psi | H | \frac{d}{d x_j}\psi} + \braket{\frac{d}{d x_j}\psi | H | \frac{d}{d x_i}\psi}

        //secondDerivTensor_{ija} = \frac{d}{d x_i}\frac{d}{d x_j}\ket{\Psi} {Note j <= i the rest is via symmetry}
        // const std::vector<std::vector<vector<numType>>>& secondDerivTensor = myAnsatz->getSecondDerivTensor();
        // size_t iSize = secondDerivTensor.size();
        // size_t jSize = secondDerivTensor.back().size();
        //Hessian_{ij}

        Matrix<realNumType>::EigenMatrix Hmunu(m_numberOfUniqueParameters,m_numberOfUniqueParameters); //uninitialized by default
        Matrix<numType>::EigenMatrix derivTangentSpaceEM;
        if (useFusedEvolve)
        {
            FE->evolveHessian(Hmunu,gradVectorCalc,anglesV,&derivTangentSpaceEM,&Energies[rpIndex]);
            vector<realNumType> gradVectorCalc2;
            FE->evolveDerivative(dest,gradVectorCalc2,anglesV);
            logger().log("1.1",(gradVectorCalc.dot(gradVectorCalc)));
            logger().log("1.2",(gradVectorCalc.dot(gradVectorCalc2)));
            logger().log("2.2",(gradVectorCalc2.dot(gradVectorCalc2)));
        }
        else
        {
            // vector<realNumType> temp;
            myAnsatz->getHessianAndDerivative(m_Ham,Hmunu,gradVectorCalc,&m_compressMatrix);
        }

        if (!useFusedEvolve)
        {
            derivTangentSpaceEM = convert(derivTangentSpace).transpose();
        }
        Matrix<numType>::EigenMatrix derivTangentSpaceEMCondensed =  derivTangentSpaceEM * m_compressMatrix.transpose();
        Matrix<numType>::EigenMatrix metricTensor = (derivTangentSpaceEMCondensed.adjoint() * derivTangentSpaceEMCondensed).real();

        vector<realNumType>::EigenVector gradVector;
        if (!useFusedEvolve)
            gradVector = m_compressMatrix * (vector<realNumType>::EigenVector)gradVectorCalc;
        else
            gradVector = gradVectorCalc;


        writeMatrix(m_runPath + "_Path_" + std::to_string(rpIndex) + "_Hessian",Hmunu);
        if (!useFusedEvolve)
            writeMatrix(m_runPath + "_Path_" + std::to_string(rpIndex) + "_Metric",metricTensor);

        Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> esH(Hmunu,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<std::complex<realNumType>>::EigenVector hessianEigVal = esH.eigenvalues();
        auto hessianEigVec = esH.eigenvectors();
        vector<std::complex<realNumType>>::EigenVector hessianDiagVals = Hmunu.diagonal();

        Eigen::SelfAdjointEigenSolver<Matrix<numType>::EigenMatrix> esM(metricTensor,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<std::complex<realNumType>>::EigenVector metricEigVal = esM.eigenvalues();

        auto metricEigenVectors = esM.eigenvectors();
        vector<std::complex<realNumType>>::EigenVector metricDiagVals = metricTensor.diagonal();


        std::vector<vector<std::complex<realNumType>>::EigenVector> metricZeroEigenVectors;

        // Energies[rpIndex] = m_Ham.braket(dest, dest, &temp);
        // logger().log("Mag2",dest.dot(dest));
        if (!useFusedEvolve)
            Energies[rpIndex] = m_Ham->apply(dest,temp).dot(dest);

        {
            vector<realNumType> r = dest.real();
            vector<realNumType> hr = m_Ham->apply(dest).real();
            RealEnergies[rpIndex] = hr.dot(r);
            RealEnergies[rpIndex] /= r.dot(r);
        }
        EnergiesAndNucEnergy[rpIndex] = Energies[rpIndex]+m_NuclearEnergy;

        // es.compute(Hij,true);
        // for (int i = 0; i < 15; i++)
        // {
        //     vector<std::complex<realNumType>>::EigenVector offendingEigVec = es.eigenvectors().col(i);
        //     fprintf(stderr,"HessianEigVal: %lg\n",es.eigenvalues()[i].real());
        //     calculateNumericalSecondDerivative(rp, Energies[rpIndex], Ham, offendingEigVec, myAnsatz);
        // }

        realNumType zeroThreshold = 1e-10;

        for (long int i = 0; i < hessianEigVal.rows(); i++)
        {
            auto he  = hessianEigVal[i];
            realNumType e = he.real();
            if (e > zeroThreshold)
                NumberOfPositiveHessianEValues[rpIndex]++;
            else if (e < -zeroThreshold)
            {
                NumberOfNegativeHessianEValues[rpIndex]++;
                fprintf(stderr,"NHE: " realNumTypeCode ", Path:%zu\n",e,rpIndex);
                const vector<numType>::EigenVector &evCondensed = hessianEigVec.col(i);
                vector<numType>::EigenVector ev(evCondensed.rows());
                ev = m_deCompressMatrix * evCondensed;
                // calculateNumericalSecondDerivative(rp,Energies[rpIndex],m_Ham,ev,myAnsatz);

            }
            else if (e >= -zeroThreshold && e <= zeroThreshold)
                NumberOfNearZeroHessianEValues[rpIndex]++;
            else
                fprintf(stderr,"Problem with HEigenvalue: " realNumTypeCode "\n",e);
            if (std::fabs(he.imag()) > zeroThreshold)
                fprintf(stderr,"Problem with Imag HEigenvalue: (" realNumTypeCode "," realNumTypeCode ")\n",he.real(),he.imag());
        }

        for (auto he : hessianDiagVals)
        {
            realNumType e = he.real();
            if (e > zeroThreshold)
                NumberOfPositiveHessianDiagValues[rpIndex]++;
            else if (e < -zeroThreshold)
                NumberOfNegativeHessianDiagValues[rpIndex]++;
            else if (e >= -zeroThreshold && e <= zeroThreshold)
                NumberOfNearZeroHessianDiagValues[rpIndex]++;
            else
                fprintf(stderr,"Problem with HDiagvalue: " realNumTypeCode "\n",e);
            if (std::fabs(he.imag()) > zeroThreshold)
                fprintf(stderr,"Problem with Imag HDiagvalue: (" realNumTypeCode "," realNumTypeCode ")\n",he.real(),he.imag());
        }



        for (long int i =0; i < metricEigVal.rows();i++)
        {
            auto me = metricEigVal[i];
            realNumType e = me.real();
            if (e > zeroThreshold)
                NumberOfPositiveMetricEValues[rpIndex]++;
            else if (e >= -zeroThreshold && e <= zeroThreshold)
            {
                NumberOfZeroMetricEValues[rpIndex]++;
                metricZeroEigenVectors.push_back(metricEigenVectors.col(i));
            }
            else
                fprintf(stderr,"Problem with MEigenvalue: " realNumTypeCode "\n",e);
            if (std::fabs(me.imag()) > zeroThreshold)
                fprintf(stderr,"Problem with Imag MEigenvalue: (" realNumTypeCode "," realNumTypeCode ")\n",me.real(),me.imag());
        }
        for (auto me : metricDiagVals)
        {
            realNumType e = me.real();
            if (e >= -zeroThreshold && e <= zeroThreshold)
                NumberOfZeroMetricDiagonalValues[rpIndex]++;
            if (std::fabs(me.imag()) > zeroThreshold)
                fprintf(stderr,"Problem with Imag MDiagvalue: (" realNumTypeCode "," realNumTypeCode ")\n",me.real(),me.imag());
        }

        NormOfGradVector[rpIndex] = gradVector.norm();
        realNumType Mag = dest.dot(dest);
        if (std::fabs(Mag-1.) > 1e-5)
            fprintf(stderr,"Magnitude for path %zu is not 1 but " realNumTypeCode "\n",rpIndex,Mag);
    }

    //numType HFEnergy = Ham.braket(start,start,&temp);

    writeMatrix(m_runPath + "_FrechetDistance",FrechetDistance);

    printOutputHeaders(rotationPaths.size()-1);

    printOutputLine(Energies,"Elec. Energy");
    printOutputLine(RealEnergies,"Real Elec. Energy");
    printOutputLine(EnergiesAndNucEnergy,"Elec. + Nuc. Energy");

    printOutputLine(NumberOfNegativeHessianEValues,"NumberOfNegativeHessianEValues");
    printOutputLine(NumberOfNearZeroHessianEValues,"NumberOfNearZeroHessianEValues");
    printOutputLine(NumberOfPositiveHessianEValues,"NumberOfPositiveHessianEValues");

    printOutputLine(NumberOfNegativeHessianDiagValues,"NumberOfNegativeHessianDiagValues");
    printOutputLine(NumberOfNearZeroHessianDiagValues,"NumberOfNearZeroHessianDiagValues");
    printOutputLine(NumberOfPositiveHessianDiagValues,"NumberOfPositiveHessianDiagValues");

    printOutputLine(NumberOfPositiveMetricEValues,"NumberOfPositiveMetricEValues");
    printOutputLine(NumberOfZeroMetricEValues,"NumberOfZeroMetricEValues");
    printOutputLine(NumberOfZeroMetricDiagonalValues,"NumberOfZeroMetricDiagonalValues");


    printOutputLine(NormOfGradVector,"NormOfGradVector");
    if (m_Ham->rows() < 100000 && false)
    {
        fprintf(stderr,"Finding lowest EigenValue\n");
        vector<numType> start;
        start.copy(dest);
        vector<numType> next;
        m_Ham->apply(start,next);
        while(abs(next.dot(start) + 1) > 1e-10)
        {
            static_cast<Matrix<numType>&>(start) = std::move(next);
            m_Ham->apply(start,next);
            // fprintf(stderr,"NExtNorm: %lg\n", next.norm());
            next.normalize();
            // fprintf(stderr,"error: %lg\n", abs(next.dot(start) + 1));
        }
        fprintf(stderr, "Largest E Value,%.16lg",(m_Ham->apply(next).dot(next)));

        // auto TrueEigenValues = Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> (m_HamEm,Eigen::EigenvaluesOnly).eigenvalues();
        // fprintf(stderr,"TrueEigenValues:\n");
        // for (long i = 0; i < TrueEigenValues.rows() && i < 10; i++)
        //     fprintf(stderr,"%20.10lg",TrueEigenValues[i]);
        // fprintf(stderr,"\n");
    }
    else
        logger().log("Hamiltonian too big to diagonalise here");
}

void TUPSQuantities::calculateNumericalSecondDerivative(const std::vector<ansatz::rotationElement> &rp, realNumType startEnergy, const sparseMatrix<realNumType,numType> &Ham, const vector<numType>::EigenVector &direction, stateAnsatz *myAnsatz)
{
    //for sanity checking.
    fprintf(stderr,"Varying energies: starting at " realNumTypeCode "\n",startEnergy);

    vector<numType> temp;
    realNumType offsetSize = 0.001;
    realNumType E[10];
    for (int i = -5; i < 5;i++)
    {
        std::vector<ansatz::rotationElement> rpNew = rp;
        for (size_t idx = 0; idx < rpNew.size(); idx++)
        {
            rpNew[idx].second += (realNumType)i*std::real(direction[idx])*offsetSize;
        }
        myAnsatz->calcRotationAlongPath(rpNew,temp,myAnsatz->getStart());
        E[i+5] = Ham.braket(temp, temp);
        fprintf(stderr,"%15.10lg ",(double)E[i+5]);
    }
    fprintf(stderr,"\n");
    realNumType DE[9];
    for (int i = 0; i < 9; i++)
    {
        DE[i] = (E[i+1]-E[i])/offsetSize;
    }
    realNumType DDE[8];
    for (int i = 0; i < 8; i++)
    {
        DDE[i] = (DE[i+1]-DE[i])/offsetSize;
        fprintf(stderr,"%15.10lg ",(double)DDE[i]);
    }
    fprintf(stderr,"\n");
    fprintf(stderr,"\n");
}

void TUPSQuantities::buildCompressionMatrices(int numberOfUniqueParameters, std::vector<std::pair<int,realNumType>> order, sparseMatrix<realNumType,numType>::EigenSparseMatrix &deCompressMatrix, sparseMatrix<realNumType,numType>::EigenSparseMatrix &compressMatrix)
{
    //D_{i \mu} = \frac{d x_i}{d \theta_\mu}
    deCompressMatrix.resize(order.size(),numberOfUniqueParameters);
    for (size_t i = 0; i < order.size(); i++)
    {
        deCompressMatrix.insert(i,order[i].first) = order[i].second;
    }
    //C_{\mu i} = \frac{d x_i}{d \theta_\mu}
    compressMatrix = deCompressMatrix.transpose();
    //Note compressMatrix * compressMatrix =/= Id
    m_normCompressMatrix = compressMatrix;
    m_normCompressMatrix.setZero();
    for (long i = 0; i < compressMatrix.rows(); i++)
    {
        for (long j = 0; j < compressMatrix.cols(); j++)
        {
            if (compressMatrix.coeff(i,j) != 0)
            {
                m_normCompressMatrix.coeffRef(i,j) = 1/compressMatrix.coeff(i,j);
                break;
            }
        }
    }
}

void TUPSQuantities::asyncHij(const sparseMatrix<realNumType,numType> &Ham, const std::vector<std::vector<vector<numType>>> &secondDerivTensor,
                              const std::vector<vector<numType> > &derivTangentSpace, Matrix<realNumType>::EigenMatrix &Hij, const vector<numType> &dest, const size_t iSize)
{
    std::atomic_int finishCount = 0;

    auto multiply = [&](const size_t startI, const size_t endI)
    {
        vector<numType> temp;
        for (size_t i = startI; i < endI; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                Hij(i,j) = 2*Ham.braket(dest,secondDerivTensor[i][j],&temp);
                Hij(i,j) += 2*Ham.braket(derivTangentSpace[i],derivTangentSpace[j],&temp);
                Hij(j,i) = Hij(i,j);

            }
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const size_t stepSize = 2;std::max((size_t)iSize/NUM_CORES,1ul);
    std::vector<std::future<void>> futures;
    auto& pool = threadpool::getInstance(NUM_CORES);

    for (size_t i = 0; i < iSize; i += stepSize)
        futures.push_back(pool.queueWork([&,i](){multiply(i, std::min(i+stepSize,iSize));}));

    for (auto &f: futures)
        f.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

void TUPSQuantities::runNewtonMethod(FusedEvolve *myAnsatz,std::vector<realNumType> &angles,bool avoidNegativeHessianValues)
{
    // realNumType maxStepSize = 0.1;
    int maxStepCount = 500;
    realNumType zeroThreshold = 1e-10;

    int count = maxStepCount;
    size_t HessianEvals =0;
    size_t EnergyEvals = 0;
    vector<numType> temp;
    while(count-- > 0)
    {
        auto start = std::chrono::high_resolution_clock::now();

        // vector<numType> dest;
        // myAnsatz->evolve(dest,angles);

        // vector<numType>::EigenVector destEM = dest;
        // const std::vector<vector<numType>>& derivTangentSpace = myAnsatz->getDerivTangentSpace();


        // Matrix<numType>::EigenMatrix derivTangentSpaceEM = convert(derivTangentSpace).transpose();


        /* g_{\mu\nu} = \braket{\eta_\mu | \eta_\nu}
         * Where \ket{\eta_\mu} = \frac{d}{d \theta_\mu} \ket{\Psi}
         *
         * Some \theta_mu are fixed with respect to each other. we have f(x_1,x_2,x_3...etc)
         * and derivTangentSpaceEM_{ij} =  \frac{d}{d\x_j} f_i(x_1,x_2,x_3...etc)
         * Therefore \frac{d}{d\theta_1}f(x_1,x_2,x_3...etc) = \frac{d}{d\x_1}f(x_1,x_2...etc) + \frac{d}{d\x_2}f(x_1,x_2...etc).
         * We have used \frac{dx_1}{d\theta_1} = 1
         * i.e. chain rule.
        */

        //derivTangentSpaceEMCondensed_{a\mu} = compressMatrix_{\mu,i} derivTangentSpaceEM_{a,i}
        // Matrix<numType>::EigenMatrix derivTangentSpaceEMCondensed =  derivTangentSpaceEM * m_compressMatrix.transpose();

        //gradVector_{\mu} = 2 * \braket{\psi | H | \frac{d}{d\theta_\mu}\psi}

        // vector<realNumType>::EigenVector gradVector_mu = 2*(destEM.adjoint()*HamEm*derivTangentSpaceEMCondensed).real();
        // vector<realNumType>::EigenVector gradVector_i = m_deCompressMatrix * gradVector_mu;

        vector<realNumType> gradVectorCalc;
        realNumType Energy;
        // Matrix<realNumType>::EigenMatrix Hij;
        Matrix<realNumType>::EigenMatrix Hmunu;
        // myAnsatz->getHessianAndDerivative(&m_Ham,Hmunu,gradVectorCalc,&m_compressMatrix);
        myAnsatz->evolveHessian(Hmunu,gradVectorCalc,angles,nullptr,&Energy);
        HessianEvals++;
        EnergyEvals++;
        Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> gradVector_mu(&gradVectorCalc[0],gradVectorCalc.size(),1);

        // vector<realNumType>::EigenVector gradVector_mu = m_compressMatrix * gradVectorCalcEm;
        // realNumType Energy = (destEM.adjoint() * HamEm * destEM).real()(0,0);
        // realNumType Energy = myAnsatz->getEnergy(dest);//m_Ham.braket(dest,dest,&temp);


        /* some notation:
             * quantities in `compressed' notation after taking into account which angles are equivalent are denoted by the greek subscripts \mu \nu
             * in uncompressed notation with every exponential having a free angle: latin subscripts i,j
             * coefficients of a ket in a basis e.g. \ket{\Psi} in the statevector basis have the subscripts a,b
             *
             * So \frac{d}{d \theta_\mu} is a derivative with respect to restricted angles.
             * H_{ij} is the Hessian with each angle acting independently.
             * v_a = \ket{\Psi} is also valid
             */


        // H_{\mu\nu}: = \frac{dx_j}{d\theta_nu} \frac{dx_i}{d\theta_mu} H_{ij} (for real vectors)
        // H_{ij}      = \braket{\psi | H | \frac{d^2}{dx_i d x_j} \psi } + \braket{\frac{d^2}{dx_i d x_j} \psi | H | \psi} +
        //             + \braket{\frac{d}{d x_i}\psi | H | \frac{d}{d x_j}\psi} + \braket{\frac{d}{d x_j}\psi | H | \frac{d}{d x_i}\psi}

        //secondDerivTensor_{ija} = \frac{d}{d x_i}\frac{d}{d x_j}\ket{\Psi} {Note j <= i the rest is via symmetry}
        // const std::vector<std::vector<vector<numType>>>& secondDerivTensor = myAnsatz->getSecondDerivTensor();
        // size_t iSize = secondDerivTensor.size();
        // size_t jSize = secondDerivTensor.back().size();
        //Hessian_{ij}
        // Hij.resize(iSize,jSize);
        //Hessian_{\mu\nu}



        // asyncHij(m_Ham,secondDerivTensor,derivTangentSpace,Hij,dest,iSize);

        // Hmunu = m_compressMatrix * Hij * m_compressMatrix.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> es(Hmunu,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<realNumType>::EigenVector hessianEigVal = es.eigenvalues();
        vector<realNumType>::EigenVector InvhessianEigVal(hessianEigVal.rows());
        auto hessianEigVec = es.eigenvectors();




        vector<realNumType>::EigenVector updateAngles(angles.size());
        updateAngles.setZero(angles.size());

        vector<realNumType>::EigenVector negativeEigenValueDirections(hessianEigVal.rows());
        negativeEigenValueDirections.setZero();

        auto curveDamp = [](realNumType v){/*if (v > 1e6) return 1e6; else */return v;};
        bool allPositiveEigenvalues = true;
        for (long int i = 0; i < hessianEigVal.rows(); i++)
        {
            if(abs(hessianEigVal[i]) >= zeroThreshold)
            {
                if (avoidNegativeHessianValues)
                {
                    InvhessianEigVal[i] = curveDamp(abs(1./(hessianEigVal[i])));
                    /*if (hessianEigVal[i].real() < 0 && abs(gradVector_mu.dot(hessianEigVec.col(i))) < zeroThreshold)
                        negativeEigenValueDirections += hessianEigVec.col(i) * 0.1;*/
                    // if (hessianEigVal[i].real() < 0)
                    // {
                    //     InvhessianEigVal[i] = 0;
                    //     negativeEigenValueDirections += hessianEigVec.col(i) * 0.1;
                    // }
                }
                else
                    InvhessianEigVal[i] = curveDamp(1./(hessianEigVal[i]));
                if (hessianEigVal[i] < 0)
                    allPositiveEigenvalues = false;
            }
            else
            {
                InvhessianEigVal[i] = 0;
            }

        }
        // Matrix<std::complex<realNumType>>::EigenMatrix testingEigenDecompose = (hessianEigVec * hessianEigVal.asDiagonal()*hessianEigVec.adjoint()) - Hmunu;
        vector<std::complex<realNumType>>::EigenVector testingUpdateAngles = - hessianEigVec * InvhessianEigVal.asDiagonal()*hessianEigVec.adjoint() * gradVector_mu + negativeEigenValueDirections;
        updateAngles = m_deCompressMatrix * testingUpdateAngles.real();

        if (/*gradVector_mu.norm() > 1e-8 ||*/ !allPositiveEigenvalues && false)
        {// from Numerical Optimization, 2nd Ed. Springer, 2006.
            constexpr size_t maxSteps = 10;
            size_t stepCount = 0;
            std::vector<realNumType> anglesCopy = angles;
            vector<realNumType> direction;
            {
                Eigen::VectorXd dir = testingUpdateAngles.real();
                direction.copyFromBuffer(dir.data(),dir.rows());
            }
            long double alpha1 = 1;
            long double alpha0 = 0;
            long double newAlpha = alpha1;
            for (size_t i = 0; i < angles.size();i++)
                angles[i] += alpha1*updateAngles[i];

            realNumType tempEnergy;
            long double energy1;
            long double energy0 = Energy;
            long double energyTrial;
            vector<numType> trial;
            vector<numType> deriv(testingUpdateAngles.rows());

            long double directionalDeriv1;
            long double directionalDeriv0 =  direction.dot(gradVectorCalc);
            long double directionalDerivAt0 = direction.dot(gradVectorCalc);
            long double c1 = 1e-4;
            long double c2 = 1e-3;
            long double directionalDerivTrial;

            myAnsatz->evolve(trial,angles);
            myAnsatz->evolveDerivative(trial,deriv,angles,&tempEnergy);
            energy1 = tempEnergy;
            // energy1 = myAnsatz->getEnergy(trial);
            EnergyEvals++;
            directionalDeriv1 = direction.dot(deriv);
            bool doesntSatisfySufficientDecrease = energy1 > Energy + c1*alpha1*directionalDerivAt0;
            while (doesntSatisfySufficientDecrease)
            {
                stepCount++;
                long double d1 = directionalDeriv0 + directionalDeriv1 - 3*((energy0 - energy1)/(alpha0 - alpha1));
                long double d2 = ((alpha1 - alpha0) > 0 ? 1 : -1)*std::sqrt(d1*d1-directionalDeriv0*directionalDeriv1);
                newAlpha = alpha1 - (alpha1-alpha0)*((directionalDeriv1 + d2 - d1)/(directionalDeriv1 - directionalDeriv0 + 2*d2));

                bool tobreak = false;
                if (abs(alpha1-alpha0) < 1e-10)
                    tobreak = true;

                if (!std::isfinite(newAlpha))
                {
                    logger().log("alpha1",alpha1);
                    logger().log("alpha0",alpha0);
                    logger().log("1/(alpha1-alpha0)",1./(alpha0 - alpha1));
                    logger().log("newAlpha",newAlpha);
                    logger().log("d1",d1);
                    logger().log("d2",d2);
                    logger().log("d1*d1-directionalDeriv0*directionalDeriv1",d1*d1-directionalDeriv0*directionalDeriv1);
                }
                if (newAlpha > std::max(alpha1,alpha0))
                { //interpolation suggested a step that takes us above alpha1, alpha1 may not satisfy the decrease condition
                    if (alpha1 > alpha0)
                    {
                        if (energy1 <= Energy + c1*alpha1*directionalDerivAt0)
                            newAlpha = alpha1;
                        else
                            newAlpha = alpha0; //alpha 0 is either 0 or satisfies the decrease conditions
                    }
                    if (alpha0 > alpha1)
                    {
                        if (energy0 <= Energy + c1*alpha0*directionalDerivAt0)
                            newAlpha = alpha0;
                        else
                            newAlpha = alpha1; //alpha 1 is either 0 or satisfies the decrease conditions
                    }
                    tobreak = true;
                }
                if (newAlpha < std::min(alpha1,alpha0))
                {
                    newAlpha = std::min(alpha1,alpha0);
                    tobreak = true;
                }

                if (abs(newAlpha) < 1e-3) // without doing complicated things this should be safe ish
                {
                    newAlpha = 1e-3; // force progress
                    tobreak = true;
                }
                if (stepCount >= maxSteps)
                    tobreak = true;

                angles = anglesCopy;
                for (size_t i = 0; i < angles.size();i++)
                    angles[i] += newAlpha*updateAngles[i];
                if (tobreak)
                    break;



                myAnsatz->evolve(trial,angles);
                myAnsatz->evolveDerivative(trial,deriv,angles,&tempEnergy);
                energyTrial = tempEnergy;
                // energyTrial = myAnsatz->getEnergy(trial);
                EnergyEvals++;
                directionalDerivTrial = direction.dot(deriv);

                if (energyTrial > Energy + c1*newAlpha*directionalDerivAt0 || energyTrial >= energy0)
                {
                    alpha1 = newAlpha;
                    directionalDeriv1 = directionalDerivTrial;
                    energy1 = energyTrial;
                }
                else
                {
                    if (abs(directionalDerivTrial) <= -c2*directionalDerivAt0)
                        break; // set angles to newAlpha
                    if (directionalDerivTrial*(alpha1-alpha0) >= 0)
                    {
                        alpha1 = alpha0;
                        directionalDeriv1 = directionalDeriv0;
                        energy1 = energy0;
                    }
                    alpha0 = newAlpha;
                    directionalDeriv0 = directionalDerivTrial;
                    energy0 = energyTrial;

                }
            }
            logger().log("newAlpha", (double)newAlpha);

        }
        else
        {
            // E = E_0 + \partial_i x_i + 0.5 \partial_i \partial_j x_i x_j + 1/6 \partial_i \partial_j \partial_k x_i x_j x_k
            // The newton raphson step is:
            // \partial E(x) / \partial_i  = \partial_i + \partial_i \partial_j x_j => solve for x_j
            // Given a direction x_j, we can do a third order correction to the step size
            // let x'_j = t x_j for t > 0. Therefore taking a directional deriviative of E along this direction: \partial E / \partial t = \partial E(x)/\partial_i x_i
            // 0 = x_i \partial_i + t x_i  \partial_i \partial_j x_j + 0.5 t^2 x_i \partial_i \partial_j \partial_k x_j x_k
            // This is a quadratic
            // 0 = at^2 + bt + c
            // Where a = 0.5 x_i \partial_i \partial_j \partial_k x_j x_k
            //       b = x_i  \partial_i \partial_j x_j
            //       c = x_i \partial_i
            // at t = 1. bt+c = 0 by virtue of this being the newton step.
            // Therefore \partial E(x + (t=1))/\partial_i x_i = a
            // There will in general be two solutions. But one will be nearest t=1. This is the most positive solution. Plot and check
            // Therefore the new guess is t = (-b + sqrt(b^2-4ac))/2a. eq (1)
            // If the hessian is positive definite then b >= 0. Likewise since x_i is a downhill step c <=0
            // If there are negative eigenvalues then this analysis is not strictly correct since we do not solve the newton step.
            // Therefore b+c =/= 0 for us.
            // so \partial E(x + (t=1))/\partial_i x_i = a + b + c. Since b and c are known this is fine. b will likely still be positive. c negative
            // If there are solutions, go for the positive t one. (move in the downhill direction). if not go for the stationary point (closest to zero).

            // If the hessian is not positive definite then b+c =/= 0. But they are both still valid and indeed the energy can be expanded as
            // E = E_0 + at^3/3 + bt^2/2 + ct
            // The lowest point of this is sought which is either at the end point t = 1. or internal and given by eq (1)
            // We allow it to extrapolate by limiting the valid range to: t \in (0,1.5)
            // If A > 0 and c < 0 then the quadratic has solutions and we go with this, capping it to the acceptable range.
            // If A < 0 then the cubic implies t = infinity is best.
            // If eq (1) has solutions then we choose the smaller of E(t) or E(t_Max) otherwise t = t_max

            constexpr size_t maxSearch = 20;
            size_t searchCount = 0;
            long double tMax = 1.5;
            long double t = 1;
            long double newT = 1;
            std::vector<realNumType> anglesCopy = angles;
            for (size_t i = 0; i < angles.size();i++)
                angles[i] += t*updateAngles[i];

            vector<numType> trialCubic;
            vector<realNumType> trialGradCubic;
            realNumType energyTrial;
            long double a;
            long double b;
            long double c;

            long double E1;
            long double E2;
            long double foundDirectionalDeriv;
            long double initialDirectionalDeriv = gradVector_mu.dot(testingUpdateAngles.real());
            long double c2 = 1e-3;
            bool doingForwardsSteps = false;
            long double lastGoodTFowardSteps = newT;
            do
            {
                searchCount++;
                myAnsatz->evolve(trialCubic,angles);
                myAnsatz->evolveDerivative(trialCubic,trialGradCubic,angles,&energyTrial);
                EnergyEvals++;

                b = testingUpdateAngles.real().transpose() * (Hmunu * testingUpdateAngles.real());
                c = testingUpdateAngles.real().dot(gradVector_mu);
                Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> trialGradMap(&trialGradCubic[0],trialGradCubic.size(),1);

                foundDirectionalDeriv = trialGradMap.dot(testingUpdateAngles.real());
                a = (foundDirectionalDeriv - (b + c))/(t*t);
                b /= t;
                // if (searchCount == 1 && energyTrial < Energy && gradVector_mu.norm() > 1e-8)
                //     doingForwardsSteps = true;


                long double discriminant = b*b-4*a*c;
                if (c >= 0)
                    logger().log("c >= 0", c);
                if (a >= 0)
                {
                    newT = (-b + std::sqrt(discriminant))/(2*a);
                    // energyTrial = Energy + a*t*t*t/3 + b*t*t/2 + c*t;
                }
                else
                {
                    if (discriminant >= 0)
                    {
                        long double t1 = (-b + std::sqrt(discriminant))/(2*a);
                        E1 = a*t1*t1*t1/3 + b*t1*t1/2 + c*t1 + Energy;
                        E2 = a*tMax*tMax*tMax/3 + b*tMax*tMax/2 + c*tMax + Energy;
                        if (E2 < E1)
                        {
                            newT = tMax;
                            // energyTrial = E2;
                        }
                        else
                        {
                            newT = t1;
                            // energyTrial = E1;
                        }
                    }
                    else //monotone decrease up to boundary
                    {
                        newT = tMax;
                        // energyTrial = a*tMax*tMax*tMax/3 + b*tMax*tMax/2 + c*tMax + Energy;
                    }
                }
                bool toBreak = false;

                angles = anglesCopy;
                for (size_t i = 0; i < angles.size();i++)
                    angles[i] += newT*updateAngles[i];
                if (toBreak)
                    break;


                if (energyTrial > Energy && gradVector_mu.norm() > 1e-8)
                {
                    if (doingForwardsSteps)
                    {
                        if (lastGoodTFowardSteps > tMax/2)
                            lastGoodTFowardSteps = tMax/2;
                        t = t/2;
                        tMax = tMax/2;
                        newT = lastGoodTFowardSteps;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += newT*updateAngles[i];
                        logger().log("Recovering Forward");
                        break;
                    }
                    doingForwardsSteps = false;
                    t = t/2;
                    tMax = tMax/2;
                    angles = anglesCopy;
                    for (size_t i = 0; i < angles.size();i++)
                        angles[i] += t*updateAngles[i];
                    if (searchCount > maxSearch)
                        break;
                }
                else
                {
                    if (doingForwardsSteps)
                    {
                        t = 2*t;
                        tMax = 2*tMax;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += t*updateAngles[i];
                        lastGoodTFowardSteps = newT;
                    }
                    else
                    {
                        if (newT > tMax)
                            newT = tMax;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += newT*updateAngles[i];
                        break;
                    }

                }
            }
            while(true);

            // logger().log("tMax",tMax);
            // logger().log("Computed t",t);
            // logger().log("new T",newT);
            // logger().log("Energy", Energy);
            // logger().log("Energy Trial", energyTrial);
            // logger().log("initialDirectionalDeriv",initialDirectionalDeriv);
            // logger().log("foundDirectionalDeriv",foundDirectionalDeriv);
            // if (!(abs(foundDirectionalDeriv) < -c2*initialDirectionalDeriv) && searchCount != 1)
            //     logger().log("Failed to meet wolfe condition",searchCount);

        }







        // for (size_t i = 0; i < angles.size();i++)
        //     angles[i] += updateAngles[i];
        // if (/*gradVector_mu.norm() < 1e-3 &&*/ allPositiveEigenvalues)
        // {
        //     vector<numType> trialCubic;
        //     vector<realNumType> trialGradCubic;
        //     myAnsatz->evolve(trialCubic,angles);
        //     myAnsatz->evolveDerivative(trialCubic,trialGradCubic,angles);
        //     // realNumType b = testingUpdateAngles.real().transpose() * (hessianEigVec * (hessianEigVal.cwiseAbs().asDiagonal()*(hessianEigVec.adjoint() * testingUpdateAngles.real())));
        //     long double b = testingUpdateAngles.real().transpose() * (Hmunu * testingUpdateAngles.real());
        //     long double c = testingUpdateAngles.real().dot(gradVector_mu);
        //     Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> trialGradMap(&trialGradCubic[0],trialGradCubic.size(),1);

        //     long double foundDirectionalDeriv = trialGradMap.dot(testingUpdateAngles.real());
        //     long double a = foundDirectionalDeriv - (b + c);
        //     long double t = 1;
        //     long double discriminant = b*b-4*a*c;
        //     if (b > 0)
        //     {
        //         if (discriminant >= 0)
        //             t = (-b + std::sqrt(discriminant))/(2*a);
        //         else
        //             t = -b/(2*a);
        //     }
        //     for (size_t i = 0; i < angles.size();i++)
        //         angles[i] += (t-1)*updateAngles[i];
        //     logger().log("Computed t",(double)t);
        // }
        // else
        // {

        //     //Implements backtracking
        //     realNumType tBacktrack = 1;
        //     realNumType tBackTrackStepSize = 1;
        //     realNumType EnergyTrial =0;
        //     realNumType lastEnergyTrial = Energy;
        //     int BacktrackCount = 0;
        //     vector<realNumType>::EigenVector updateAnglesCopy = updateAngles;
        //     realNumType biggestAngle  = updateAnglesCopy[0];
        //     for (auto a : updateAnglesCopy)
        //         biggestAngle = std::max(biggestAngle,abs(a));
        //     // logger().log("Biggest Angle at start",biggestAngle);
        //     while(true)
        //     {
        //         vector<numType> trial;
        //         myAnsatz->evolve(trial,angles);
        //         // vector<numType>::EigenVector destEMTrial = trial;
        //         // EnergyTrial = (destEMTrial.adjoint() * HamEm * destEMTrial).real()(0,0);
        //         EnergyTrial = myAnsatz->getEnergy(trial);//m_Ham.braket(trial,trial,&temp);
        //         EnergyEvals++;
        //         if (biggestAngle < 1e-13 || BacktrackCount >= 3)
        //             break;

        //         if (EnergyTrial > lastEnergyTrial)
        //         {//Backtracking
        //             // logger().log("Backtracking",BacktrackCount);
        //             biggestAngle  = abs(updateAnglesCopy[0]/2);
        //             for (size_t i = 0; i < angles.size();i++)
        //             {
        //                 updateAnglesCopy[i] /=2;
        //                 angles[i] -= updateAnglesCopy[i];
        //                 biggestAngle = std::max(biggestAngle,abs(updateAnglesCopy[i]));
        //             }
        //             tBackTrackStepSize /= 2;
        //             tBacktrack -= tBackTrackStepSize;
        //             BacktrackCount++;
        //         }
        //         else
        //         {
        //             break;
        //             // logger().log("Forwardtracking",BacktrackCount);
        //             lastEnergyTrial = EnergyTrial;
        //             biggestAngle  = abs(updateAnglesCopy[0]/2);
        //             for (size_t i = 0; i < angles.size();i++)
        //             {
        //                 updateAnglesCopy[i] /=2;
        //                 angles[i] += updateAnglesCopy[i];
        //                 biggestAngle = std::max(biggestAngle,abs(updateAnglesCopy[i]));
        //             }
        //             tBackTrackStepSize /= 2;
        //             tBacktrack += tBackTrackStepSize;
        //             BacktrackCount++;
        //         }
        //     }
        //     logger().log("Backtracked t",tBacktrack);
        // }


        auto stop = std::chrono::high_resolution_clock::now();
        fprintf(stderr,"Energy: %.10lg GradNorm: " realNumTypeCode " Time (ms): %li, Energy Evals: %zu, Hess Evals: %zu\n",
                Energy,gradVector_mu.norm(),std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(),EnergyEvals,HessianEvals);
        if (gradVector_mu.norm() < 1e-12)
            break;
    }
}

void TUPSQuantities::runNewtonMethodProjected(FusedEvolve *myAnsatz,std::vector<realNumType> &angles, const vector<numType>& psiH, const vector<numType>& prevDest)
{ // Projected energy
    bool avoidNegativeHessianValues = true;
    int maxStepCount = 50;
    realNumType zeroThreshold = 1e-10;

    int count = maxStepCount;
    vector<numType> temp;
    realNumType InvnormOfPsiH =1./std::sqrt(psiH.dot(psiH));
    size_t EnergyEvals = 0;
    size_t HessianEvals = 0;

    while(count-- > 0)
    {
        auto start = std::chrono::high_resolution_clock::now();

        vector<numType> dest;
        myAnsatz->evolve(dest,angles);

        vector<realNumType> EgradVectorCalc;
        Matrix<realNumType>::EigenMatrix HEmunu;

        realNumType Energy;// = dest.dot(psiH);
        myAnsatz->evolveHessianProj(HEmunu,EgradVectorCalc,angles,psiH,nullptr,&Energy);
        HessianEvals++;
        EnergyEvals++;
        //Suppose Eproj = <\psi_0|H\psi>/<\psi_0|\psi>
        // dE/di = <\psi_0|H\d_i psi>/<\psi_0|\psi> - <\psi_0|H\psi><\psi_0|d_i \psi>/<\psi_0|\psi>^2
        //       = G_i/S - E G^s i/S^2
        //       = G_i/S - EProj G^s/S
        // d^2E/didj = <\psi_0|H\d_i d_j psi>/<\psi_0|\psi> -  2<\psi_0|H\d_i psi><\psi_0|d_j \psi>/<\psi_0|\psi>^2 - <\psi_0|H\psi><\psi_0|d_i d_j\psi>/<\psi_0|\psi>^2 - 2<\psi_0|H\psi><\psi_0|d_i \psi><\psi_0|d_j \psi>/<\psi_0|\psi>^3
        //           = H_{ij}/S - 2G_iG^s_j/S^2 - E H^s_{ij}/S^2 - 2 E G^s_i G^s_j /S^3
        //           = H_{ij}/S - (G_iG^s_j + G_jG^s_i) /S^2 - EProj H^s_{ij}/S^2 - 2 EProj G^s_i G^s_j /S^2
        // Matrix<realNumType>::EigenMatrix Smunu; // <\psi_0|d_i d_j\psi>
        // vector<realNumType> SgradVectorCalc; // <\psi_0|d_j \psi>
        // realNumType S;
        // myAnsatz->evolveHessianProj(Smunu,SgradVectorCalc,angles,prevDest,nullptr,&S);
        // HessianEvals++;
        // EnergyEvals++;

        //Eigen quantaties
        // Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> EgradEm(&EgradVectorCalc[0],EgradVectorCalc.size(),1);
        // Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> SgradEm(&SgradVectorCalc[0],SgradVectorCalc.size(),1);
        // realNumType EProj = Energy/S;
        // Eigen::VectorXd gradVector_mu = EgradEm/S - EProj*SgradEm/S;
        // Eigen::MatrixXd Hmunu = HEmunu/S - (EgradEm * SgradEm.transpose() + SgradEm * EgradEm.transpose())/(S*S) - EProj * Smunu/(S*S) - (2*EProj/(S*S))*SgradEm*SgradEm.transpose();


        Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> gradVector_mu(&EgradVectorCalc[0],EgradVectorCalc.size(),1);
        const Eigen::MatrixXd& Hmunu = HEmunu;



        Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> es(Hmunu,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<std::complex<realNumType>>::EigenVector hessianEigVal = es.eigenvalues();
        vector<realNumType>::EigenVector InvhessianEigVal(hessianEigVal.rows());
        auto hessianEigVec = es.eigenvectors();




        vector<realNumType>::EigenVector updateAngles;

        for (long int i = 0; i < hessianEigVal.rows(); i++)
        {
            if(abs(hessianEigVal[i].real()) >= zeroThreshold)
            {
                if (avoidNegativeHessianValues)
                {
                    InvhessianEigVal[i] = abs(1./(hessianEigVal[i].real()));
                }
                else
                    InvhessianEigVal[i] = 1./(hessianEigVal[i].real());
            }
            else
                InvhessianEigVal[i] = 0;

        }
        // Matrix<std::complex<realNumType>>::EigenMatrix testingEigenDecompose = (hessianEigVec * hessianEigVal.asDiagonal()*hessianEigVec.adjoint()) - Hmunu;
        vector<std::complex<realNumType>>::EigenVector testingUpdateAngles = - hessianEigVec * InvhessianEigVal.asDiagonal()*hessianEigVec.adjoint() * gradVector_mu;
        updateAngles = m_deCompressMatrix * testingUpdateAngles.real();
        //Backtrack and cubic interpolate
        {
            constexpr size_t maxSearch = 5;
            size_t searchCount = 0;
            long double tMax = 1.5;
            long double t = 1;
            long double newT = 1;
            std::vector<realNumType> anglesCopy = angles;
            for (size_t i = 0; i < angles.size();i++)
                angles[i] += t*updateAngles[i];

            vector<numType> trialCubic;
            vector<realNumType> trialGradCubic;
            realNumType energyTrial;
            long double a;
            long double b;
            long double c;

            long double E1;
            long double E2;
            long double foundDirectionalDeriv;
            long double initialDirectionalDeriv = gradVector_mu.dot(testingUpdateAngles.real());
            long double c2 = 1e-3;
            bool doingForwardsSteps = false;
            long double lastGoodTFowardSteps = newT;
            do
            {
                searchCount++;
                myAnsatz->evolve(trialCubic,angles);
                myAnsatz->evolveDerivativeProj(trialCubic,trialGradCubic,angles,psiH,&energyTrial);
                EnergyEvals++;

                b = testingUpdateAngles.real().transpose() * (Hmunu * testingUpdateAngles.real());
                c = testingUpdateAngles.real().dot(gradVector_mu);
                Eigen::Map<Eigen::Matrix<realNumType,-1,1>,Eigen::Aligned32> trialGradMap(&trialGradCubic[0],trialGradCubic.size(),1);

                foundDirectionalDeriv = trialGradMap.dot(testingUpdateAngles.real());
                a = (foundDirectionalDeriv - (b + c))/(t*t);
                b /= t;
                if (energyTrial < Energy && gradVector_mu.norm() > 1e-5)
                    doingForwardsSteps = true;

                long double discriminant = b*b-4*a*c;
                if (c >= 0)
                    logger().log("c >= 0", c);
                if (a >= 0)
                {
                    newT = (-b + std::sqrt(discriminant))/(2*a);
                }
                else
                {
                    if (discriminant >= 0)
                    {
                        long double t1 = (-b + std::sqrt(discriminant))/(2*a);
                        E1 = a*t1*t1*t1/3 + b*t1*t1/2 + c*t1 + Energy;
                        E2 = a*tMax*tMax*tMax/3 + b*tMax*tMax/2 + c*tMax + Energy;
                        if (E2 < E1)
                        {
                            newT = tMax;
                        }
                        else
                        {
                            newT = t1;
                        }
                    }
                    else //monotone decrease up to boundary
                    {
                        newT = tMax;
                    }
                }
                if (newT > tMax)
                    newT = tMax;
                angles = anglesCopy;
                for (size_t i = 0; i < angles.size();i++)
                    angles[i] += newT*updateAngles[i];
                if (searchCount > maxSearch)
                    break;


                if (energyTrial > Energy && gradVector_mu.norm() > 1e-5)
                {
                    if (doingForwardsSteps)
                    {
                        if (lastGoodTFowardSteps > tMax/2)
                            lastGoodTFowardSteps = tMax/2;
                        t = t/2;
                        tMax = tMax/2;
                        newT = lastGoodTFowardSteps;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += newT*updateAngles[i];
                        // logger().log("Recovering Forward");
                        break;
                    }
                    doingForwardsSteps = false;
                    t = t/2;
                    tMax = tMax/2;
                    angles = anglesCopy;
                    for (size_t i = 0; i < angles.size();i++)
                        angles[i] += t*updateAngles[i];

                }
                else
                {
                    if (doingForwardsSteps)
                    {
                        t = 2*t;
                        tMax = 2*tMax;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += t*updateAngles[i];
                        lastGoodTFowardSteps = newT;
                    }
                    else
                    {
                        if (newT > tMax)
                            newT = tMax;
                        angles = anglesCopy;
                        for (size_t i = 0; i < angles.size();i++)
                            angles[i] += newT*updateAngles[i];
                        break;
                    }

                }
            }
            while(true);

            // logger().log("tMax",tMax);
            // logger().log("Computed t",t);
            // logger().log("new T",newT);
            // logger().log("Energy", Energy);
            // logger().log("Energy Trial", energyTrial);
            // logger().log("initialDirectionalDeriv",initialDirectionalDeriv);
            // logger().log("foundDirectionalDeriv",foundDirectionalDeriv);
            // if (!(abs(foundDirectionalDeriv) < -c2*initialDirectionalDeriv) && searchCount != 1)
            //     logger().log("Failed to meet wolfe condition",searchCount);

        }

        auto stop = std::chrono::high_resolution_clock::now();

        realNumType overlap = psiH.dot(dest)*InvnormOfPsiH;
        fprintf(stderr,"EnergyProj: %.10lg GradNorm: " realNumTypeCode " OverlapWithPsiH: %.10lg Time (ms): %li Energy Evals: %zu Hessian Evals %zu\n",
                Energy,gradVector_mu.norm(),overlap,std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(),EnergyEvals,HessianEvals);
        if (gradVector_mu.norm() < 1e-10)
            break;
    }
}

bool TUPSQuantities::doStepsUntilHessianIsPositiveDefinite(FusedEvolve *myAnsatz, std::vector<realNumType>& angles, bool doDerivativeSteps = true)
{
    realNumType stepSize = 0.01;

    int numberOfUniqueParameters = m_compressMatrix.rows();

    bool foundOne = false;
    bool solvedSomething = false;
    int count = 10;
    vector<realNumType> gradVec;
    vector<numType> temp;
    while(count-- > 0)
    {
        vector<numType> dest;
        myAnsatz->evolve(dest,angles);

        // const std::vector<vector<numType>>& derivTangentSpace = myAnsatz->getDerivTangentSpace();


        /* some notation:
             * quantities in `compressed' notation after taking into account which angles are equivalent are denoted by the greek subscripts \mu \nu
             * in uncompressed notation with every exponential having a free angle: latin subscripts i,j
             * coefficients of a ket in a basis e.g. \ket{\Psi} in the statevector basis have the subscripts a,b
             *
             * So \frac{d}{d \theta_\mu} is a derivative with respect to restricted angles.
             * H_{ij} is the Hessian with each angle acting independently.
             * v_a = \ket{\Psi} is also valid
             */


        // H_{\mu\nu}: = \frac{dx_j}{d\theta_nu} \frac{dx_i}{d\theta_mu} H_{ij} (for real vectors)
        // H_{ij}      = \braket{\psi | H | \frac{d^2}{dx_i d x_j} \psi } + \braket{\frac{d^2}{dx_i d x_j} \psi | H | \psi} +
        //             + \braket{\frac{d}{d x_i}\psi | H | \frac{d}{d x_j}\psi} + \braket{\frac{d}{d x_j}\psi | H | \frac{d}{d x_i}\psi}

        //secondDerivTensor_{ija} = \frac{d}{d x_i}\frac{d}{d x_j}\ket{\Psi} {Note j <= i the rest is via symmetry}
        // const std::vector<std::vector<vector<numType>>>& secondDerivTensor = myAnsatz->getSecondDerivTensor();
        // size_t iSize = secondDerivTensor.size();
        // size_t jSize = secondDerivTensor.back().size();
        //Hessian_{ij}
        Matrix<realNumType>::EigenMatrix Hij; //uninitialized by default
        //Hessian_{\mu\nu}
        Matrix<realNumType>::EigenMatrix Hmunu;


        // asyncHij(*Ham,secondDerivTensor,derivTangentSpace,Hij,dest,iSize);


        myAnsatz->evolveHessian(Hmunu,gradVec,angles);

        Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> es(Hmunu,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<realNumType>::EigenVector hessianEigVal = es.eigenvalues();
        auto hessianEigVec = es.eigenvectors();



        realNumType zeroThreshold = 1e-10;
        vector<realNumType>::EigenVector updateAngles(angles.size());
        updateAngles.setZero(angles.size());
        foundOne = false;
        for (long int i = 0; i < hessianEigVal.rows(); i++)
        {
            auto he  = hessianEigVal[i];
            realNumType e = he;

            if (e < -1e-7)
            {
                foundOne = true;
                solvedSomething = true;
                fprintf(stderr,"NHE, solving: " realNumTypeCode "\n",e);
                const vector<realNumType>::EigenVector &evCondensed = hessianEigVec.col(i);
                updateAngles += m_deCompressMatrix * evCondensed*stepSize;
            }
        }
        for (size_t i = 0; i < angles.size();i++)
            angles[i] += updateAngles[i];

        if (!foundOne)
            break;
    }
    for (int i =0; i < 1000 && doDerivativeSteps; i++)
    {
        vector<numType> dest;
        myAnsatz->evolve(dest,angles);
        // const std::vector<vector<numType>>& derivTangentSpace = myAnsatz->getDerivTangentSpace();

        // Matrix<numType>::EigenMatrix derivTangentSpaceEM = convert(derivTangentSpace).transpose();
        // vector<numType>::EigenVector destEM = dest;

        /* g_{\mu\nu} = \braket{\eta_\mu | \eta_\nu}
         * Where \ket{\eta_\mu} = \frac{d}{d \theta_\mu} \ket{\Psi}
         *
         * Some \theta_mu are fixed with respect to each other. we have f(x_1,x_2,x_3...etc)
         * and derivTangentSpaceEM_{ij} =  \frac{d}{d\x_j} f_i(x_1,x_2,x_3...etc)
         * Therefore \frac{d}{d\theta_1}f(x_1,x_2,x_3...etc) = \frac{d}{d\x_1}f(x_1,x_2...etc) + \frac{d}{d\x_2}f(x_1,x_2...etc).
         * We have used \frac{dx_1}{d\theta_1} = 1
         * i.e. chain rule.
        */

        //derivTangentSpaceEMCondensed_{a\mu} = compressMatrix_{\mu,i} derivTangentSpaceEM_{a,i}
        // Matrix<numType>::EigenMatrix derivTangentSpaceEMCondensed =  derivTangentSpaceEM * m_compressMatrix.transpose();

        //gradVector_{\mu} = 2 * \braket{\psi | H | \frac{d}{d\theta_\mu}\psi}
        // sparseMatrix<realNumType,numType>::EigenSparseMatrix HamEm = *Ham;
        // vector<realNumType>::EigenVector gradVector = 2*(destEM.adjoint()*HamEm*derivTangentSpaceEMCondensed).real();
        // gradVector = m_deCompressMatrix * gradVector;

        vector<realNumType>::EigenVector gradVectorEM;
        myAnsatz->evolveDerivative(dest,gradVec,angles);
        gradVectorEM = gradVec;
        gradVectorEM = m_deCompressMatrix *(m_compressMatrix * gradVectorEM);

        realNumType Energy = m_Ham->apply(dest,temp).dot(dest);



        //opt->m_vars.count++;
        //realNumType maxGradientValue = *std::max_element(g,g+n,[](realNumType a, realNumType b) {return std::abs(a) < std::abs(b);}); // actual gradient
        //realNumType minGradientValue = *std::min_element(g,g+n,[](realNumType a, realNumType b) {return std::abs(a) < std::abs(b);}); // actual gradient
        //fprintf(stderr,"LBFGS-LS MaxGradientValue: %g, MaxGradientValue2: %g, MaxGradCoeff:%g\n",maxGradientValue,minGradientValue, opt->m_vars.maxGradCoeff);
        //opt->printDistance(opt->m_vars.count);
        fprintf(stderr,"Tryling gradient step: Energy: " realNumTypeCode " GradNorm: " realNumTypeCode "\n", Energy, gradVectorEM.norm());
        for (size_t idx = 0; idx < angles.size(); idx++)
        {
            angles[idx] -= 0.05*gradVectorEM[idx];
        }


    }
    return solvedSomething;


}

void TUPSQuantities::doSubspaceDiagonalisation(std::shared_ptr<stateAnsatz> myAnsatz, std::shared_ptr<FusedEvolve> FE, size_t numberOfMinima, const std::vector<std::vector<ansatz::rotationElement>>& rotationPaths)
{
    bool useFusedEvolve = false;

    if (!myAnsatz)
        useFusedEvolve = true;
    //skip over HF state
    // m_HamEm = m_Ham;
    numberOfMinima = std::min(rotationPaths.size()-1,numberOfMinima);
    Matrix<numType>::EigenMatrix HMat(numberOfMinima,numberOfMinima);
    Matrix<numType>::EigenMatrix SMat(numberOfMinima,numberOfMinima);
    if (!useFusedEvolve)
    {
        myAnsatz->setCalculateFirstDerivatives(false);
        myAnsatz->setCalculateSecondDerivatives(false);
    }
    std::vector<vector<numType>> states(numberOfMinima);
    std::vector<vector<numType>> hstates(numberOfMinima);

    for (size_t i = 0; i < numberOfMinima; i++)
    {
        if (useFusedEvolve)
        {
            std::vector<realNumType> angles(rotationPaths[i+1].size());
            std::transform(rotationPaths[i+1].begin(),rotationPaths[i+1].end(),angles.begin(),[](const ansatz::rotationElement& r){return r.second;});
            FE->evolve(states[i],angles);
        }
        else
            myAnsatz->calcRotationAlongPath(rotationPaths[i+1],states[i],myAnsatz->getStart());
        m_Ham->apply(states[i],hstates[i]);
    }

#ifdef useComplex
    for (size_t i = 0; i < numberOfMinima; i++)
    {
        for (size_t j = 0; j <= i; j++)
        {
            HMat(i,j) = states[i].cdot(hstates[j]);
            HMat(j,i) = std::conj(HMat(i,j));
            SMat(i,j) = states[i].cdot(states[j]);
            SMat(j,i) = std::conj(SMat(i,j));
        }
    }
#else
    for (size_t i = 0; i < numberOfMinima; i++)
    {
        for (size_t j = 0; j <= i; j++)
        {
            HMat(i,j) = states[i].dot(hstates[j]);
            HMat(j,i) = HMat(i,j);
            SMat(i,j) = states[i].dot(states[j]);
            SMat(j,i) = SMat(i,j);
        }
    }
#endif

    realNumType numericalPrecision = 1e-14;

    Eigen::SelfAdjointEigenSolver<Matrix<numType>::EigenMatrix> esSMat(SMat,Eigen::ComputeEigenvectors);
    auto SEigenValues = esSMat.eigenvalues();
    size_t rankOfNullSpace = 0;
    for (auto& se : SEigenValues)
    {
        if (se < -numericalPrecision)
        {
            logger().log("Matrix is not positive definite", se);
            se = 0;
            rankOfNullSpace++;
        }
        else if (se < numericalPrecision)
        {
            se = 0;
            rankOfNullSpace++;
        }
        else
        {
            se = 1./std::sqrt(se);
        }
    }
    //Lowdin symmetrise
    Matrix<numType>::EigenMatrix InvSqrtSMat = esSMat.eigenvectors() * SEigenValues.asDiagonal() * esSMat.eigenvectors().adjoint();
    Matrix<numType>::EigenMatrix SymmetrisedHMat = InvSqrtSMat * HMat * InvSqrtSMat;
    Eigen::SelfAdjointEigenSolver<Matrix<numType>::EigenMatrix> es(SymmetrisedHMat,Eigen::EigenvaluesOnly);

    logger().log("Rank of Null space of overlap matrix", rankOfNullSpace);

    auto foundEigenValues = es.eigenvalues();
    for (long i = 0; i < foundEigenValues.rows() && i < 10; i++)
        fprintf(stderr,"%20.16lf",foundEigenValues[i]);
    fprintf(stderr,"\n");
    if (m_Ham->rows() < 1500 && m_Ham->canGetSparse())
    {
        Eigen::SelfAdjointEigenSolver<Matrix<numType>::EigenMatrix> es(m_Ham->getSparse(),Eigen::ComputeEigenvectors);
        auto TrueEigenValues = es.eigenvalues();
        fprintf(stderr,"TrueEigenValues:\n");
        for (long i = 0; i < TrueEigenValues.rows() && i < 10; i++)
            fprintf(stderr,"%20.16lf",TrueEigenValues[i]);
        fprintf(stderr,"\n");

        //Construct density matrix
        vector<numType> lowestEigenVector(es.eigenvectors().col(0));

        std::shared_ptr<compressor> stateCompressor;
        bool isCompressed = states[0].getIsCompressed(stateCompressor);
        vector<numType> decompressedLowestEigenVector;
        if (isCompressed)
        {
            lowestEigenVector.setIsCompressed(isCompressed,stateCompressor);
            compressor::deCompressVector<numType>(lowestEigenVector,decompressedLowestEigenVector,stateCompressor);
        }
        else
        {
            decompressedLowestEigenVector.copy(lowestEigenVector);// this could be a move
        }
        int numberOfQubits = 1;
        // logger().log("Decompressed Size",decompressedLowestEigenVector.size());
        while(decompressedLowestEigenVector.size() > (1ul<<numberOfQubits))
        {
            numberOfQubits++;
        }
        // logger().log("numberOfQubits",numberOfQubits);
        Matrix<numType>::EigenMatrix DensityMatrix(numberOfQubits,numberOfQubits);
        for (int i = 0; i < numberOfQubits; i++)
        {//        <a^\dagger_i a_j>
            size_t iBitMask = 1<<i;

            for (int j = 0; j <=i; j++)
            {
                numType cumulate = 0;
                size_t jBitMask = 1<<j;
                for (size_t k = 0; k < decompressedLowestEigenVector.size(); k++)
                {
                    size_t k2;
                    if (k & jBitMask)
                    {
                        k2 = jBitMask ^ k;
                        if ((k2 & iBitMask) == 0)
                        {
                            k2 = iBitMask ^ k2;
                            cumulate += decompressedLowestEigenVector[k2]*decompressedLowestEigenVector[k];
                        }
                    }
                }
                DensityMatrix(i,j) = cumulate;
#ifdef useComplex
                DensityMatrix(j,i) = std::conj(DensityMatrix(i,j));
#else
                DensityMatrix(j,i) = DensityMatrix(i,j);
#endif
            }
        }
        Eigen::SelfAdjointEigenSolver<Matrix<numType>::EigenMatrix> esDensity(DensityMatrix,Eigen::EigenvaluesOnly);
        auto DensityEigenValues = esDensity.eigenvalues();
        realNumType cutoff = 1e-14;
        size_t NumberOfEigenValuesThatAreLarge = 0;
        for (long i = 0; i < DensityEigenValues.size(); i++)
        {
            if (DensityEigenValues(i) < 0)
            {
                logger().log("Density matrix is not positive definite", DensityEigenValues(i));
            }
            else if (DensityEigenValues(i) > cutoff)
                NumberOfEigenValuesThatAreLarge++;
        }
        logger().log("Density matrix Large EigenValue count", NumberOfEigenValuesThatAreLarge);
        std::cerr << "Density EigenValues:\n" << DensityEigenValues << "\n";//TODO make pretty
    }


}

realNumType TUPSQuantities::computeFrechetDistanceBetweenPaths(std::shared_ptr<stateAnsatz> myAnsatz, std::shared_ptr<FusedEvolve> FE,
                                                       const std::vector<baseAnsatz::rotationElement> &rotationPath, const std::vector<baseAnsatz::rotationElement> &rotationPath2)
{
    return 0;
    //TODO make this able to use FE
    // Matrix<realNumType>::EigenMatrix distanceMatrix;
    std::vector<vector<numType>::EigenVector> firstVectors;
    std::vector<vector<numType>::EigenVector> secondVectors;
    firstVectors.reserve(rotationPath.size());
    secondVectors.reserve(rotationPath.size());

    if (rotationPath.size() != rotationPath2.size())
    {
        fprintf(stderr,"Rotation paths not the same size?\n");
    }
    myAnsatz->resetPath();
    myAnsatz->setCalculateSecondDerivatives(false);
    for (auto rpe : rotationPath)
    {
        myAnsatz->addRotation(rpe.first,rpe.second);
        firstVectors.push_back(myAnsatz->getVec());
    }
    myAnsatz->resetPath();
    myAnsatz->setCalculateSecondDerivatives(false);
    for (auto rpe : rotationPath2)
    {
        myAnsatz->addRotation(rpe.first,rpe.second);
        secondVectors.push_back(myAnsatz->getVec());
    }
    // distanceMatrix.resize(rotationPath.size(),rotationPath2.size());
    // for (long i = 0; i < distanceMatrix.rows(); i++)
    // {
    //     for (long j = 0; j < distanceMatrix.cols(); j++)
    //     {
    //         distanceMatrix(i,j) = (firstVectors[i] - secondVectors[j]).norm();
    //     }
    // }
    auto getDistance = [&](long i,long j){return (firstVectors[i] - secondVectors[j]).norm();};

    //Do dijkstra to find the shortest route through the matrix while making sure the row and column is increasing
    struct pathObject
    {
        long i;
        long j;
        realNumType distance;
        realNumType maxDistance;
        bool operator == (const pathObject& rhs)const{return (i==rhs.i && j == rhs.j);}
        bool operator <= (const pathObject& rhs)const{return (i==rhs.i && j == rhs.j && maxDistance <= rhs.maxDistance);}
    };
    std::list<pathObject> searchStack;
    Matrix<realNumType>::EigenMatrix visitedLocations(rotationPath.size(),rotationPath2.size());
    visitedLocations.setConstant(-1);

    searchStack.push_back({0,0,getDistance(0,0),getDistance(0,0)});
    visitedLocations(0,0) = getDistance(0,0);

    while (true)
    {
        pathObject currPathObj = searchStack.front();


        searchStack.pop_front();
        if (currPathObj.i == (long)rotationPath.size()-1 && currPathObj.j == (long)rotationPath2.size()-1)
        {
            return currPathObj.maxDistance;
        }
        if (searchStack.size() > rotationPath.size()*rotationPath2.size())
            fprintf(stderr, "heh?\n");

        if (currPathObj.i < (long)rotationPath.size()-1)
        {
            pathObject newObject = currPathObj;
            newObject.i +=1;
            newObject.distance = getDistance(newObject.i,newObject.j);
            newObject.maxDistance = std::max(newObject.distance,newObject.maxDistance);

            if (visitedLocations(newObject.i,newObject.j) == -1 || visitedLocations(newObject.i,newObject.j) > newObject.maxDistance) // no shorter path here
            {
                visitedLocations(newObject.i,newObject.j) = newObject.maxDistance; // new shortest path
                searchStack.remove(newObject); // remove the longer path from the search stack (if it exists)

                for (auto it = searchStack.begin();; ++it)
                {
                    if (it->maxDistance >= newObject.maxDistance || it == searchStack.end())
                    {
                        searchStack.insert(it,newObject);
                        break;
                    }
                }
            }
        }
        if (currPathObj.i < (long)rotationPath.size()-1 && currPathObj.j < (long)rotationPath2.size()-1)
        {
            pathObject newObject = currPathObj;
            newObject.i +=1;
            newObject.j +=1;
            newObject.distance = getDistance(newObject.i,newObject.j);
            newObject.maxDistance = std::max(newObject.distance,newObject.maxDistance);

            if (visitedLocations(newObject.i,newObject.j) == -1 || visitedLocations(newObject.i,newObject.j) > newObject.maxDistance) // no shorter path here
            {
                visitedLocations(newObject.i,newObject.j) = newObject.maxDistance; // new shortest path
                searchStack.remove(newObject); // remove the longer path from the search stack (if it exists)

                for (auto it = searchStack.begin();; ++it)
                {
                    if (it->maxDistance >= newObject.maxDistance || it == searchStack.end())
                    {
                        searchStack.insert(it,newObject);
                        break;
                    }
                }
            }
        }
        if (currPathObj.j < (long)rotationPath2.size()-1)
        {
            pathObject newObject = currPathObj;
            newObject.j +=1;
            newObject.distance = getDistance(newObject.i,newObject.j);
            newObject.maxDistance = std::max(newObject.distance,newObject.maxDistance);

            if (visitedLocations(newObject.i,newObject.j) == -1 || visitedLocations(newObject.i,newObject.j) > newObject.maxDistance) // no shorter path here
            {
                visitedLocations(newObject.i,newObject.j) = newObject.maxDistance; // new shortest path
                searchStack.remove(newObject); // remove the longer path from the search stack (if it exists)

                for (auto it = searchStack.begin();; ++it)
                {
                    if (it->maxDistance >= newObject.maxDistance || it == searchStack.end())
                    {
                        searchStack.insert(it,newObject);
                        break;
                    }
                }
            }
        }
    }
}


realNumType TUPSQuantities::OptimiseTups(stateAnsatz &myAnsatz, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues)
{
    std::vector<stateRotate::exc> excs;
    std::vector<realNumType> anglesV(m_deCompressMatrix.rows());
    {
        vector<realNumType>::EigenVector angles(rp.size());
        for (size_t i = 0; i <rp.size(); i++ )
        {
            excs.emplace_back();
            dynamic_cast<stateRotate*>(myAnsatz.getLie())->convertIdxToExc(rp[i].first,excs.back());
            angles[i] = rp[i].second;// + std::rand()/(10.*RAND_MAX);
            anglesV[i] = angles[i];
        }
        angles = m_deCompressMatrix * m_normCompressMatrix * angles;
        for (long i = 0; i < angles.rows(); i++)
        {
            anglesV[i] = angles[i];
        }
    }
    FusedEvolve FE(myAnsatz.getStart(),m_Ham,m_compressMatrix,m_deCompressMatrix);
    FE.updateExc(excs);
    realNumType Energy = OptimiseTups(FE,rp,avoidNegativeHessianValues);

    return Energy;
}

realNumType TUPSQuantities::OptimiseTups(FusedEvolve& FE, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues)
{

    std::vector<realNumType> anglesV(rp.size());
    std::transform(rp.begin(),rp.end(),anglesV.begin(),[](const ansatz::rotationElement& r){return r.second;});

    realNumType normOfGradVector = 1;
    realNumType Energy = 0;
    bool amStuck = false;
    while (true)
    {
        runNewtonMethod(&FE,anglesV,avoidNegativeHessianValues);
        bool foundOne  = doStepsUntilHessianIsPositiveDefinite(&FE,anglesV,false);

        vector<numType> dest;
        FE.evolve(dest,anglesV);


        /* g_{\mu\nu} = \braket{\eta_\mu | \eta_\nu}
         * Where \ket{\eta_\mu} = \frac{d}{d \theta_\mu} \ket{\Psi}
         *
         * Some \theta_mu are fixed with respect to each other. we have f(x_1,x_2,x_3...etc)
         * and derivTangentSpaceEM_{ij} =  \frac{d}{d\x_j} f_i(x_1,x_2,x_3...etc)
         * Therefore \frac{d}{d\theta_1}f(x_1,x_2,x_3...etc) = \frac{d}{d\x_1}f(x_1,x_2...etc) + \frac{d}{d\x_2}f(x_1,x_2...etc).
         * We have used \frac{dx_1}{d\theta_1} = 1
         * i.e. chain rule.
        */

        //derivTangentSpaceEMCondensed_{a\mu} = compressMatrix_{\mu,i} derivTangentSpaceEM_{a,i}


        //gradVector_{\mu} = \braket{\psi | H | \frac{d}{d\theta_\mu}\psi} + \braket{\frac{d}{d\theta_\mu}\psi | H | \psi}
        vector<realNumType> gradVector;
        FE.evolveDerivative(dest,gradVector,anglesV,&Energy);
        vector<realNumType>::EigenVector gradVectorEm  = gradVector;
        // gradVectorEm = m_compressMatrix * gradVectorEm;

        normOfGradVector = gradVectorEm.norm();
        // Energy = m_Ham->apply(dest).dot(dest);
        fprintf(stderr, "Energy: " realNumTypeCode " GradNorm: " realNumTypeCode "\n", Energy, normOfGradVector);

        if (normOfGradVector < 1e-12)
            break;
        if (!foundOne && amStuck)
            break;
        else if (!foundOne)
            amStuck = true;
        else
            amStuck = false;
    }
    fprintf(stderr,"Optimised Angles: \n");
    Eigen::Vector<realNumType,Eigen::Dynamic> angles(anglesV.size());
    for(size_t i =0; i < anglesV.size(); i++)
    {
        angles[i] = anglesV[i];
        fprintf(stderr,"%.15lg\n",(double)anglesV[i]);
    }
    fprintf(stderr,"Condensed Angles: \n");
    vector<realNumType>::EigenVector angles2 = m_normCompressMatrix * angles;
    for(long i =0; i < angles2.rows(); i++)
    {
        fprintf(stderr,"%.15lg\n",(double)angles2[i]);
    }
    for (size_t i = 0; i < rp.size(); i++)
    {
        rp[i].second = anglesV[i];
    }
    return Energy;



}

realNumType innerProduct(const vector<numType>::EigenVector& a, const vector<numType>::EigenVector& b)
{
    return std::real(a.dot(b));
};

realNumType TUPSQuantities::iterativeTups(stateAnsatz &myAnsatz, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues)
{
    std::vector<stateRotate::exc> excs;
    std::vector<realNumType> anglesV(m_deCompressMatrix.rows());
    {
        vector<realNumType>::EigenVector angles(rp.size());
        for (size_t i = 0; i <rp.size(); i++ )
        {
            excs.emplace_back();
            dynamic_cast<stateRotate*>(myAnsatz.getLie())->convertIdxToExc(rp[i].first,excs.back());
            angles[i] = rp[i].second;// + std::rand()/(10.*RAND_MAX);
            anglesV[i] = angles[i];
        }
        angles = m_deCompressMatrix * m_normCompressMatrix * angles;
        for (long i = 0; i < angles.rows(); i++)
        {
            anglesV[i] = angles[i];
        }
    }
    FusedEvolve FE(myAnsatz.getStart(),m_Ham,m_compressMatrix,m_deCompressMatrix);
    FE.updateExc(excs);
    realNumType Energy = iterativeTups(FE,rp,avoidNegativeHessianValues);

    return Energy;
}

realNumType TUPSQuantities::iterativeTups(FusedEvolve& FE, std::vector<baseAnsatz::rotationElement> &rp, bool avoidNegativeHessianValues)
{
    bool doDIIS = false;

    std::vector<realNumType> anglesV;
    {
        vector<realNumType>::EigenVector angles(rp.size());
        for (size_t i = 0; i <rp.size(); i++ )
        {
            angles[i] = rp[i].second;// + std::rand()/(10.*RAND_MAX);
        }
        angles = m_deCompressMatrix * m_normCompressMatrix * angles;
        for (size_t i = 0; i <rp.size(); i++ )
        {
            anglesV.push_back(angles[i]);
        }
    }
    realNumType normOfGradVector = 1;
    realNumType Energy = 0;
    vector<numType> psiH;

    vector<numType>::EigenVector prevDestEm;
    vector<numType> dest;
    vector<numType> prevDest;
    FE.evolve(prevDest,anglesV);
    prevDestEm = prevDest;
    m_Ham->apply(prevDest,psiH);

    EDIIS<vector<numType>::EigenVector,vector<realNumType>::EigenVector,innerProduct> myDIIS(15);
    int count = 0;
    while (count++ < 3000)
    {


        // vector<numType> psiH;
        // mul(m_Ham,prevDest,psiH);



        runNewtonMethodProjected(&FE,anglesV,psiH,prevDest);


        FE.evolve(dest,anglesV);
        m_Ham->apply(dest,psiH);
        vector<realNumType> gradVector;

        FE.evolveDerivativeProj(dest,gradVector,anglesV,psiH,&Energy);
        vector<realNumType>::EigenVector gradVectorEm  = gradVector;
        normOfGradVector = gradVectorEm.norm();

        fprintf(stderr, "Energy: " realNumTypeCode " GradNorm: " realNumTypeCode "\n", Energy, normOfGradVector);

        if (normOfGradVector < 1e-5)
            break;

        //Prepare for next
        if (doDIIS)
        {
            vector<realNumType>::EigenVector currAngles(anglesV.size());
            for (size_t i = 0; i <anglesV.size(); i++ )
                currAngles[i] = anglesV[i];

            currAngles = m_normCompressMatrix*currAngles;
            vector<numType>::EigenVector newDestEM = dest;
            myDIIS.addNew(currAngles,newDestEM-prevDestEm);
            // myDIIS.addNew(currAngles,gradVectorEm);
            vector<numType>::EigenVector extrapolatedError;
            myDIIS.getNext(currAngles,extrapolatedError);
            currAngles = m_deCompressMatrix*currAngles;

            for (size_t i = 0; i <anglesV.size(); i++ )
            {
                anglesV[i] = currAngles[i];
            }
            prevDestEm = std::move(newDestEM);
            static_cast<Matrix<numType>&>(prevDest) = std::move(dest);
        }
        else
        {
            prevDestEm = dest;
            static_cast<Matrix<numType>&>(prevDest) = std::move(dest);
        }


    }
    fprintf(stderr,"Optimised Angles: \n");
    vector<realNumType>::EigenVector angles(anglesV.size());
    for(long i =0; i < angles.rows(); i++)
    {
        angles[i] = anglesV[i];
        rp[i].second = anglesV[i];
        fprintf(stderr,"%.15lg\n",(double)angles[i]);
    }
    fprintf(stderr,"Condensed Angles: \n");
    vector<realNumType>::EigenVector angles2 = m_normCompressMatrix * angles;
    for(long i =0; i < angles2.rows(); i++)
    {
        fprintf(stderr,"%.15lg\n",(double)angles2[i]);
    }
    return Energy;
}
#define columnSize "26"
#define textColumnSize "40"

void TUPSQuantities::printOutputLine(std::vector<double>& toPrint, std::string name)
{
    fprintf(m_file,"%-" textColumnSize "s",name.c_str());
    //fprintf(stderr,"%-" columnSize "s", "N/A");
    for (auto n : toPrint)
    {
        fprintf(m_file,"%-" columnSize ".16lg", n);
    }
    fprintf(m_file,"\n");

}
void TUPSQuantities::printOutputLine(std::vector<long double>& toPrint, std::string name)
{
    fprintf(m_file,"%-" textColumnSize "s",name.c_str());
    //fprintf(stderr,"%-" columnSize "s", "N/A");
    for (auto n : toPrint)
    {
        fprintf(m_file,"%-" columnSize ".10Lg", n);
    }
    fprintf(m_file,"\n");

}
void TUPSQuantities::printOutputHeaders(size_t numberOfPathsExHF)
{


    fprintf(m_file,"\n");
    fprintf(m_file,"%-" textColumnSize "s %40s \n","Results:", "Path Number");
    fprintf(m_file,"%-" textColumnSize "s",""); //placeholder for type of printout
    fprintf(m_file,"%-" columnSize "s","HF");
    for (size_t i = 1; i <= numberOfPathsExHF;i++)
    {
        fprintf(m_file,"%-" columnSize "zu", i);
    }
    fprintf(m_file,"\n");
}


