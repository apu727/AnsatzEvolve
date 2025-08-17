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
    if (m_Ham->rows() < 100000)
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
    int maxStepCount = 250;
    realNumType zeroThreshold = 1e-10;

    int count = maxStepCount;
    size_t HessianEvals =0;
    size_t EnergyEvals = 0;
    vector<numType> temp;
    while(count-- > 0)
    {
        auto start = std::chrono::high_resolution_clock::now();

        vector<numType> dest;
        myAnsatz->evolve(dest,angles);

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

        auto curveDamp = [](realNumType v){if (v > 1000) return (v-1000)*0.1 +1000; else return v;};

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
            }
            else
                InvhessianEigVal[i] = 0;

        }
        // Matrix<std::complex<realNumType>>::EigenMatrix testingEigenDecompose = (hessianEigVec * hessianEigVal.asDiagonal()*hessianEigVec.adjoint()) - Hmunu;
        vector<std::complex<realNumType>>::EigenVector testingUpdateAngles = - hessianEigVec * InvhessianEigVal.asDiagonal()*hessianEigVec.adjoint() * gradVector_mu + negativeEigenValueDirections;
        updateAngles = m_deCompressMatrix * testingUpdateAngles.real();
        // realNumType maxAngleStep = 0;
        // for (realNumType a : updateAngles)
        //     maxAngleStep = std::max(a,maxAngleStep);
        // if (abs(maxAngleStep) < 1e-8)
        // {
        //     if (zeroThreshold == 1e-8)
        //         break;
        //     zeroThreshold = 1e-8;
        // }

        // if (maxAngleStep > maxStepSize)
        //     updateAngles /= (maxAngleStep/maxStepSize);

        for (size_t i = 0; i < angles.size();i++)
            angles[i] += updateAngles[i];

        //Implements backtracking
        realNumType EnergyTrial =0;
        int BacktrackCount = 0;
        while(true)
        {
            vector<numType> trial;
            myAnsatz->evolve(trial,angles);
            // vector<numType>::EigenVector destEMTrial = trial;
            // EnergyTrial = (destEMTrial.adjoint() * HamEm * destEMTrial).real()(0,0);
            EnergyTrial = myAnsatz->getEnergy(trial);//m_Ham.braket(trial,trial,&temp);
            EnergyEvals++;
            if (EnergyTrial > Energy && BacktrackCount < 30)
            {//Backtracking
                // logger().log("Backtracking",BacktrackCount);
                for (size_t i = 0; i < angles.size();i++)
                {
                    updateAngles[i] /=2;
                    angles[i] -= updateAngles[i];
                }
                BacktrackCount++;
            }
            else
                break;
        }
        auto stop = std::chrono::high_resolution_clock::now();
        fprintf(stderr,"Energy: " realNumTypeCode " GradNorm: " realNumTypeCode " Time (ms): %li, Energy Evals: %zu, Hess Evals: %zu\n",
                Energy,gradVector_mu.norm(),std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count(),EnergyEvals,HessianEvals);

        if (gradVector_mu.norm() < 1e-12)
            break;
    }
}

void TUPSQuantities::runNewtonMethodProjected(stateAnsatz *myAnsatz,bool avoidNegativeHessianValues)
{ // Projected energy
    avoidNegativeHessianValues = true;
    myAnsatz->setCalculateFirstDerivatives(false);
    myAnsatz->setCalculateSecondDerivatives(false);
    int maxStepCount = 250;
    realNumType zeroThreshold = 1e-10;

    // myAnsatz->setCalculateSecondDerivatives(true);
    const std::vector<ansatz::rotationElement> &rp = myAnsatz->getRotationPath();

    std::vector<realNumType> angles;
    for (auto rpe : rp)
    {
        angles.push_back(rpe.second);
    }

    int count = maxStepCount;
    vector<numType> temp;
    vector<numType> psiH;
    vector<numType> previousDestCopy;
    previousDestCopy.copy(myAnsatz->getVec());
    m_Ham->apply(previousDestCopy,psiH);
    realNumType InvnormOfPsiH =1./std::sqrt(psiH.dot(psiH));
    bool amStuck = false;
    while(count-- > 0)
    {
        auto start = std::chrono::high_resolution_clock::now();
        myAnsatz->updateAngles(angles);

        const vector<numType>& dest = myAnsatz->getVec();

        vector<realNumType> gradVectorCalc;
        Matrix<realNumType>::EigenMatrix Hij; //uninitialized by default

        myAnsatz->getHessianAndDerivativeProj(psiH,Hij,gradVectorCalc);
        vector<realNumType>::EigenVector gradVectorCalcEm = gradVectorCalc;

        vector<realNumType>::EigenVector gradVector_mu = m_compressMatrix * gradVectorCalcEm;
        realNumType Energy = dest.dot(psiH);

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

        // secondDerivTensor_{ija} = \frac{d}{d x_i}\frac{d}{d x_j}\ket{\Psi} {Note j <= i the rest is via symmetry}
        // Hessian_{ij}
        //Hessian_{\mu\nu}
        Matrix<realNumType>::EigenMatrix Hmunu(m_numberOfUniqueParameters,m_numberOfUniqueParameters); //uninitialized by default
        Hmunu = m_compressMatrix * Hij * m_compressMatrix.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix<realNumType>::EigenMatrix> es(Hmunu,Eigen::DecompositionOptions::ComputeEigenvectors);
        vector<std::complex<realNumType>>::EigenVector hessianEigVal = es.eigenvalues();
        vector<realNumType>::EigenVector InvhessianEigVal(hessianEigVal.rows());
        auto hessianEigVec = es.eigenvectors();




        vector<realNumType>::EigenVector updateAngles(rp.size());
        updateAngles.setZero(rp.size());

        vector<realNumType>::EigenVector negativeEigenValueDirections(hessianEigVal.rows());
        negativeEigenValueDirections.setZero();

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
        vector<std::complex<realNumType>>::EigenVector testingUpdateAngles = - hessianEigVec * InvhessianEigVal.asDiagonal()*hessianEigVec.adjoint() * gradVector_mu + negativeEigenValueDirections;
        updateAngles = m_deCompressMatrix * testingUpdateAngles.real();

        for (size_t i = 0; i < rp.size();i++)
            angles[i] += updateAngles[i];

        //Implements backtracking

        int BacktrackCount = 0;

        realNumType EnergyTrial =0;
        while(true)
        {
            vector<numType> trial;
            myAnsatz->updateAnglesNoDeriv(angles,trial);
            EnergyTrial = psiH.dot(trial);
            if (EnergyTrial > Energy && BacktrackCount < 30)
            {//Backtracking
                // logger().log("Backtracking",BacktrackCount);
                for (size_t i = 0; i < rp.size();i++)
                {
                    updateAngles[i] /=2;
                    angles[i] -= updateAngles[i];
                }
                BacktrackCount++;
            }
            else
                break;
        }



        auto stop = std::chrono::high_resolution_clock::now();

        realNumType overlap = psiH.dot(dest)*InvnormOfPsiH;
        // fprintf(stderr,"Energy: " realNumTypeCode " GradNorm: " realNumTypeCode " Overlap: " realNumTypeCode " Time (ms): %li\n", Energy,gradVector_mu.norm(),overlap,std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        if (BacktrackCount >= 30 && amStuck)
        {
            logger().log("Break on Stuck");
            break;
        }
        else if (BacktrackCount >= 30)
            amStuck = true;
        else
            amStuck = false;
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
    myAnsatz->setCalculateFirstDerivatives(false);
    myAnsatz->setCalculateSecondDerivatives(false);
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
        FE.evolveDerivative(dest,gradVector,anglesV);
        vector<realNumType>::EigenVector gradVectorEm  = gradVector;
        // gradVectorEm = m_compressMatrix * gradVectorEm;

        normOfGradVector = gradVectorEm.norm();
        Energy = m_Ham->apply(dest).dot(dest);
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

void TUPSQuantities::iterativeTups(sparseMatrix<realNumType,numType> &Ham, const std::vector<baseAnsatz::rotationElement> &rp, stateAnsatz &myAnsatz, bool avoidNegativeHessianValues)
{
    bool doDIIS = true;
    myAnsatz.setCalculateFirstDerivatives(false);
    myAnsatz.setCalculateSecondDerivatives(false);
    myAnsatz.resetPath();
    {
        vector<realNumType>::EigenVector angles(rp.size());
        for (size_t i = 0; i <rp.size(); i++ )
        {
            angles[i] = rp[i].second;// + std::rand()/(10.*RAND_MAX);
        }
        angles = m_deCompressMatrix * m_normCompressMatrix * angles;
        for (size_t i = 0; i <rp.size(); i++ )
        {
            myAnsatz.addRotation(rp[i].first,angles[i]);
        }
    }
    realNumType normOfGradVector = 1;
    realNumType Energy = 0;

    EDIIS<vector<numType>::EigenVector,vector<realNumType>::EigenVector,innerProduct> myDIIS(15);
    int count = 0;
    while (count++ < 3000)
    {
        vector<numType> prevDest;
        prevDest.copy(myAnsatz.getVec());
        vector<numType>::EigenVector prevDestEm;
        prevDestEm = prevDest;
        // vector<numType> psiH;
        // mul(m_Ham,prevDest,psiH);



        runNewtonMethodProjected(&myAnsatz,true);

        const vector<numType>& dest = myAnsatz.getVec();


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

        myAnsatz.getDerivativeVec(m_Ham,gradVector); // True Derivative
        // myAnsatz.getDerivativeVecProj(psiH,gradVector); // True Derivative
        vector<realNumType>::EigenVector gradVectorEm  = gradVector;
        gradVectorEm = m_compressMatrix * gradVectorEm;




        normOfGradVector = gradVectorEm.norm();
        Energy = Ham.braket(dest,dest);
        fprintf(stderr, "Energy: " realNumTypeCode " GradNorm: " realNumTypeCode "\n", Energy, normOfGradVector);

        if (normOfGradVector < 1e-5)
            break;

        //Prepare for next
        if (doDIIS)
        {
            auto rotationPath = myAnsatz.getRotationPath();
            vector<realNumType>::EigenVector currAngles(rotationPath.size());
            for (size_t i = 0; i <rotationPath.size(); i++ )
            {
                currAngles[i] = rotationPath[i].second;
            }

            currAngles = m_normCompressMatrix*currAngles;
            vector<numType>::EigenVector newDestEM = dest;
            myDIIS.addNew(currAngles,newDestEM-prevDestEm);
            // myDIIS.addNew(currAngles,gradVectorEm);
            vector<numType>::EigenVector extrapolatedError;
            myDIIS.getNext(currAngles,extrapolatedError);
            currAngles = m_deCompressMatrix*currAngles;

            std::vector<realNumType> nextAngles(rotationPath.size());
            for (size_t i = 0; i <rotationPath.size(); i++ )
            {
                nextAngles[i] = currAngles[i];
            }
            myAnsatz.updateAngles(nextAngles);
        }
    }
    fprintf(stderr,"Optimised Angles: \n");
    const std::vector<baseAnsatz::rotationElement> &rp2 = myAnsatz.getRotationPath();
    vector<realNumType>::EigenVector angles(rp2.size());
    for(long i =0; i < angles.rows(); i++)
    {
        angles[i] = rp2[i].second;
        fprintf(stderr,"%.15lg\n",(double)angles[i]);
    }
    fprintf(stderr,"Condensed Angles: \n");
    vector<realNumType>::EigenVector angles2 = m_normCompressMatrix * angles;
    for(long i =0; i < angles2.rows(); i++)
    {
        fprintf(stderr,"%.15lg\n",(double)angles2[i]);
    }



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


