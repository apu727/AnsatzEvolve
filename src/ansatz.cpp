/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "ansatz.h"
#include "globals.h"
#include "threadpool.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <future>
#include <cstring>

void asyncRotate(std::vector<vector<numType>>& m_derivList, const matrixType &rotationGenerator, realNumType S, realNumType C)
{
    std::atomic_int finishCount = 0;

    auto multiply = [&](size_t startD, size_t endD)
    {
        auto d = m_derivList.begin()+startD;
        auto end = m_derivList.begin()+endD;
        while (d != end)
        {
            rotationGenerator.rotate(S,C,*d,*d);
            d++;
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max((size_t)m_derivList.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i<m_derivList.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_derivList.size());
        futures.push_back(pool.queueWork([=,&multiply](){ multiply(i,endIndex);}));

    }

    for (auto& fut : futures)
        fut.wait();
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

void asyncRotate(std::vector<vectorView<Matrix<numType>>>& m_derivList, const matrixType &rotationGenerator, realNumType S, realNumType C)
{
    std::atomic_int finishCount = 0;

    auto multiply = [&](size_t startD, size_t endD)
    {
        auto d = m_derivList.begin()+startD;
        auto end = m_derivList.begin()+endD;
        while (d != end)
        {
            rotationGenerator.rotate(S,C,*d,*d);
            d++;
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max((size_t)m_derivList.size()/NUM_CORES,1ul);

    auto& pool = threadpool::getInstance(NUM_CORES);
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i<m_derivList.size(); i += stepSize)
    {
        auto endIndex = std::min(i+stepSize,m_derivList.size());
        futures.push_back(pool.queueWork([=,&multiply](){ multiply(i,endIndex);}));

    }

    for (auto& fut : futures)
        fut.wait();
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

void baseAnsatz::rotateState(const matrixType &rotationGenerator, realNumType theta, size_t indexInPath)
{
    m_lieTangentSpaceCacheValid = false;
    m_lieParallelSpaceCacheValid = false;
    m_lieConstSpaceCacheValid = false;

    //assert(allClose(rotationGenerator.T()+rotationGenerator,0));

    double S = 0;
    double C = 0;
    mysincos(theta,&S,&C);


    vector<numType> normDir;
    vector<numType> parallelDir;
    vector<numType> constDir;

    rotationGenerator.rotate(S,C,m_current,m_current);


    if (m_calculateFirstDerivatives || calculateSecondDerivatives)
    {
        asyncRotate(m_derivList,rotationGenerator,S,C);
        asyncRotate(m_derivParallelList,rotationGenerator,S,C);
        asyncRotate(m_derivConstList,rotationGenerator,S,C);
    }


    //add this vector to the derivList

    if (!(m_calculateFirstDerivatives || calculateSecondDerivatives))
    {
        if (indexInPath >= m_derivSpaceNotEvolvedCache.size())
        {
            m_derivSpaceNotEvolvedCache.emplace_back();
            rotationGenerator.multiply(m_current,m_derivSpaceNotEvolvedCache.back());
        }
        else
        {
            rotationGenerator.multiply(m_current,m_derivSpaceNotEvolvedCache[indexInPath]);
        }
    }
    else
    {
        rotationGenerator.multiply(m_current,normDir);// normDir = rotationGenerator * m_current
        if (indexInPath >= m_derivSpaceNotEvolvedCache.size())
        {
            m_derivSpaceNotEvolvedCache.emplace_back();
            m_derivSpaceNotEvolvedCache.back().copy(normDir);
        }
        else
            m_derivSpaceNotEvolvedCache[indexInPath].copy(normDir);
    }

    if (m_calculateFirstDerivatives || calculateSecondDerivatives)
    {
        rotationGenerator.multiply(normDir,parallelDir); //TODO isnt this the same as "SIN(theta)norm + cos(theta)parallel" dir before?? can avoid the Matrix multiplication
        add(m_current,parallelDir,constDir);
        parallelDir*=-1;//parallelDir = -1 * rotationGenerator * normDir

        m_derivList.push_back(std::move(normDir));
        m_derivParallelList.push_back(std::move(parallelDir));
        m_derivConstList.push_back(std::move(constDir));
    }


    if (calculateSecondDerivatives)
    {
        for (auto& pastsecondDerivs : m_secondDerivList)
            asyncRotate(pastsecondDerivs,rotationGenerator,S,C);

        m_secondDerivList.emplace_back();
        std::vector<vector<numType>> &currentLayer =  m_secondDerivList.back();

        for (const vector<numType> &vd : m_derivList)
        {
            currentLayer.emplace_back(0);
            rotationGenerator.multiply(vd,currentLayer.back());
        }
    }
}


void baseAnsatz::resetState()
{
    m_current.copy(m_start);
    m_derivList.clear();
    m_derivParallelList.clear();
    m_derivConstList.clear();
    // m_derivSpaceNotEvolvedCache.clear();
    m_secondDerivList.clear();
    m_lieTangentSpaceCacheValid = false;
    m_lieParallelSpaceCacheValid = false;
    m_lieConstSpaceCacheValid = false;
    //m_derivQFMListCacheValid = false;
    m_derivCoeffCacheValid = false;
}



const vector<numType>& baseAnsatz::getVec() const
{
    return m_current;
}


void baseAnsatz::updateAngles(const std::vector<realNumType> &angles)
{
    resetState();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < m_rotationPath.size(); i++)
    {
        rotationElement& rp = m_rotationPath[i];
        rp.second = angles[i];
        rotateState(*m_lie->getLieAlgebraMatrix(rp.first),rp.second,i);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto __attribute__ ((unused)) duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    // logger().log("update Angles Time taken (ms)",duration);
}

void baseAnsatz::updateAnglesNoDeriv(const std::vector<realNumType> &angles,vector<numType>& dest)
{

    dest.copy(m_start);

    vector<numType> normDir;
    vector<numType> parallelDir;

    for (size_t i = 0; i < m_rotationPath.size(); i++)
    {
        rotationElement& rp = m_rotationPath[i];

        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
        realNumType theta = angles[i];

        double S = 0;
        double C = 0;
        mysincos(theta,&S,&C);

        rotationGenerator.rotate(S,C,dest,dest);
    }
}

void baseAnsatz::calcRotationAlongPath(const std::vector<rotationElement> &rotationPath, vector<numType> &dest, const vector<numType> &start)
{
    if (m_lieIsCompressed)
    {
        compressor::compressVector<numType>(start,dest,m_compressor);
    }
    else
        dest.copy(start);

    for (size_t i = 0; i < rotationPath.size(); i++)
    {
        const rotationElement& rp = rotationPath[i];

        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
        realNumType theta = rotationPath[i].second;
        rotationGenerator.rotate(theta,dest,dest);

    }
}

void baseAnsatz::getDerivativeVec(std::shared_ptr<HamiltonianMatrix<realNumType, numType> > ExpMat, vector<realNumType>& deriv)
{
    deriv.resize(m_rotationPath.size(),false,nullptr);
    vector<numType> hPsi;
    auto superstart = std::chrono::high_resolution_clock::now();
    // ExpMat->multiply(m_current,hPsi);

    {
        // hPsi.resize(m_current.size(),m_lieIsCompressed,m_compressor,false);
        // Eigen::Map<Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> currentMap(&m_current.at(0,0),m_current.m_iSize,m_current.m_jSize);
        // Eigen::Map<Eigen::Matrix<numType,1,-1,Eigen::RowMajor>,Eigen::Aligned32> destMap(&hPsi.at(0,0),hPsi.m_iSize,hPsi.m_jSize);
        // destMap.noalias() = currentMap*m_HamEm;
        ExpMat->apply(m_current,hPsi);
    }

    m_hPsiEvolvedList.resize(m_rotationPath.size());
    m_hPsiEvolvedList.back().copy(hPsi); // this can be removed if only the der
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> futs;
    futs.reserve(m_rotationPath.size());

    for (size_t i = m_rotationPath.size()-1; i > 0 ; i--)
    {
        vector<numType>& src = m_hPsiEvolvedList[i];
        vector<numType>& dest = m_hPsiEvolvedList[i-1];

        const rotationElement& rp = m_rotationPath[i];


        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
        realNumType theta = -m_rotationPath[i].second;

        double S = 0;
        double C = 0;
        mysincos(theta,&S,&C);
        rotationGenerator.rotate(S,C,src,dest);
        futs.push_back(threadpool::getInstance(NUM_CORES).queueWork([i,&deriv,this](){deriv[i] = 2* m_derivSpaceNotEvolvedCache[i].dot(m_hPsiEvolvedList[i]);}));
    }
    deriv[0] = 2* m_derivSpaceNotEvolvedCache[0].dot(m_hPsiEvolvedList[0]);
    for (auto& f : futs)
        f.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start-superstart).count();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("hpsi Time taken (ms)",duration1);
    logger().log("DerivEvolve Time taken (ms)",duration2);
}

void baseAnsatz::getHessianAndDerivative(std::shared_ptr<HamiltonianMatrix<realNumType, numType> > ExpMat, Matrix<realNumType>::EigenMatrix &Hessian, vector<realNumType>& deriv, Eigen::SparseMatrix<realNumType,Eigen::RowMajor>* compressMatrix)
{
    //Its better to assume nothing is cached since we need to recompute everything anyway basically.


    //TODO memory fragmentation
    // H_{ij}      = \braket{\psi | H | \frac{d^2}{dx_i d x_j} \psi } + \braket{\frac{d^2}{dx_i d x_j} \psi | H | \psi} +
    //             + \braket{\frac{d}{d x_i}\psi | H | \frac{d}{d x_j}\psi} + \braket{\frac{d}{d x_j}\psi | H | \frac{d}{d x_i}\psi}

    //It is assumed that ExpMat is Hermitian i.e. Conjugate transpose symmetric
    auto start = std::chrono::high_resolution_clock::now();
    getDerivativeVec(ExpMat,deriv);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    logger().log("getDerivativeVec Time taken (ms)",duration);
    // m_hPsiEvolvedList is now setup
    Hessian.resize(m_rotationPath.size(), m_rotationPath.size());
    Hessian.setZero();
    Eigen::MatrixXd THT;
    if (m_rotationPath.size() < 1)
        return;



    // Do the Double derivative terms

    auto work = [&Hessian,this](size_t starti){
        vector<numType> normDir;
        vector<numType> parallelDir;
        vector<numType> currPos;
        currPos.copy(m_derivSpaceNotEvolvedCache[starti]);
        for (size_t x_j = starti+1; x_j < m_rotationPath.size(); x_j++)
        {
            const rotationElement& rp = m_rotationPath[x_j];


            const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
            realNumType theta = m_rotationPath[x_j].second;

            double S = 0;
            double C = 0;
            mysincos(theta,&S,&C);

            rotationGenerator.rotateAndBraketWithTangentOfResult(S,C,currPos,currPos,m_hPsiEvolvedList[x_j],Hessian(starti,x_j));
            Hessian(starti,x_j) *= 2;
            // rotationGenerator.multiply(currPos,normDir); // normDir = rotationGenerator * m_current
            //normDir now has  \ket{\frac{d^2}{dx_i d x_j}}
            // Hessian(starti,x_j) = 2*normDir.dot(m_hPsiEvolvedList[x_j]);
            // Hessian(x_j,starti) = Hessian(starti,x_j);
        }
    };

    auto& pool = threadpool::getInstance(NUM_CORES);

    std::vector<std::future<void>> futures;

    vector<numType> normDir;
    vector<numType> parallelDir;
    vector<numType> currPos;

    for (size_t x_i = 0; x_i < m_rotationPath.size(); x_i++)
    {
        //Special case the diagonal term
        const vector<numType>& currPos = m_derivSpaceNotEvolvedCache[x_i];
        const rotationElement& rp = m_rotationPath[x_i];
        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);

        rotationGenerator.multiply(currPos,normDir); // normDir = rotationGenerator * m_current
        Hessian(x_i,x_i) = 2*normDir.dot(m_hPsiEvolvedList[x_i]);

        //Queue the off diagonal terms
        futures.push_back(pool.queueWork([&,x_i](){work(x_i);}));

    }

    for (auto& fut : futures)
        fut.wait();

    //Do the cross terms, effectively need to construct m_derivList
    auto start1 = std::chrono::high_resolution_clock::now();
    auto start2 = std::chrono::high_resolution_clock::now();
    auto start3 = std::chrono::high_resolution_clock::now();
    auto start4 = std::chrono::high_resolution_clock::now();



    {
        Matrix<numType> mat;
        mat.resize(m_rotationPath.size(),m_start.size(),m_lieIsCompressed,m_compressor);

        mat.getJVectorView(0).copy(m_derivSpaceNotEvolvedCache[0]);
        std::vector<vectorView<Matrix<numType>,Eigen::RowMajor> > derivList;
        derivList.push_back(mat.getJVectorView(0));

        for (size_t x_i = 1; x_i < m_rotationPath.size(); x_i++)
        {
            const rotationElement& rp = m_rotationPath[x_i];


            const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
            realNumType theta = m_rotationPath[x_i].second;

            double S = 0;
            double C = 0;
            mysincos(theta,&S,&C);


            asyncRotate(derivList,rotationGenerator,S,C);

            m_derivList.emplace_back();
            mat.getJVectorView(x_i).copy(m_derivSpaceNotEvolvedCache[x_i]);
            derivList.push_back(mat.getJVectorView(x_i));
        }

        Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> HPsiDerivC;
        Eigen::Map<Eigen::Matrix<numType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32>matMapR(&mat.at(0,0),mat.m_iSize,mat.m_jSize);

        start2 = std::chrono::high_resolution_clock::now();
        /*This ordering swap is worth it somehow. Probably because the loops can be restructured to be:
             * A += B*C
             * //Slow access
             *  for (j)
             *      //Access the sparse matrix in order
             *      for (k)
             *          //UNROLL/use avx Since B(i,.) and A(i,.) are adjacent in memory
             *          for (i)
             *               A(i,j) += B(i,k)*C(k,j)
             */
        if (compressMatrix != nullptr)
        {
            Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> matMapR_Comp = *compressMatrix*matMapR; //Has roughly the same cost as the dot product. I.e. cheap compared to matrix multiplication
            Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> matMapC = matMapR_Comp;
            // HPsiDerivC.noalias() = matMapC * m_HamEm;
            ExpMat->apply(matMapC,HPsiDerivC);
            THT.resize(compressMatrix->rows(),compressMatrix->rows());

            start3 = std::chrono::high_resolution_clock::now();
            if constexpr(std::is_same_v<numType,typename Eigen::NumTraits<numType>::Real>)
            {
                THT.triangularView<Eigen::Upper>() = 2*matMapR_Comp * HPsiDerivC.transpose();
            }
            else if constexpr(std::is_same_v<std::complex<realNumType>,numType>)
            {
                //Have to do magic trickery to avoid an 8x slowdown, With this its only 4x slower. There are only 2x as many flops so im not sure why? If the conjugate is one flop (its not quite) then 3x flops?
                //https://en.cppreference.com/w/cpp/numeric/complex.html This is well defined
                Eigen::Map<Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32>matMapRReal((realNumType*)matMapR_Comp.data(),matMapR_Comp.rows(),matMapR_Comp.cols()*2);
                Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> HPsiDerivR = HPsiDerivC;

                Eigen::Map<Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor>>HPsiDerivRReal((realNumType*)HPsiDerivR.data(),HPsiDerivR.rows(),HPsiDerivR.cols()*2);
                THT.triangularView<Eigen::Upper>() = 2*matMapRReal * HPsiDerivRReal.transpose();
            }
            else
            {
                static_assert(std::is_same_v<numType,typename Eigen::NumTraits<numType>::Real> || std::is_same_v<std::complex<realNumType>,numType>);
                logger().log("Impossible?");
            }
        }
        else
        {
            Eigen::Matrix<numType,-1,-1,Eigen::ColMajor> matMapC = matMapR; //Has roughly the same cost as the dot product. I.e. cheap compared to matrix multiplication
            // HPsiDerivC.noalias() = matMapC * m_HamEm;
            ExpMat->apply(matMapC,HPsiDerivC);
            THT.resize(m_rotationPath.size(),m_rotationPath.size());

            start3 = std::chrono::high_resolution_clock::now();
            if constexpr(std::is_same_v<numType,typename Eigen::NumTraits<numType>::Real>)
            {
                THT.triangularView<Eigen::Upper>() = 2*matMapR * HPsiDerivC.transpose();
            }
            else if constexpr(std::is_same_v<std::complex<realNumType>,numType>)
            {
                //Have to do magic trickery to avoid an 8x slowdown, With this its only 4x slower. There are only 2x as many flops so im not sure why? If the conjugate is one flop (its not quite) then 3x flops?
                //https://en.cppreference.com/w/cpp/numeric/complex.html This is well defined
                Eigen::Map<Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor>,Eigen::Aligned32>matMapRReal((realNumType*)&mat.at(0,0),mat.m_iSize,mat.m_jSize*2);
                Eigen::Matrix<numType,-1,-1,Eigen::RowMajor> HPsiDerivR = HPsiDerivC;

                Eigen::Map<Eigen::Matrix<realNumType,-1,-1,Eigen::RowMajor>>HPsiDerivRReal((realNumType*)HPsiDerivR.data(),HPsiDerivR.rows(),HPsiDerivR.cols()*2);
                THT.triangularView<Eigen::Upper>() = 2*matMapRReal * HPsiDerivRReal.transpose();
            }
            else
            {
                static_assert(std::is_same_v<numType,typename Eigen::NumTraits<numType>::Real> || std::is_same_v<std::complex<realNumType>,numType>);
                logger().log("Impossible?");
            }
        }
        start4 = std::chrono::high_resolution_clock::now();
    }
    for (long i = 0; i < Hessian.rows(); i++)
    {
        for (long j = i; j < Hessian.rows(); j++)
        {
            Hessian(j,i) = Hessian(i,j);
        }
    }
    for (long i = 0; i < THT.rows(); i++)
    {
        for (long j = i; j < THT.rows(); j++)
        {
            THT(j,i) = THT(i,j);
        }
    }
    if (compressMatrix != nullptr)
        Hessian = *compressMatrix * Hessian * compressMatrix->transpose();
    Hessian += THT;

    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start2-start1).count();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(start3-start2).count();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(start4-start3).count();
    logger().log("Hessian Duration1:",duration1);
    logger().log("Hessian Duration2:",duration2);
    logger().log("Hessian Duration3:",duration3);
}

void baseAnsatz::getDerivativeVecProj(const vector<numType> &projVec, vector<realNumType> &deriv)
{
    deriv.resize(m_rotationPath.size(),false,nullptr);
    auto superstart = std::chrono::high_resolution_clock::now();

    m_hPsiEvolvedList.resize(m_rotationPath.size());
    m_hPsiEvolvedList.back().copy(projVec); // this can be removed if only the der
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> futs;
    futs.reserve(m_rotationPath.size());
    for (size_t i = m_rotationPath.size()-1; i > 0 ; i--)
    {
        vector<numType>& src = m_hPsiEvolvedList[i];
        vector<numType>& dest = m_hPsiEvolvedList[i-1];

        const rotationElement& rp = m_rotationPath[i];


        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
        realNumType theta = -m_rotationPath[i].second;

        double S = 0;
        double C = 0;
        mysincos(theta,&S,&C);
        rotationGenerator.rotate(S,C,src,dest);
        // deriv[i] = m_derivSpaceNotEvolvedCache[i].dot(src);
        futs.push_back(threadpool::getInstance(NUM_CORES).queueWork([i,&deriv,this](){deriv[i] = 2* m_derivSpaceNotEvolvedCache[i].dot(m_hPsiEvolvedList[i]);}));
    }
    deriv[0] = m_derivSpaceNotEvolvedCache[0].dot(m_hPsiEvolvedList[0]);
    for (auto& f : futs)
        f.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    auto __attribute__ ((unused))duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(start-superstart).count();
    auto __attribute__ ((unused))duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    // logger().log("hpsi Time taken (ms)",duration1);
    // logger().log("DerivEvolve Time taken (ms)",duration2);
}

void baseAnsatz::getHessianAndDerivativeProj(const vector<numType> &projVec, Matrix<realNumType>::EigenMatrix &Hessian, vector<realNumType> &deriv)
{
    //Its better to assume nothing is cached since we need to recompute everything anyway basically.


    //TODO memory fragmentation
    // H_{ij}      = \braket{\psi | H | \frac{d^2}{dx_i d x_j} \psi }

    //It is assumed that ExpMat is Hermitian i.e. Conjugate transpose symmetric
    auto start = std::chrono::high_resolution_clock::now();
    getDerivativeVecProj(projVec,deriv);
    auto stop = std::chrono::high_resolution_clock::now();
    auto __attribute__ ((unused))duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count();
    // logger().log("getDerivativeVec Time taken (ms)",duration);
    // m_hPsiEvolvedList is now setup
    Hessian.resize(m_rotationPath.size(), m_rotationPath.size());
    Hessian.setZero();
    if (m_rotationPath.size() < 1)
        return;



    // Do the Double derivative terms

    auto work = [&Hessian,this](size_t starti){
        vector<numType> normDir;
        vector<numType> parallelDir;
        vector<numType> currPos;
        currPos.copy(m_derivSpaceNotEvolvedCache[starti]);
        for (size_t x_j = starti+1; x_j < m_rotationPath.size(); x_j++)
        {
            const rotationElement& rp = m_rotationPath[x_j];


            const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);
            realNumType theta = m_rotationPath[x_j].second;

            double S = 0;
            double C = 0;
            mysincos(theta,&S,&C);

            rotationGenerator.rotateAndBraketWithTangentOfResult(S,C,currPos,currPos,m_hPsiEvolvedList[x_j],Hessian(starti,x_j));
            // rotationGenerator.multiply(currPos,normDir); // normDir = rotationGenerator * m_current
            //normDir now has  \ket{\frac{d^2}{dx_i d x_j}}
            // Hessian(starti,x_j) = 2*normDir.dot(m_hPsiEvolvedList[x_j]);
            Hessian(x_j,starti) = Hessian(starti,x_j);
        }
    };

    auto& pool = threadpool::getInstance(NUM_CORES);

    std::vector<std::future<void>> futures;

    vector<numType> normDir;

    for (size_t x_i = 0; x_i < m_rotationPath.size(); x_i++)
    {
        //Special case the diagonal term
        const vector<numType>& currPos = m_derivSpaceNotEvolvedCache[x_i];
        const rotationElement& rp = m_rotationPath[x_i];
        const matrixType &rotationGenerator = *m_lie->getLieAlgebraMatrix(rp.first);

        rotationGenerator.multiply(currPos,normDir); // normDir = rotationGenerator * m_current
        Hessian(x_i,x_i) = normDir.dot(m_hPsiEvolvedList[x_i]);

        //Queue the off diagonal terms
        futures.push_back(pool.queueWork([&,x_i](){work(x_i);}));

    }

    for (auto& fut : futures)
        fut.wait();
}

void baseAnsatz::resetPath()
{
    resetState();
    m_rotationPath.clear();
}


void baseAnsatz::addRotation(int rotationGeneratorIdx, realNumType angle)
{
    rotateState(*m_lie->getLieAlgebraMatrix(rotationGeneratorIdx),angle,m_rotationPath.size());
    m_rotationPath.push_back({rotationGeneratorIdx,angle});
}

void baseAnsatz::setCalculateFirstDerivatives(bool val)
{
    m_calculateFirstDerivatives = val;
}

void baseAnsatz::setCalculateSecondDerivatives(bool val){calculateSecondDerivatives = val;}

bool baseAnsatz::getCalculateSecondDerivatives(){return calculateSecondDerivatives;}

const vector<numType> &baseAnsatz::getStart(){return m_start;}

const std::vector<std::vector<vector<numType>>>& baseAnsatz::getSecondDerivTensor()
{
    return m_secondDerivList;
}
const std::vector<ansatz::rotationElement> &baseAnsatz::getRotationPath() const
{
    return m_rotationPath;
}
void asyncMultiplyTangent(std::vector<vector<numType>>& dest, const std::unordered_map<size_t,matrixType>& mats, const vector<numType>& vec)
{
    std::atomic_int finishCount = 0;
    dest.resize(mats.size());

    auto multiply = [&](size_t startIndex, size_t endIndex)
    {
        for (size_t i = startIndex; i<endIndex; i++)
            mats.at(i).multiply(vec,dest[i]);//dest[i] = mats.at(i) * vec;
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max((size_t)mats.size()/NUM_CORES,1ul);
    std::vector<std::future<void>> futures;
    threadpool& pool = threadpool::getInstance(NUM_CORES);

    for (size_t i = 0; i<mats.size(); i += stepSize)
            futures.push_back(pool.queueWork([&,i](){multiply(i,std::min(i+stepSize,mats.size()));}));

    for (auto &f: futures)
        f.wait();
    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)futures.size())
            fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

void asyncMultiplyParallel(std::vector<vector<numType>>& dest, const std::unordered_map<size_t,matrixType>& mats, const std::vector<vector<numType>>& vecs)
{
    std::atomic_int finishCount = 0;
    dest.resize(mats.size());

    auto multiply = [&](size_t startIndex, size_t endIndex)
    {
        for (size_t i = startIndex; i<endIndex; i++)
        {
            mats.at(i).multiply(vecs[i],dest[i]);
            dest[i] *= -1;//dest[i] = -1*(mats.at(i) * vecs[i]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };
    const int stepSize = std::max((size_t)mats.size()/NUM_CORES,1ul);
    std::vector<std::future<void>> futures;
    threadpool& pool = threadpool::getInstance(NUM_CORES);

    for (size_t i = 0; i<mats.size(); i += stepSize)
        futures.push_back(pool.queueWork([&,i](){multiply(i,std::min(i+stepSize,mats.size()));}));

    for (auto &f: futures)
        f.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)NUM_CORES)
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}

void asyncSubtractConst(std::vector<vector<numType>>& dest,const vector<numType>& current, const std::vector<vector<numType>>& parallels)
{
    std::atomic_int finishCount = 0;
    dest.resize(parallels.size());

    auto subtract = [&](size_t startIndex, size_t endIndex)
    {
        vector<numType> temp;
        for (size_t i = startIndex; i<endIndex; i++)
        {
            mul(-1,parallels[i],temp);
            add(current,temp,dest[i]);
        }
        std::atomic_fetch_add_explicit(&finishCount,1,std::memory_order_release);
    };

    const int stepSize = std::max((size_t)parallels.size()/NUM_CORES,1ul);
    std::vector<std::future<void>> futures;
    threadpool& pool = threadpool::getInstance(NUM_CORES);

    for (size_t i = 0; i<parallels.size(); i += stepSize)
        futures.push_back(pool.queueWork([&,i](){subtract(i,std::min(i+stepSize,parallels.size()));}));

    for (auto &f: futures)
        f.wait();

    while (std::atomic_load_explicit(&finishCount,std::memory_order_acquire) < (int)NUM_CORES)
        fprintf(stderr,"Wait returned but not all done?");
    std::atomic_thread_fence(std::memory_order_acquire);
}


const std::vector<vector<numType> > & baseAnsatz::getLieTangentSpace()
{
    if (!m_lieTangentSpaceCacheValid)
    {
        //m_lieTangentSpaceCache.clear();

        asyncMultiplyTangent(m_lieTangentSpaceCache,*m_lie->getLieAlgebraMatrices(),m_current);
        m_lieTangentSpaceCacheValid = true;

    }
    return m_lieTangentSpaceCache;
}


const std::vector<vector<numType> > &baseAnsatz::getLieParallelSpace()
{
    if (!m_lieParallelSpaceCacheValid)
    {
        //m_lieParallelSpaceCache.clear();

        const std::vector<vector<numType> >& lt = getLieTangentSpace();

        asyncMultiplyParallel(m_lieParallelSpaceCache,*m_lie->getLieAlgebraMatrices(),lt);

        m_lieParallelSpaceCacheValid = true;
    }
    return m_lieParallelSpaceCache;
}


const std::vector<vector<numType> > &baseAnsatz::getLieConstSpace()
{
    if (!m_lieConstSpaceCacheValid)
    {
        //m_lieParallelSpaceCache.clear();

        const std::vector<vector<numType> >& lp = getLieParallelSpace();

        asyncSubtractConst(m_lieConstSpaceCache,m_current,lp);

        m_lieConstSpaceCacheValid = true;
    }
    return m_lieConstSpaceCache;
}


const std::vector<vector<numType> > &baseAnsatz::getDerivTangentSpace()
{
    return m_derivList;
}


const std::vector<vector<numType> > &baseAnsatz::getDerivParallelSpace()
{
    return m_derivParallelList;
}


const std::vector<vector<numType>>& baseAnsatz::getDerivConstSpace()
{
    return m_derivConstList;
}


const std::vector<realNumType>& baseAnsatz::getGradVector(bool spaceBased, const vector<numType>& targetVectorPauliBasis)
{
    if (m_derivCoeffCacheValid)
    {
        return m_derivCoeffListCache;
    }
    //generate the derivs
    auto& currentVec = getVec();
    if (spaceBased)
    {
        vector<numType> projCurrent;
        vector<numType> projTangent;
        mul(*m_target,currentVec,projCurrent);

        m_derivCoeffListCache.resize(m_rotationPath.size());
        for (size_t i = 0; i < m_rotationPath.size(); i++)
        {
            mul(*m_target,m_derivList[i],projTangent);
            m_derivCoeffListCache[i] = -2.*projTangent.dot(projCurrent);
        }
    }
    else
    {
        m_derivCoeffListCache.resize(m_rotationPath.size());
        for (size_t i = 0; i < m_rotationPath.size(); i++)
        {
            m_derivCoeffListCache[i] = -1*targetVectorPauliBasis.dot(m_derivList[i]);
        }
    }
    m_derivCoeffCacheValid = true;
    return m_derivCoeffListCache;
}




baseAnsatz::baseAnsatz(targetMatrix<numType,numType> *target, const vector<numType> &start, operatorPool *lie)
{
    m_lie = lie;
    m_lieIsCompressed = lie->getCompressor(m_compressor);
    if (m_lieIsCompressed)
        compressor::compressVector<numType>(start,m_start,m_compressor);

    else
        m_start.copy(start);
    m_current.copy(m_start);
    m_target = target;
    //m_lieAlgebraElems = m_lie->getLieAlgebraMatrices();
    resetState();
}

stateAnsatz::stateAnsatz(targetMatrix<numType,numType>* target,const vector<numType>& start, stateRotate* lie) : baseAnsatz(target,start,lie)
{
}
