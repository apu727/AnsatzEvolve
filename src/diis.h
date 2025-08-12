/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef DIIS_H
#define DIIS_H
#include "logger.h"
#include <list>

bool simpleNewtonRaphson(const std::function<Eigen::MatrixXd (Eigen::VectorXd)> &HessianFunc,
                         const std::function<Eigen::VectorXd (Eigen::VectorXd)> &GradFunc,
                         const std::function<realNumType (Eigen::VectorXd)> &ErrorFunc,
                         Eigen::VectorXd& point);

template <typename errorVectorType, typename quantityType, auto innerProductFunc, bool EDIIS>
class DIIS
{
    size_t m_maxDIISSize;
    std::list<errorVectorType> m_pastEVecs;
    std::list<quantityType> m_pastQuantities;
public:
    DIIS(size_t maxDIISSize)
    {
        m_maxDIISSize = maxDIISSize;
    }
    void reset()
    {
        m_pastEVecs.clear();
        m_pastQuantities.clear();
    }
    void addNew(const quantityType& quantity, const errorVectorType& eVec)
    {
        m_pastEVecs.push_back(eVec);
        m_pastQuantities.push_back(quantity);
        if (m_pastEVecs.size() > m_maxDIISSize)
        {
            m_pastEVecs.pop_front();
            m_pastQuantities.pop_front();
        }
    }

    void getNext(quantityType& nextQuantity, errorVectorType& extrapolatedError)
    {
        releaseAssert(m_pastEVecs.size() != 0,"DIIS must have at least one vector in it");

        size_t currDIISSize = m_pastEVecs.size();

        Eigen::MatrixXd B(currDIISSize,currDIISSize);


        auto iIterator = m_pastEVecs.begin();

        for (size_t i = 0; i < currDIISSize; i++,iIterator++)
        {
            auto jIterator = m_pastEVecs.begin();
            for(size_t j = 0; j < currDIISSize; j++,jIterator++)
            {
                B(i,j) = innerProductFunc(*iIterator,*jIterator);
            }
        }
        // Eigen::Matrix<numType,CDIISSize,CDIISSize> BInv(B.inverse());
        Eigen::VectorXd rhsVec(currDIISSize);
        rhsVec.setConstant(1);
        Eigen::VectorXd CoeffVec;

        if constexpr(EDIIS)
        {
            //Normally
            //Error = c_i B_{ij} c_j - \lambda( c_ii -1)
            //Derivative_i = B_{ij} c_j + c_j B_{ji}  - \lambda
            //For Real inner products, usually the case, things B_{ij} = B_{ji}
            //Derivative_i = 2B_{ij} c_j - \lambda
            //This leads to Pulay DIIS

            //Substituting c_i = t_i^2 gives
            //Error = t^2_i B_{ij} t^2_j - \lambda( t_i t_i -1)
            //Derivative_i = (t_i 4B_{ij} t^2_j - 2\lambda t_i No sum on i
            //Derivative_lambda = 1-t_i t_i
            //Hessian_{ij} = \delta_{ij} 4B_{jk} t^2_k + 8 t_i B_{ij} t_j - 2\lambda \delta_{ij} No sum on i or j
            //Hessian_{iLambda} = -2t_i
            //Hessian_{Lambda i} = -2t_i
            //Hessian_lambdaLambda = 0
            auto ErrorFunc = [&B](const Eigen::VectorXd& point)
            {
                Eigen::VectorXd ts = point.cwiseProduct(point).segment(0,point.size()-1);
                realNumType lambda = point(point.size()-1);
                realNumType error = ts.adjoint() * B * ts;
                error -= lambda*(ts.sum()-1);
                return error;
            };

            auto GradFunc = [&B](const Eigen::VectorXd& point)
            {
                Eigen::VectorXd t = point.segment(0,point.size()-1);
                Eigen::VectorXd tSq = point.cwiseProduct(point).segment(0,point.size()-1);
                realNumType lambda = point(point.size()-1);

                Eigen::VectorXd GradTs = 4*(B * tSq);
                GradTs = GradTs.cwiseProduct(t);
                GradTs -= 2*lambda*t;

                Eigen::VectorXd Grad(point.size());
                for (long i = 0; i < GradTs.size(); i++)
                    Grad[i] = GradTs[i];

                Grad[point.size()-1] = 1-tSq.sum();
                return Grad;
            };



            auto HessianFunc = [&B](const Eigen::VectorXd& point)
            {
                Eigen::VectorXd t = point.segment(0,point.size()-1);
                Eigen::VectorXd tSq = point.cwiseProduct(point).segment(0,point.size()-1);
                realNumType lambda = point(point.size()-1);

                Eigen::MatrixXd Hess(point.rows(),point.rows());

                Eigen::VectorXd BTsq = 4*B*tSq;

                for (long i = 0; i < tSq.rows(); i++)
                {
                    for (long j = 0; j < tSq.rows(); j++)
                    {
                        Hess(i,j) = (i == j ? BTsq(i) : 0)  + 8*t(i)*B(i,j)*t(j) - (i == j ? 2*lambda : 0);
                    }
                }
                //Lambda block
                for (long i = 0; i < tSq.rows(); i++)
                {
                    Hess(point.size()-1,i) = -2*t(i);
                    Hess(i,point.size()-1) = -2*t(i);
                }
                Hess(point.size()-1,point.size()-1) = 0;

                return Hess;
            };

            Eigen::VectorXd point(currDIISSize+1);
            // for (int i = 0; i < currDIISSize; i++)
            // {
            //     point(i) = i*i;
            // }
            point.setConstant(sqrt(currDIISSize));
            point(currDIISSize) = 1; // lambda = 1

            simpleNewtonRaphson(HessianFunc,GradFunc,ErrorFunc,point);
            CoeffVec = point.cwiseProduct(point).segment(0,point.size()-1);
            logger().log("CoeffVec Normalisation", CoeffVec.sum());
            Eigen::VectorXd logVector = CoeffVec.array().log();
            logger().log("CoeffVecEntropy", -CoeffVec.cwiseProduct(logVector).sum());
            logger().log("MaxEntropy", log(currDIISSize) );

        }
        else
        {
            CoeffVec = Eigen::VectorXd(B.completeOrthogonalDecomposition().solve(rhsVec));
            // Preconditioned version see arXiv:2112.08890v1 eq 14.
            realNumType totalSum = CoeffVec.sum();

            if (totalSum == 0)
            {
                CoeffVec.setZero();
                CoeffVec(CoeffVec.rows()-1) = 1;
            }
            else
                CoeffVec /= totalSum; // makes sure the coeffs add to 1.
        }

        auto pastquantityIt = m_pastQuantities.begin();
        auto errorIt = m_pastEVecs.begin();


        for (size_t i = 0; i < currDIISSize; )
        {
            if (i == 0)
            { // avoid the problem with initialisation
                nextQuantity = CoeffVec[i] * *pastquantityIt;
                extrapolatedError = CoeffVec[i] * *errorIt;
            }
            else
            {
                nextQuantity += CoeffVec[i] * *pastquantityIt;
                extrapolatedError += CoeffVec[i] * *errorIt;
            }

            i++;
            pastquantityIt++;
            errorIt++;
        }
        //Calc error

        logger().log("DIIS Extrapolated error norm",std::sqrt(innerProductFunc(extrapolatedError,extrapolatedError)));
        logger().log("DIIS Curr error norm",std::sqrt(innerProductFunc(m_pastEVecs.back(),m_pastEVecs.back())));

    }
};

template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using PulayDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,false>;


template <typename errorVectorType, typename quantityType, auto innerProductFunc>
using EDIIS = DIIS<errorVectorType,quantityType,innerProductFunc,true>;


#endif // DIIS_H
