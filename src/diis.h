/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef DIIS_H
#define DIIS_H
#include "logger.h"

#include <list>
template <typename errorVectorType, typename quantityType, auto innerProductFunc>
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

        Eigen::VectorXd CoeffVec(B.completeOrthogonalDecomposition().solve(rhsVec));
        // Preconditioned version see arXiv:2112.08890v1 eq 14.
        realNumType totalSum = CoeffVec.sum();
        if (totalSum == 0)
        {
            CoeffVec.setZero();
            CoeffVec(CoeffVec.rows()-1) = 1;
        }
        else
            CoeffVec /= totalSum; // makes sure the coeffs add to 1.

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

#endif // DIIS_H
