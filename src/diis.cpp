/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "globals.h"
#include "diis.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

bool simpleNewtonRaphson(const std::function<Eigen::MatrixXd (Eigen::VectorXd)> &HessianFunc,
                                    const std::function<Eigen::VectorXd (Eigen::VectorXd)> &GradFunc,
                                    const std::function<realNumType (Eigen::VectorXd)> &/*ErrorFunc*/,
                                    Eigen::VectorXd& point)
{
    int numberOfStepsLeft = 200;
    realNumType zeroThreshold = 1e-6;
    realNumType machinePrecision = 1e-15;
    bool converged = false;
    Eigen::MatrixXd Hess;
    Eigen::VectorXd Grad;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
    Eigen::VectorXd eVal;
    Eigen::MatrixXd eVec;

    Eigen::VectorXd step;
    Eigen::VectorXd newGrad;

    while(numberOfStepsLeft-- > 0)
    {
        Hess = HessianFunc(point);
        Grad = GradFunc(point);
        if (Grad.norm() < 1e-12)
        {
            converged = true;
            break;
        }
        // realNumType Error = ErrorFunc(point);

        es.compute(Hess,Eigen::ComputeEigenvectors);
        eVal = es.eigenvalues();
        eVec = es.eigenvectors();
        for (auto& e : eVal)
        {
            if (std::abs(e) < zeroThreshold)
                e = 0;
            else
                e = 1./e;
        }
        step = -(eVec * (eVal.asDiagonal() * (eVec.adjoint() * Grad)));
        point += step;
        newGrad = GradFunc(point);
        while (newGrad.norm() > Grad.norm() && step.lpNorm<Eigen::Infinity>() > machinePrecision)
        {
            step /= 2;
            point -= step;
            newGrad = GradFunc(point);
        }
        if (!(step.lpNorm<Eigen::Infinity>() > machinePrecision))
        {
            // fprintf(stderr, "Backtracking failed in simpleNewtonRaphson\n");
            break;
        }
    }
    return converged;
}
