/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef BENCHMARK_H
#define BENCHMARK_H
#include "ansatz.h"

void benchmark(stateAnsatz* ansatz, std::vector<ansatz::rotationElement> rp, sparseMatrix<realNumType,numType>& Ham,sparseMatrix<realNumType,numType>::EigenSparseMatrix& compressMatrix, sparseMatrix<realNumType,numType>::EigenSparseMatrix &m_deCompressMatrix);

#endif // BENCHMARK_H
