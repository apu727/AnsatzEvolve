/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef GLOBALS_H
#define GLOBALS_H


//#define useEigen




#include <complex>
// #define useComplex

#ifdef useComplex
using std::real;
using std::imag;
//COMPLEX NUMBERS
typedef double realNumType;
typedef std::complex<realNumType> numType;

#define numTypeCode "%lg,%lg j"
#define PRINTABLENUMTYPE(x) real(x),imag(x)
#define realNumTypeCode "%lg"


#else
//REAL NUMBERS
typedef double numType;
typedef double realNumType;
#define numTypeCode "%lg"
#define PRINTABLENUMTYPE(x) x
#define realNumTypeCode "%lg"

//end useComplex
#endif
static std::complex<realNumType> iu = std::complex<realNumType>(0,1);
#define LOGTIMINGS
#if defined(LOGTIMINGS) //|| !defined(NDEBUG)
inline constexpr bool logTimings = true;
#else
inline constexpr bool logTimings = false;
#endif



template <typename dataType,typename vectorType>
class sparseMatrix;

typedef sparseMatrix<numType,numType> matrixType;



const realNumType TOLERANCE = 1e-5;
extern unsigned long NUM_CORES;

#define Bool2String(x) (x ? "True" : "False")

#define EXPORT_FUNCTION _attribute__((visibility("default")))
#endif // GLOBALS_H
