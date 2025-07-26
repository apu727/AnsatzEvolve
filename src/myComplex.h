/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef MYCOMPLEX_H
#define MYCOMPLEX_H
#include "globals.h"
#include <cassert>

template <typename T>
inline T makeComplex(realNumType R, realNumType I){assert(I == 0);return R;};

template<>
inline std::complex<realNumType> makeComplex(realNumType R, realNumType I){return std::complex<realNumType>(R,I);};

inline realNumType myConj(realNumType val)
{
    return val;
}

inline std::complex<realNumType> myConj(std::complex<realNumType> val)
{
    return std::conj(val);
}



#endif // MYCOMPLEX_H
