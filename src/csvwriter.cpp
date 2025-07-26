/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "csvwriter.h"


csvWriter::csvWriter(std::string filename,std::ios_base::openmode mode, int precision)
{
    m_file.open(filename,mode);
    m_file.precision(precision);
}





