/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef CSVWRITER_H
#define CSVWRITER_H

#include "globals.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

class csvWriter
{
    std::string m_filename;
    std::ofstream m_file;
public:
    csvWriter(std::string filename,std::ios_base::openmode mode = std::ios_base::out | std::ios_base::trunc,int precision = 7);

    template <typename T>
    void writeIterableLine(const T &elems)
    {
        writeIterable(elems);
        m_file << std::endl;
    }
    template <typename T>
    void writeIterable(const T& elems)
    {
        for (const auto & elem : elems)
            m_file << elem << ",";
    }
    template <typename T>
    void writeLine(const T& elems){m_file << elems << std::endl;}
    template <typename T>
    void write(const T& elems){m_file << elems << ',';}
};

#endif // CSVWRITER_H
