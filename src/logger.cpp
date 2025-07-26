/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "logger.h"
#include "operatorpool.h"
#include "csvwriter.h"
#include "bitset"

logger::logger()
{
    m_file = stderr;
}

logger::logger(std::string filename, bool append)
{
    if (append)
        m_file = fopen(filename.c_str(), "a");
    else
        m_file = fopen(filename.c_str(), "w");
    if (m_file == nullptr)
    {
        logger().log("Could not open file for logging",filename);
        logger().log("Error:",errno);
        m_file = stderr;
    }
    if (!append)
        log("Start Logging\n");
}

logger::~logger()
{
    if (m_file != stderr)
        fclose(m_file);
}

void logger::log(const std::vector<std::string> &names, const std::vector<realNumType> &quantities)
{
    for (size_t i = 0; i < names.size(); i++)
    {
        fprintf(m_file, "%s: " realNumTypeCode ", " , names[i].c_str(), quantities[i]);
    }
    fprintf(m_file,"\n");
}
#ifdef useComplex
void logger::log(const std::vector<std::string> &names, const std::vector<numType> &quantities)
{
    for (size_t i = 0; i < names.size(); i++)
    {
        fprintf(m_file, "%s: " realNumTypeCode " + " realNumTypeCode"j, " , names[i].c_str(), quantities[i].real(),quantities[i].imag());
    }
    fprintf(m_file,"\n");
}
#endif
void logger::log(const std::string& message)
{
    fprintf(m_file, "%s\n", message.c_str());

}





void logger::log(std::string name, std::string value)
{
    fprintf(m_file, "%s: %s\n", name.c_str(), value.c_str());
}


void writeVector(std::string filename, const vector<numType> &vec, stateRotate &)
{
    csvWriter stateLog(filename);
    if (std::is_same_v<numType,realNumType>)
    {
        stateLog.writeLine("S,R,I");
    }
    else if (std::is_same_v<numType,std::complex<realNumType>>)
        stateLog.writeLine("P,R,I");
    //silently ignore
    uint8_t nQubits = std::log2(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
    {

        stateLog.write(std::bitset<32>(i).to_string().c_str()+(32-nQubits));
        stateLog.write(std::real(vec[i]));
        stateLog.writeLine(std::imag(vec[i]));
    }
}
