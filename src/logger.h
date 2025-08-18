/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef LOGGER_H
#define LOGGER_H

#include "globals.h"
#include "operatorpool.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>


class logger
{
    FILE* m_file = nullptr;
public:
    logger(std::string filename, bool append = false);
    logger();
    virtual ~logger();
    void log(const std::string& message);

#ifdef useComplex
    void log(const std::vector<std::string>& names, const std::vector<numType>& quantities);
    void log(const std::vector<std::string>& names, const std::vector<realNumType>& quantities);
#else
    void log(const std::vector<std::string>& names, const std::vector<numType>& quantities);
#endif

#ifdef useComplex
    void log(std::string name, numType quantity){fprintf(m_file, "%s: " numTypeCode "\n", name.c_str(), PRINTABLENUMTYPE(quantity));}
#endif
    void log(std::string name, realNumType quantity){fprintf(m_file, "%s: " realNumTypeCode "\n", name.c_str(), quantity);}
    void log(std::string name, long double quantity){fprintf(m_file, "%s: %Lg\n", name.c_str(), quantity);}

    void log(std::string name, uint32_t quantity){fprintf(m_file, "%s: %u\n", name.c_str(), quantity);}
    void log(std::string name, int quantity){fprintf(m_file, "%s: %i\n", name.c_str(), quantity);}
    void log(std::string name, long quantity){fprintf(m_file, "%s: %li\n", name.c_str(), quantity);}
    void log(std::string name, size_t quantity){fprintf(m_file, "%s: %zu\n", name.c_str(), quantity);}
    void log(std::string name, std::atomic<int>& quantity) {log(name, (int)quantity);}
    void log(std::string name, std::string value);
    template<typename T>
    void log(std::string name, const std::vector<T>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (auto& o :object)
            fprintf(m_file, "%s, ", std::to_string(o).c_str());
        fprintf(m_file, "]\n");
    }
    void log(std::string name, const std::vector<numType>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (auto& o :object)
            fprintf(m_file, numTypeCode ", ", PRINTABLENUMTYPE(o));
        fprintf(m_file, "]\n");
    }
    void log(std::string name, void* ptr) {fprintf(m_file,"%s: %p\n",name.c_str(),ptr);}
    void logAccurate(std::string name, const std::vector<double>& object)
    {
        fprintf(m_file, "%s: [", name.c_str());
        for (auto& o :object)
            fprintf(m_file,  "%.16lf, ", o);
        fprintf(m_file, "]\n");
    }

};

void writeVector(std::string filename, const vector<numType>& vec,stateRotate&);

inline void releaseAssert(bool val, std::string message)
{
    if (!val)
    {
        logger().log("Release Assert Failed: ",message);
        __builtin_trap();
    }
}

#endif // LOGGER_H
