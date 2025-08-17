/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#include "threadpool.h"
#include "logger.h"
#ifdef _OPENMP
#include <omp.h>
#endif
unsigned long NUM_CORES = 2;

void workFunction(threadpool* pool)
{
    while (!pool->m_toExit)
    {
        std::pair<std::function<void()>,std::promise<void>> work;
        {
            std::unique_lock lock(pool->m_workQueueMutex);
            if (pool->m_workQueue.empty())
            {
                pool->m_newWork.wait(lock);
                if (pool->m_workQueue.empty())
                    continue;
            }
            work = std::move(pool->m_workQueue.front());
            pool->m_workQueue.pop();
        }
        work.first();
        work.second.set_value();
    }
}
threadpool::threadpool(int num)
{
    int myGuess = NUM_CORES;
    const char* threads = std::getenv("OMP_NUM_THREADS");
    if (threads && (myGuess = std::stoi(threads)) > 0 && num != 1)
    {
        logger().log("Changing thread count from OMP_NUM_THREADS env variable to",myGuess);
    }
    else
    {
#ifdef _OPENMP
        myGuess = omp_get_num_procs();
#else
        myGuess = std::thread::hardware_concurrency();
#endif
    }
    if (myGuess != num && num != 1/*1 is special and used for debugging*/)
    {
        logger().log("Overriding number of threads to make to", myGuess);
        logger().log("If this is not what you indended, set the environment variable OMP_NUM_THREADS to what you want");
        NUM_CORES = myGuess;
        num = myGuess;
    }
    if (num == 1)
        return;
    for (int i = 0; i < num; i++)
    {
        m_workers.push_back(std::async(std::launch::async,workFunction,this));
    }
}

threadpool::~threadpool()
{
    m_toExit = true;
    m_newWork.notify_all();
    for (auto& w : m_workers)
        while(w.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout)
            m_newWork.notify_all();
}

std::future<void> threadpool::queueWork(std::function<void ()> work, bool dontQueue)
{
    std::lock_guard lock(m_workQueueMutex);
    std::promise<void> prom;
    std::future<void> fut = prom.get_future();
    if(m_workers.size() == 0 || dontQueue)
    {
        work();
        prom.set_value();
    }
    else
    {
        m_workQueue.push({work,std::move(prom)});
        m_newWork.notify_all();
    }
    return fut;
}


