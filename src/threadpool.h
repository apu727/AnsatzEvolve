/* Copyright (C) 2025 Bence Csakany
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <queue>


class threadpool
{
private:
    std::vector<std::future<void>> m_workers;

    std::queue<std::pair<std::function<void()>,std::promise<void>>> m_workQueue;
    std::mutex m_workQueueMutex;

    std::condition_variable m_newWork;
    std::atomic_bool m_toExit = false;

    friend class threadworker;
    friend void workFunction(threadpool*);

    threadpool(int num);

public:

    threadpool(threadpool& other) = delete;
    ~threadpool();

    std::future<void> queueWork(std::function<void()> work);
    static threadpool& getInstance(int num){static threadpool me(num); return me;};
};




#endif // THREADPOOL_H
