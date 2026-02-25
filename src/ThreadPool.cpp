/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "dolphin/ThreadPool.h"
#include <iostream>
#include <spdlog/spdlog.h>

ThreadPool::ThreadPool(size_t numThreads, std::function<void()> threadInitFunc)
    : stop(false),
    activeWorkers(std::atomic<int>(numThreads)) {
    newTaskCondition = [](){return true;};
    for(size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this, threadInitFunc] {
            threadInitFunc();
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(tasks_mutex);
                    condition.wait(lock, [this]{ return stopThreads > 0 || stop || !tasks.empty(); });
                    if(stopThreads > 0){
                        stopThreads -= 1;
                        return; // thread basically "removing" itself from threadpool
                    }
                    if(stop && tasks.empty()){
                        activeWorkers -= 1;
                        return;
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                spdlog::get("default")->debug("Thread ({}) starting task", std::hash<std::thread::id>{}(std::this_thread::get_id()));
                task();
                spdlog::get("default")->debug("Thread ({}) finished task", std::hash<std::thread::id>{}(std::this_thread::get_id()));
                queueSpace.notify_one();

            }
        });
    }
}
    

    
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(tasks_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers){
        if(worker.joinable()){
            worker.join();
        }
        else spdlog::get("default")->critical("thread not joinable");
        if (std::this_thread::get_id() == worker.get_id()) {
            spdlog::get("default")->critical("Attempting to join self!");
        }
    } 
}

void ThreadPool::setCondition(std::function<bool()> condition){
    newTaskCondition = condition;
}

bool ThreadPool::reduceActiveWorkers(int amount){
    activeWorkers -= amount;
    return activeWorkers < 1;
}
//returns max reached
//TODO messy function, technically the thread isnt deactivated yet, but in executor it just waits,
// move the deactivation to the deconvolution
bool ThreadPool::reduceNumberThreads(int amount){
    
    if (stopThreads == 0){
        stopThreads += amount;
    }
    condition.notify_all();
    if (stopThreads >= activeWorkers){
        return true;
    }
    return false;
}

