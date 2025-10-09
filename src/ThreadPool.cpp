#include "ThreadPool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    newTaskCondition = [](){return true;};
    for(size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(tasks_mutex);
                    condition.wait(lock, [this]{ return stop || !tasks.empty(); });
                    if(stop && tasks.empty()) return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                queueSpace.notify_one();
                task();
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
    for(std::thread &worker: workers) worker.join();
}

void ThreadPool::setCondition(std::function<bool()> condition){
    newTaskCondition = condition;
}
