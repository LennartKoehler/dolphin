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

#pragma once
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t numThreads = 1, std::function<void()> threadInitFunc = [](){});
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        std::shared_ptr<std::packaged_task<return_type()>> task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(tasks_mutex);
            // wait until queue has space
            queueSpace.wait(lock, [this] {
                return stop || newTaskCondition();
            });

            if (stop) throw std::runtime_error("DeconvolutionBackendThreadManager stopped");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    
    ~ThreadPool();
    void setCondition(std::function<bool()> condition);
    size_t queueSize(){ return tasks.size(); } // remember you have to lock the tasks_mutex
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable condition;
    std::condition_variable queueSpace;
    bool stop;


    std::function<bool()> newTaskCondition;
};