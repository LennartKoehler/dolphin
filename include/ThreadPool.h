// Add ThreadPool utility class
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

class ThreadPool {
public:
    ThreadPool(size_t numThreads = 1);
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // wait until queue has space
            queueSpace.wait(lock, [this] {
                return stop || newTaskCondition();
            });

            if (stop) throw std::runtime_error("ThreadManager stopped");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    
    ~ThreadPool();
    void setCondition(std::function<bool()> condition);
    bool hasSpace() const { return tasks.size() < maxQueueSize;};
    bool isEmpty() const { return tasks.empty();};
    size_t queueSize() const { return tasks.size();};

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable queueSpace;
    bool stop;
    size_t maxQueueSize;

    std::function<bool()> newTaskCondition;
};