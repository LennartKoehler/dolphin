#include <gtest/gtest.h>
#include "dolphin/ThreadPool.h"
#include "dolphin/Logging.h"
#include <atomic>
#include <chrono>
#include <vector>

class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(ThreadPoolTest, SingleTask) {
    ThreadPool pool(1);
    auto future = pool.enqueue([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, MultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.enqueue([i] { return i * 2; }));
    }
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(futures[i].get(), i * 2);
    }
}

TEST_F(ThreadPoolTest, TaskWithArguments) {
    ThreadPool pool(2);
    auto future = pool.enqueue([](int a, int b) { return a + b; }, 3, 4);
    EXPECT_EQ(future.get(), 7);
}

TEST_F(ThreadPoolTest, VoidReturnType) {
    ThreadPool pool(1);
    std::atomic<int> counter{0};
    auto future = pool.enqueue([&counter] { counter = 100; });
    future.get();
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(ThreadPoolTest, ConcurrentEnqueue) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    std::atomic<int> total{0};

    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&total, i] { total += 1; return i; }));
    }

    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(total.load(), 100);
}

TEST_F(ThreadPoolTest, ThreadInitFunction) {
    std::atomic<int> initCount{0};
    ThreadPool pool(4, [&initCount] { initCount++; });
    auto future = pool.enqueue([] { return 1; });
    future.get();
    EXPECT_EQ(initCount.load(), 4);
}

TEST_F(ThreadPoolTest, SetCondition) {
    ThreadPool pool(2);
    pool.setCondition([] { return true; });
    auto future = pool.enqueue([] { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, ReduceNumberThreads) {
    ThreadPool pool(4);
    auto f1 = pool.enqueue([] { return 1; });
    f1.get();
    bool result = pool.reduceNumberThreads(2);
    auto f2 = pool.enqueue([] { return 2; });
    EXPECT_EQ(f2.get(), 2);
}

TEST_F(ThreadPoolTest, ExceptionPropagation) {
    ThreadPool pool(1);
    auto future = pool.enqueue([]() -> int {
        throw std::runtime_error("test error");
    });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadPoolTest, SingleThread) {
    ThreadPool pool(1);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 5; i++) {
        futures.push_back(pool.enqueue([i] { return i; }));
    }
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(futures[i].get(), i);
    }
}

TEST_F(ThreadPoolTest, ManyThreads) {
    ThreadPool pool(8);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; i++) {
        futures.push_back(pool.enqueue([&counter] { counter++; }));
    }
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter.load(), 1000);
}
