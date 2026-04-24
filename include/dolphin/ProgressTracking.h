#pragma once

#include <atomic>
#include <functional>

using progressCallbackFn = std::function<void(std::atomic<float>& counter, float max)>;

class ProgressTracking{
public:
    ProgressTracking() = default;
    ProgressTracking(float max) : max(max){}
    void setCallback(progressCallbackFn callback) {this->progressCallback = callback;}
    void setMax(float max) {this->max = max;}
    void reset() {counter.store(0);}

    void add(float value){
        counter += value;
        if(mutex.try_lock()) {progressCallback(counter, max); mutex.unlock();}
    }
private:
    float max;
    std::atomic<float> counter{0};
    std::mutex mutex;
    progressCallbackFn progressCallback;
};

