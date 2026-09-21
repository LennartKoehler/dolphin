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

#include <atomic>
#include <functional>
#include <mutex>

using progressCallbackFn = std::function<void(std::atomic<float>& counter, float max)>;

class ProgressTracking{
public:
    ProgressTracking() = default;
    ProgressTracking(float max) : max(max){}
    void setCallback(progressCallbackFn callback) {this->progressCallback = callback;}
    void setMax(float max) {this->max = max;}
    void reset() {counter.store(0);}

    void add(float value){
        float old = counter.load(std::memory_order_relaxed);
        while (!counter.compare_exchange_weak(old, old + value, std::memory_order_relaxed)) {}
        if(mutex.try_lock()) {
            if (progressCallback) progressCallback(counter, max);
            mutex.unlock();
        }
    }
private:
    float max;
    std::atomic<float> counter{0};
    std::mutex mutex;
    progressCallbackFn progressCallback;
};

