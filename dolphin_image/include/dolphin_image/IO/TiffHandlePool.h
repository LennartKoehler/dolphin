/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knoll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once

#include <tiffio.h>
#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>
#include "dolphin_image/IO/TiffExceptions.h"

class TiffHandlePool {
public:
    explicit TiffHandlePool(const std::string& filename, size_t poolSize)
        : filename_(filename) {
        for (size_t i = 0; i < poolSize; ++i) {
            TIFF* tif = TIFFOpen(filename.c_str(), "r");
            if (!tif) {
                for (TIFF* h : allHandles_) {
                    if (h) TIFFClose(h);
                }
                throw TiffFileOpenException(filename);
            }
            allHandles_.push_back(tif);
            available_.push(tif);
        }
    }

    ~TiffHandlePool() {
        for (TIFF* tif : allHandles_) {
            if (tif) TIFFClose(tif);
        }
    }

    TiffHandlePool(const TiffHandlePool&) = delete;
    TiffHandlePool& operator=(const TiffHandlePool&) = delete;

    class HandleGuard {
    public:
        HandleGuard(TIFF* tif, TiffHandlePool* pool)
            : tif_(tif), pool_(pool) {}

        ~HandleGuard() {
            if (pool_) pool_->release(tif_);
        }

        HandleGuard(const HandleGuard&) = delete;
        HandleGuard& operator=(const HandleGuard&) = delete;

        HandleGuard(HandleGuard&& other) noexcept
            : tif_(other.tif_), pool_(other.pool_) {
            other.tif_ = nullptr;
            other.pool_ = nullptr;
        }

        HandleGuard& operator=(HandleGuard&& other) noexcept {
            if (this != &other) {
                if (pool_) pool_->release(tif_);
                tif_ = other.tif_;
                pool_ = other.pool_;
                other.tif_ = nullptr;
                other.pool_ = nullptr;
            }
            return *this;
        }

        TIFF* get() const { return tif_; }
        TIFF* operator->() const { return tif_; }

    private:
        TIFF* tif_;
        TiffHandlePool* pool_;
    };

    HandleGuard acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !available_.empty(); });
        TIFF* tif = available_.front();
        available_.pop();
        return HandleGuard(tif, this);
    }

    size_t size() const { return allHandles_.size(); }

private:
    void release(TIFF* tif) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push(tif);
        cv_.notify_one();
    }

    std::string filename_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<TIFF*> available_;
    std::vector<TIFF*> allHandles_;
};
