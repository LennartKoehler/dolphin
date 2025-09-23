#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <atomic>
#include <complex>
#include <limits>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>

// Common GPU/CPU type definitions
#ifdef CUDA_AVAILABLE
#include <cufftw.h>
#include <cuda_runtime.h>
#else
#include <fftw3.h>
#endif

// Forward declarations
namespace cv {
    class Mat;
}

namespace DeconvolutionHelpers {

    // ================================
    // Data Types and Type Definitions
    // ================================

    // Common complex type for cross-platform compatibility
    template<typename T = double>
    using Complex = std::complex<T>;

    // Memory handle types for different backends
#ifdef CUDA_AVAILABLE
    using GPUComplexArray = void*; // cuda pointer
    using CPUMemoryHandle = void*;
#else
    using GPUComplexArray = void*; // fallback to CPU
    using CPUMemoryHandle = void*;
#endif

    // ================================
    // Memory Management Helpers
    // ================================

    /**
     * @brief Memory allocation tracker for monitoring peak memory usage
     */
    class MemoryTracker {
    public:
        static MemoryTracker& getInstance();
        
        void allocate(size_t size);
        void deallocate(size_t size);
        size_t getCurrentUsage() const;
        size_t getPeakUsage() const;
        void reset();
        void printMemoryUsage(const std::string& operation) const;
        static std::string formatBytesStatic(size_t bytes);

    private:
        MemoryTracker() = default;
        std::mutex m_mutex;
        std::atomic<size_t> m_currentUsage{0};
        std::atomic<size_t> m_peakUsage{0};
    };

    /**
     * @brief RAII wrapper for memory allocation
     */
    template<typename T = void>
    class AlignedBuffer {
    public:
        explicit AlignedBuffer(size_t size) : m_size(size) {
            if (size > 0) {
                m_data = allocateAligned<T>(size);
                if (!m_data) {
                    throw std::runtime_error("Failed to allocate aligned memory");
                }
            }
        }

        ~AlignedBuffer() {
            if (m_data) {
                deallocateAligned(m_data);
            }
        }

        // Non-copyable
        AlignedBuffer(const AlignedBuffer&) = delete;
        AlignedBuffer& operator=(const AlignedBuffer&) = delete;

        // Movable
        AlignedBuffer(AlignedBuffer&& other) noexcept : m_data(other.m_data), m_size(other.m_size) {
            other.m_data = nullptr;
            other.m_size = 0;
        }

        AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
            if (this != &other) {
                if (m_data) deallocateAligned(m_data);
                m_data = other.m_data;
                m_size = other.m_size;
                other.m_data = nullptr;
                other.m_size = 0;
            }
            return *this;
        }

        T* get() const { return m_data; }
        T* operator->() const { return m_data; }
        T& operator[](size_t index) { return m_data[index]; }
        const T& operator[](size_t index) const { return m_data[index]; }
        size_t size() const { return m_size; }
        bool empty() const { return m_size == 0; }

    private:
        T* m_data = nullptr;
        size_t m_size = 0;

        static T* allocateAligned(size_t size);
        static void deallocateAligned(void* ptr);
    };

    /**
     * @brief Allocate aligned memory (64-byte alignment for performance)
     */
    template<typename T>
    T* AlignedBuffer<T>::allocateAligned(size_t size) {
        const size_t alignment = 64;
        T* ptr = nullptr;
        
#ifdef _POSIX_C_SOURCE
        posix_memalign(reinterpret_cast<void**>(&ptr), alignment, size * sizeof(T));
        return ptr;
#elif defined(_WIN32)
        ptr = static_cast<T>(_aligned_malloc(size * sizeof(T), alignment));
        return ptr;
#else
        // Fallback to regular allocation
        return static_cast<T*>(malloc(size * sizeof(T)));
#endif
    }

    /**
     * @brief Deallocate aligned memory
     */
    template<typename T>
    void AlignedBuffer<T>::deallocateAligned(void* ptr) {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
    }

    // ================================
    // Data Validation and Processing Helpers
    // ================================

    /**
     * @brief Check if a complex number is valid (not NaN or Inf)
     */
    template<typename T>
    inline bool isComplexValid(const std::complex<T>& value) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
    }

    /**
     * @brief Check if FFTW complex array contains valid values
     */
    template<typename T = double>
    bool isValidComplexArray(const std::complex<T>* data, size_t size, std::string& errorDetails) {
        if (!data) {
            errorDetails = "Null pointer";
            return false;
        }

        size_t invalidCount = 0;
        size_t nanCount = 0;
        size_t infCount = 0;

        for (size_t i = 0; i < size; ++i) {
            if (!isComplexValid(data[i])) {
                invalidCount++;
                if (std::isnan(data[i].real()) || std::isnan(data[i].imag())) {
                    nanCount++;
                }
                if (std::isinf(data[i].real()) || std::isinf(data[i].imag())) {
                    infCount++;
                }
            }
        }

        if (invalidCount > 0) {
            errorDetails = "Invalid values found: " + std::to_string(invalidCount) + 
                          " total (" + std::to_string(nanCount) + " NaN, " + 
                          std::to_string(infCount) + " Inf)";
            return false;
        }

        return true;
    }

    /**
     * @brief Check if FFTW complex array contains valid values
     */
    inline bool isValidFFTWComplexArray(complex* data, size_t size, std::string& errorDetails) {
        if (!data) {
            errorDetails = "Null pointer";
            return false;
        }

        size_t invalidCount = 0;
        size_t nanCount = 0;
        size_t infCount = 0;

        for (size_t i = 0; i < size; ++i) {
            double real = data[i][0];
            double imag = data[i][1];
            
            if (!std::isfinite(real) || !std::isfinite(imag)) {
                invalidCount++;
                if (std::isnan(real) || std::isnan(imag)) {
                    nanCount++;
                }
                if (std::isinf(real) || std::isinf(imag)) {
                    infCount++;
                }
            }
        }

        if (invalidCount > 0) {
            errorDetails = "Invalid values found: " + std::to_string(invalidCount) + 
                          " total (" + std::to_string(nanCount) + " NaN, " + 
                          std::to_string(infCount) + " Inf)";
            return false;
        }

        return true;
    }

    /**
     * @brief Normalize a complex array by dividing by its sum
     */
    template<typename T = double>
    void normalizeComplexArray(std::complex<T>* data, size_t size, double epsilon = 1e-12) {
        if (!data || size == 0) return;

        T sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += std::abs(data[i]);
        }

        if (sum > epsilon) {
            T normalizationFactor = static_cast<T>(1.0 / sum);
            for (size_t i = 0; i < size; ++i) {
                data[i] *= normalizationFactor;
            }
        } else {
            // Zero out if sum is too small
            std::fill(data, data + size, std::complex<T>(0, 0));
        }
    }

    /**
     * @brief Normalize FFTW complex array
     */
    inline void normalizeFFTWComplexArray(complex* data, size_t size, double epsilon = 1e-12) {
        if (!data || size == 0) return;

        double sum = 0;
        for (size_t i = 0; i < size; ++i) {
            sum += std::sqrt(data[i][0] * data[i][0] + data[i][1] * data[i][1]);
        }

        if (sum > epsilon) {
            double normalizationFactor = 1.0 / sum;
            for (size_t i = 0; i < size; ++i) {
                data[i][0] *= normalizationFactor;
                data[i][1] *= normalizationFactor;
            }
        } else {
            // Zero out if sum is too small
            for (size_t i = 0; i < size; ++i) {
                data[i][0] = 0;
                data[i][1] = 0;
            }
        }
    }

    /**
     * @brief Copy complex array from source to destination
     */
    template<typename T = double>
    void copyComplexArray(const std::complex<T>* source, std::complex<T>* destination, size_t size) {
        if (source && destination && size > 0) {
            std::copy(source, source + size, destination);
        }
    }

    /**
     * @brief Copy FFTW complex array
     */
    inline void copyFFTWComplexArray(const complex* source, complex* destination, size_t size) {
        if (source && destination && size > 0) {
            std::copy(source, source + size, destination);
        }
    }

    /**
     * @brief Apply threshold to complex values below epsilon
     */
    template<typename T = double>
    void thresholdComplexArray(std::complex<T>* data, size_t size, T thresholdValue) {
        if (!data || size == 0) return;

        for (size_t i = 0; i < size; ++i) {
            T magnitude = std::abs(data[i]);
            if (magnitude < thresholdValue) {
                data[i] = std::complex<T>(0, 0);
            }
        }
    }

    /**
     * @brief Apply threshold to FFTW complex values below epsilon
     */
    inline void thresholdFFTWComplexArray(complex* data, size_t size, double thresholdValue) {
        if (!data || size == 0) return;

        for (size_t i = 0; i < size; ++i) {
            double magnitude = std::sqrt(data[i][0] * data[i][0] + data[i][1] * data[i][1]);
            if (magnitude < thresholdValue) {
                data[i][0] = 0;
                data[i][1] = 0;
            }
        }
    }

    // ================================
    // Configuration and Setup Helpers
    // ================================

    /**
     * @brief Configuration validation bundle
     */
    struct ValidationResult {
        bool valid;
        std::vector<std::string> warnings;
        std::vector<std::string> errors;
        
        ValidationResult() : valid(true) {}
        
        void addWarning(const std::string& warning) {
            warnings.push_back(warning);
        }
        
        void addError(const std::string& error) {
            errors.push_back(error);
            valid = false;
        }
        
        void merge(const ValidationResult& other) {
            warnings.insert(warnings.end(), other.warnings.begin(), other.warnings.end());
            errors.insert(errors.end(), other.errors.begin(), other.errors.end());
            valid = valid && other.valid;
        }
    };

    /**
     * @brief Validate deconvolution configuration parameters
     */
    ValidationResult validateConfiguration(
        int iterations,
        double epsilon,
        int subimageSize,
        int psfSafetyBorder,
        int cubeSize,
        bool useGrid,
        int borderType
    );

    /**
     * @brief Log configuration summary
     */
    void logConfigurationSummary(
        int iterations,
        double epsilon,
        int subimageSize,
        int psfSafetyBorder,
        int cubeSize,
        bool useGrid,
        int borderType,
        const std::string& algorithmName = "Unknown"
    );

    /**
     * @brief Calculate optimal cube size based on PSF and image dimensions
     */
    int calculateOptimalCubeSize(
        int psfWidth,
        int psfHeight,
        int psfDepth,
        int imageWidth,
        int imageHeight,
        int imageDepth,
        int minCubeSize = 32,
        int maxCubeSize = 256
    );

    /**
     * @brief Validate image and PSF dimensions compatibility
     */
    ValidationResult validateImageAndPsfDimensions(
        int imageWidth,
        int imageHeight,
        int imageDepth,
        int psfWidth,
        int psfHeight,
        int psfDepth
    );

    // ================================
    // Performance Monitoring Helpers
    // ================================

    /**
     * @brief High-resolution timer for performance measurement
     */
    class PerformanceTimer {
    public:
        PerformanceTimer() { start(); }
        
        void start() {
            m_startTime = std::chrono::high_resolution_clock::now();
        }
        
        void stop() {
            m_endTime = std::chrono::high_resolution_clock::now();
        }
        
        double elapsedMilliseconds() const {
            return std::chrono::duration<double, std::milli>
                (m_endTime - m_startTime).count();
        }
        
        double elapsedSeconds() const {
            return std::chrono::duration<double>
                (m_endTime - m_startTime).count();
        }
        
        void reset() {
            start();
        }

    private:
        std::chrono::high_resolution_clock::time_point m_startTime;
        std::chrono::high_resolution_clock::time_point m_endTime;
    };

    /**
     * @brief Progress tracking with thread-safe reporting
     */
    class ProgressTracker {
    public:
        ProgressTracker(size_t totalItems, bool printProgress = true);
        
        void incrementProgress();
        void setProgress(size_t currentProgress);
        double getProgressPercentage() const;
        void printProgress() const;
        
        static void setProgressPattern(const std::string& pattern);
        
    private:
        std::atomic<size_t> m_currentProgress{0};
        const size_t m_totalItems;
        const bool m_printProgress;
        static std::string m_progressPattern;
        static std::mutex m_printMutex;
    };

    /**
     * @brief Simple performance profiler for function timing
     */
    template<typename Func>
    auto profileFunction(const std::string& name, Func&& func) 
        -> decltype(func()) {
        
        PerformanceTimer timer;
        auto result = func();
        double elapsedMs = timer.elapsedMilliseconds();
        
        std::cout << "[PERF] " << name << " took " << elapsedMs << " ms" << std::endl;
        return result;
    }

    // ================================
    // Mathematical Operations Helpers
    // ================================

    /**
     * @brief Compute sum of squared magnitudes of complex array
     */
    template<typename T = double>
    T sumSquaredMagnitudes(const std::complex<T>* data, size_t size) {
        T sum = 0;
        for (size_t i = 0; i < size; ++i) {
            T magnitude = std::abs(data[i]);
            sum += magnitude * magnitude;
        }
        return sum;
    }

    /**
     * @brief Compute sum of squared magnitudes of FFTW complex array
     */
    inline double sumSquaredFFTWComplexMagnitudes(complex* data, size_t size) {
        double sum = 0;
        for (size_t i = 0; i < size; ++i) {
            double magnitude = std::sqrt(data[i][0] * data[i][0] + data[i][1] * data[i][1]);
            sum += magnitude * magnitude;
        }
        return sum;
    }

    /**
     * @brief Compute mean value of complex array
     */
    template<typename T = double>
    std::complex<T> meanComplexArray(const std::complex<T>* data, size_t size) {
        if (size == 0) return std::complex<T>(0, 0);
        
        std::complex<T> sum(0, 0);
        for (size_t i = 0; i < size; ++i) {
            sum += data[i];
        }
        return sum / static_cast<T>(size);
    }

    /**
     * @brief Compute standard deviation of complex array magnitudes
     */
    template<typename T = double>
    T standardDeviation(const std::complex<T>* data, size_t size, T meanValue = std::numeric_limits<T>::quiet_NaN()) {
        if (size == 0) return 0;

        // Compute mean if not provided
        if (std::isnan(meanValue)) {
            meanValue = std::abs(meanComplexArray(data, size));
        }

        T sumSqDiff = 0;
        for (size_t i = 0; i < size; ++i) {
            T diff = std::abs(data[i]) - meanValue;
            sumSqDiff += diff * diff;
        }

        return std::sqrt(sumSqDiff / static_cast<T>(size));
    }

    /**
     * @brief Find minimum and maximum magnitudes in complex array
     */
    template<typename T = double>
    std::pair<T, T> minMaxMagnitudes(const std::complex<T>* data, size_t size) {
        if (size == 0) return {0, 0};

        T minMag = std::numeric_limits<T>::max();
        T maxMag = std::numeric_limits<T>::min();

        for (size_t i = 0; i < size; ++i) {
            T magnitude = std::abs(data[i]);
            if (magnitude < minMag) minMag = magnitude;
            if (magnitude > maxMag) maxMag = magnitude;
        }

        return {minMag, maxMag};
    }

    /**
     * @brief Apply window function to reduce edge artifacts
     */
    template<typename T = double>
    void applyHammingWindow3D(std::complex<T>* data, int width, int height, int depth) {
        if (!data) return;

        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Hamming window coefficients
                    double windowX = 0.54 - 0.46 * std::cos(2.0 * M_PI * x / (width - 1));
                    double windowY = 0.54 - 0.46 * std::cos(2.0 * M_PI * y / (height - 1));
                    double windowZ = 0.54 - 0.46 * std::cos(2.0 * M_PI * z / (depth - 1));
                    
                    double windowFactor = windowX * windowY * windowZ;
                    
                    size_t index = z * width * height + y * width + x;
                    data[index] *= windowFactor;
                }
            }
        }
    }

    /**
     * @brief Check if value is within reasonable bounds for image data
     */
    template<typename T = double>
    bool isImageDataValid(T value) {
        return std::isfinite(value) && 
               value >= -1e6 && 
               value <= 1e6;
    }

    /**
     * @brief Clean up invalid values in image data
     */
    template<typename T = double>
    void cleanImageData(std::complex<T>* data, size_t size, T threshold = 1e-6) {
        if (!data) return;

        for (size_t i = 0; i < size; ++i) {
            if (!isImageDataValid(data[i].real()) || !isImageDataValid(data[i].imag())) {
                data[i] = std::complex<T>(0, 0);
            }
            
            T magnitude = std::abs(data[i]);
            if (magnitude < threshold) {
                data[i] = std::complex<T>(0, 0);
            }
        }
    }

} // namespace DeconvolutionHelpers