#include "CPUBackend.h"
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <cstring>
#include <cassert>

#define FFTW_CHECK(plan, operation) { \
    if (plan == nullptr) { \
        throw std::runtime_error(std::string("FFTW error: ") + operation + " - plan creation failed"); \
    } \
}

#define FFTW_MALLOC_CHECK(ptr, size) { \
    if (ptr == nullptr) { \
        throw std::runtime_error(std::string("FFTW malloc failed for size: ") + std::to_string(size)); \
    } \
}

#define NULL_PTR_CHECK(ptr, context) { \
    if (ptr == nullptr) { \
        throw std::runtime_error(std::string("Null pointer in ") + context); \
    } \
}

// CPUBackendMemoryManager implementation
bool CPUBackendMemoryManager::isOnDevice(void* ptr) {
    // For CPU backend, all valid pointers are "on device"
    return ptr != nullptr;
}

void CPUBackendMemoryManager::allocateMemoryOnDevice(ComplexData& data) {
    if (data.data != nullptr) {
        return; // Already allocated
    }
    
    data.data = (complex*)fftw_malloc(sizeof(complex) * data.size.volume);
    FFTW_MALLOC_CHECK(data.data, sizeof(complex) * data.size.volume);
    data.backend = this;
}

ComplexData CPUBackendMemoryManager::allocateMemoryOnDevice(const RectangleShape& shape) {
    ComplexData result{this, (complex*)fftw_malloc(sizeof(complex) * shape.volume), shape}; 
    FFTW_MALLOC_CHECK(result.data, sizeof(complex) * shape.volume);
    return result;
}

ComplexData CPUBackendMemoryManager::moveDataToDevice(const ComplexData& srcdata) {
    NULL_PTR_CHECK(srcdata.data, "moveDataToDevice - source data");
    ComplexData result = allocateMemoryOnDevice(srcdata.size);
    std::memcpy(result.data, srcdata.data, srcdata.size.volume * sizeof(complex));
    return result;
}

ComplexData CPUBackendMemoryManager::moveDataFromDevice(const ComplexData& srcdata) {
    NULL_PTR_CHECK(srcdata.data, "moveDataFromDevice - source data");
    ComplexData result = allocateMemoryOnDevice(srcdata.size);
    std::memcpy(result.data, srcdata.data, srcdata.size.volume * sizeof(complex));
    return result;
}

ComplexData CPUBackendMemoryManager::copyData(const ComplexData& srcdata) {
    NULL_PTR_CHECK(srcdata.data, "copyData - source data");
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    std::memcpy(destdata.data, srcdata.data, srcdata.size.volume * sizeof(complex));
    return destdata;
}

void CPUBackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData){
    NULL_PTR_CHECK(srcData.data, "memCopy - source data");
    NULL_PTR_CHECK(destData.data, "memCopy - destination data");
    if (srcData.size.volume != destData.size.volume) {
        throw std::runtime_error("Size mismatch in memCopy");
    }
    std::memcpy(destData.data, srcData.data, srcData.size.volume * sizeof(complex));
}

void CPUBackendMemoryManager::freeMemoryOnDevice(ComplexData& data){
    NULL_PTR_CHECK(data.data, "freeMemoryOnDevice - data pointer");
    fftw_free(data.data);
    data.data = nullptr;
} 

size_t CPUBackendMemoryManager::getAvailableMemory() {
    // For CPU backend, return available system memory
    std::unique_lock<std::mutex>lock(backendMutex);
    size_t memory;
    #ifdef __linux__
        #include <unistd.h>
        long pagesize = sysconf(_SC_PAGESIZE);
        long pages = sysconf(_SC_AVPHYS_PAGES);
        if (pagesize > 0 && pages > 0) {
            memory = static_cast<size_t>(pagesize) * static_cast<size_t>(pages);
        }
    #elif _WIN32
        #include <windows.h>
        MEMORYSTATUSEX status;
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        memory = static_cast<size_t>(status.ullAvailPhys);
    #endif
    assert(memory != 0 && "[ERROR] Something went wrong while trying to get available memory");
    return memory;
}


// #####################################################################################################
// CPUDeconvolutionBackend implementation
CPUDeconvolutionBackend::CPUDeconvolutionBackend() {
}

CPUDeconvolutionBackend::~CPUDeconvolutionBackend() {
    destroyFFTPlans();
}

void CPUDeconvolutionBackend::init(const RectangleShape& shape) {
    try {
        initializeFFTPlans(shape);
        std::cout << "[STATUS] CPU backend initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CPU initialization: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::cleanup() {
    try {
        destroyFFTPlans();
        std::cout << "[STATUS] CPU backend postprocessing completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CPU postprocessing: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::initializeFFTPlans(const RectangleShape& cube) {
    std::unique_lock<std::mutex>lock(backendMutex);
    if (plansInitialized_) return;
    
    try {
        // Allocate temporary memory for plan creation
        complex* temp = (complex*)fftw_malloc(sizeof(complex) * cube.volume);
        FFTW_MALLOC_CHECK(temp, sizeof(complex) * cube.volume);
        
        // Create forward FFT plan
        this->forwardPlan = fftw_plan_dft_3d(cube.depth, cube.height, cube.width,
                                            temp, temp, FFTW_FORWARD, FFTW_MEASURE);
        FFTW_CHECK(this->forwardPlan, "Forward FFT plan creation");
       
        // Create backward FFT plan
        this->backwardPlan = fftw_plan_dft_3d(cube.depth, cube.height, cube.width,
                                             temp, temp, FFTW_BACKWARD, FFTW_MEASURE);
        FFTW_CHECK(this->backwardPlan, "Backward FFT plan creation");
            
        fftw_free(temp);
        plansInitialized_ = true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in FFT plan initialization: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::destroyFFTPlans() {
    std::unique_lock<std::mutex> lock(backendMutex);
    if (plansInitialized_) {
        if (forwardPlan) {
            fftw_destroy_plan(forwardPlan);
            forwardPlan = nullptr;
        }
        if (backwardPlan) {
            fftw_destroy_plan(backwardPlan);
            backwardPlan = nullptr;
        }
        plansInitialized_ = false;
    }
}

// FFT Operations
void CPUDeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) {
    try {
        NULL_PTR_CHECK(forwardPlan, "forwardFFT - FFT plan");
        NULL_PTR_CHECK(in.data, "forwardFFT - input data");
        NULL_PTR_CHECK(out.data, "forwardFFT - output data");
        fftw_execute_dft(forwardPlan, reinterpret_cast<fftw_complex*>(in.data), reinterpret_cast<fftw_complex*>(out.data));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in forwardFFT: " << e.what() << std::endl;
        throw;
    }
}

void CPUDeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) {
    try {
        NULL_PTR_CHECK(backwardPlan, "backwardFFT - FFT plan");
        NULL_PTR_CHECK(in.data, "backwardFFT - input data");
        NULL_PTR_CHECK(out.data, "backwardFFT - output data");
        fftw_execute_dft(backwardPlan, reinterpret_cast<fftw_complex*>(in.data), reinterpret_cast<fftw_complex*>(out.data));
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in backwardFFT: " << e.what() << std::endl;
        throw;
    }
}

// Shift Operations
void CPUDeconvolutionBackend::octantFourierShift(ComplexData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int halfDepth = depth / 2;

        //#pragma omp parallel for collapse(3)
        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = ((z + halfDepth) % depth) * height * width + 
                              ((y + halfHeight) % height) * width + 
                              ((x + halfWidth) % width);

                    if (idx1 != idx2) {
                        std::swap(data.data[idx1][0], data.data[idx2][0]);
                        std::swap(data.data[idx1][1], data.data[idx2][1]);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in octantFourierShift: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::inverseQuadrantShift(ComplexData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int halfDepth = depth / 2;

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < halfHeight; ++y) {
                for (int x = 0; x < halfWidth; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x + halfWidth);

                    std::swap(data.data[idx1][0], data.data[idx2][0]);
                    std::swap(data.data[idx1][1], data.data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < halfHeight; ++y) {
                for (int x = halfWidth; x < width; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x - halfWidth);

                    std::swap(data.data[idx1][0], data.data[idx2][0]);
                    std::swap(data.data[idx1][1], data.data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = halfHeight; y < height; ++y) {
                for (int x = 0; x < halfWidth; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x + halfWidth);

                    std::swap(data.data[idx1][0], data.data[idx2][0]);
                    std::swap(data.data[idx1][1], data.data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = halfHeight; y < height; ++y) {
                for (int x = halfWidth; x < width; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x - halfWidth);

                    std::swap(data.data[idx1][0], data.data[idx2][0]);
                    std::swap(data.data[idx1][1], data.data[idx2][1]);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in inverseQuadrantShift: " << e.what() << std::endl;
    }
}

// Complex Arithmetic Operations
void CPUDeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) {
    try {
        NULL_PTR_CHECK(a.data, "complexMultiplication - input a");
        NULL_PTR_CHECK(b.data, "complexMultiplication - input b");
        NULL_PTR_CHECK(result.data, "complexMultiplication - result");
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in complexMultiplication");
        }

        //#pragma omp parallel for simd
        for (int i = 0; i < a.size.volume; ++i) {
            double real_a = a.data[i][0];
            double imag_a = a.data[i][1];
            double real_b = b.data[i][0];
            double imag_b = b.data[i][1];

            result.data[i][0] = real_a * real_b - imag_a * imag_b;
            result.data[i][1] = real_a * imag_b + imag_a * real_b;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplication: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
    try {
        NULL_PTR_CHECK(a.data, "complexDivision - input a");
        NULL_PTR_CHECK(b.data, "complexDivision - input b");
        NULL_PTR_CHECK(result.data, "complexDivision - result");
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in complexDivision");
        }

        //#pragma omp parallel for
        for (int i = 0; i < a.size.volume; ++i) {
            double real_a = a.data[i][0];
            double imag_a = a.data[i][1];
            double real_b = b.data[i][0];
            double imag_b = b.data[i][1];

            double denominator = real_b * real_b + imag_b * imag_b;

            if (denominator < epsilon) {
                result.data[i][0] = 0.0;
                result.data[i][1] = 0.0;
            } else {
                result.data[i][0] = (real_a * real_b + imag_a * imag_b) / denominator;
                result.data[i][1] = (imag_a * real_b - real_a * imag_b) / denominator;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivision: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) {
    try {
        NULL_PTR_CHECK(a.data, "complexAddition - input a");
        NULL_PTR_CHECK(b.data, "complexAddition - input b");
        NULL_PTR_CHECK(result.data, "complexAddition - result");
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in complexAddition");
        }

        for (int i = 0; i < a.size.volume; ++i) {
            result.data[i][0] = a.data[i][0] + b.data[i][0];
            result.data[i][1] = a.data[i][1] + b.data[i][1];
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexAddition: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) {
    try {
        NULL_PTR_CHECK(a.data, "scalarMultiplication - input a");
        NULL_PTR_CHECK(result.data, "scalarMultiplication - result");
        if (a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in scalarMultiplication");
        }

        for (int i = 0; i < a.size.volume; ++i) {
            result.data[i][0] = a.data[i][0] * scalar;
            result.data[i][1] = a.data[i][1] * scalar;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in scalarMultiplication: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) {
    try {
        NULL_PTR_CHECK(a.data, "complexMultiplicationWithConjugate - input a");
        NULL_PTR_CHECK(b.data, "complexMultiplicationWithConjugate - input b");
        NULL_PTR_CHECK(result.data, "complexMultiplicationWithConjugate - result");
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in complexMultiplicationWithConjugate");
        }

        //#pragma omp parallel for
        for (int i = 0; i < a.size.volume; ++i) {
            double real_a = a.data[i][0];
            double imag_a = a.data[i][1];
            double real_b = b.data[i][0];
            double imag_b = -b.data[i][1];  // Conjugate

            result.data[i][0] = real_a * real_b - imag_a * imag_b;
            result.data[i][1] = real_a * imag_b + imag_a * real_b;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplicationWithConjugate: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
    try {
        NULL_PTR_CHECK(a.data, "complexDivisionStabilized - input a");
        NULL_PTR_CHECK(b.data, "complexDivisionStabilized - input b");
        NULL_PTR_CHECK(result.data, "complexDivisionStabilized - result");
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            throw std::runtime_error("Size mismatch in complexDivisionStabilized");
        }

        //#pragma omp parallel for
        for (int i = 0; i < a.size.volume; ++i) {
            double real_a = a.data[i][0];
            double imag_a = a.data[i][1];
            double real_b = b.data[i][0];
            double imag_b = b.data[i][1];

            double mag = std::max(epsilon, real_b * real_b + imag_b * imag_b);

            result.data[i][0] = (real_a * real_b + imag_a * imag_b) / mag;
            result.data[i][1] = (imag_a * real_b - real_a * imag_b) / mag;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivisionStabilized: " << e.what() << std::endl;
    }
}

// Specialized Functions
void CPUDeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) {
    try {
        int width = psf.size.width;
        int height = psf.size.height;
        int depth = psf.size.depth;

        for (int z = 0; z < depth; ++z) {
            float wz = 2 * M_PI * z / depth;
            for (int y = 0; y < height; ++y) {
                float wy = 2 * M_PI * y / height;
                for (int x = 0; x < width; ++x) {
                    float wx = 2 * M_PI * x / width;
                    float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

                    int index = (z * height + y) * width + x;

                    laplacian.data[index][0] = psf.data[index][0] * laplacian_value;
                    laplacian.data[index][1] = psf.data[index][1] * laplacian_value;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in calculateLaplacianOfPSF: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::normalizeImage(ComplexData& resultImage, double epsilon) {
    try {
        double max_val = 0.0, max_val2 = 0.0;
        for (int j = 0; j < resultImage.size.volume; j++) {
            max_val = std::max(max_val, resultImage.data[j][0]);
            max_val2 = std::max(max_val2, resultImage.data[j][1]);
        }
        for (int j = 0; j < resultImage.size.volume; j++) {
            resultImage.data[j][0] /= (max_val + epsilon);
            resultImage.data[j][1] /= (max_val2 + epsilon);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeImage: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::rescaledInverse(ComplexData& data, double cubeVolume) {
    try {
        for (int i = 0; i < data.size.volume; ++i) {
            data.data[i][0] /= cubeVolume;
            data.data[i][1] /= cubeVolume;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in rescaledInverse: " << e.what() << std::endl;
    }
}

// Debug functions
void CPUDeconvolutionBackend::hasNAN(const ComplexData& data) {
    int nanCount = 0, infCount = 0;
    double minReal = std::numeric_limits<double>::max();
    double maxReal = std::numeric_limits<double>::lowest();
    double minImag = std::numeric_limits<double>::max();
    double maxImag = std::numeric_limits<double>::lowest();
    
    for (int i = 0; i < data.size.volume; i++) {
        double real = data.data[i][0];
        double imag = data.data[i][1];
        
        // Check for NaN
        if (std::isnan(real) || std::isnan(imag)) {
            nanCount++;
            if (nanCount <= 10) { // Only print first 10
                std::cout << "NaN at index " << i << ": (" << real << ", " << imag << ")" << std::endl;
            }
        }
        
        // Check for infinity
        if (std::isinf(real) || std::isinf(imag)) {
            infCount++;
            if (infCount <= 10) {
                std::cout << "Inf at index " << i << ": (" << real << ", " << imag << ")" << std::endl;
            }
        }
        
        // Track min/max for valid values
        if (std::isfinite(real)) {
            minReal = std::min(minReal, real);
            maxReal = std::max(maxReal, real);
        }
        if (std::isfinite(imag)) {
            minImag = std::min(minImag, imag);
            maxImag = std::max(maxImag, imag);
        }
    }
    
    std::cout << "[DEBUG] Data stats - NaN: " << nanCount << ", Inf: " << infCount << std::endl;
    std::cout << "[DEBUG] Real range: [" << minReal << ", " << maxReal << "]" << std::endl;
    std::cout << "[DEBUG] Imag range: [" << minImag << ", " << maxImag << "]" << std::endl;
}

// Layer and Visualization Functions
void CPUDeconvolutionBackend::reorderLayers(ComplexData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        int layerSize = width * height;
        int halfDepth = depth / 2;
        
        complex* temp = (complex*)fftw_malloc(sizeof(complex) * data.size.volume);
        FFTW_MALLOC_CHECK(temp, sizeof(complex) * data.size.volume);

        int destIndex = 0;

        // Copy the middle layer to the first position
        std::memcpy(temp + destIndex * layerSize, data.data + halfDepth * layerSize, sizeof(complex) * layerSize);
        destIndex++;

        // Copy the layers after the middle layer
        for (int z = halfDepth + 1; z < depth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex) * layerSize);
            destIndex++;
        }

        // Copy the layers before the middle layer
        for (int z = 0; z < halfDepth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex) * layerSize);
            destIndex++;
        }

        // Copy reordered data back to the original array
        std::memcpy(data.data, temp, sizeof(complex) * data.size.volume);
        fftw_free(temp);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in reorderLayers: " << e.what() << std::endl;
    }
}

// Gradient and TV Functions
void CPUDeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) {
    try {
        int width = image.size.width;
        int height = image.size.height;
        int depth = image.size.depth;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width - 1; ++x) {
                    int index = z * height * width + y * width + x;
                    int nextIndex = index + 1;

                    gradX.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradX.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                }

                int lastIndex = z * height * width + y * width + (width - 1);
                gradX.data[lastIndex][0] = 0.0;
                gradX.data[lastIndex][1] = 0.0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientX: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) {
    try {
        int width = image.size.width;
        int height = image.size.height;
        int depth = image.size.depth;
        
        for (int z = 0; z < depth; ++z) {
            for (int y = 0; y < height - 1; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    int nextIndex = index + width;

                    gradY.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradY.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                }
            }

            for (int x = 0; x < width; ++x) {
                int lastIndex = z * height * width + (height - 1) * width + x;
                gradY.data[lastIndex][0] = 0.0;
                gradY.data[lastIndex][1] = 0.0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientY: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) {
    try {
        int width = image.size.width;
        int height = image.size.height;
        int depth = image.size.depth;
        
        for (int z = 0; z < depth - 1; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    int nextIndex = index + height * width;

                    gradZ.data[index][0] = image.data[index][0] - image.data[nextIndex][0];
                    gradZ.data[index][1] = image.data[index][1] - image.data[nextIndex][1];
                }
            }
        }

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int lastIndex = (depth - 1) * height * width + y * width + x;
                gradZ.data[lastIndex][0] = 0.0;
                gradZ.data[lastIndex][1] = 0.0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientZ: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) {
    try {
        int nxy = gx.size.width * gx.size.height;

        for (int z = 0; z < gx.size.depth; ++z) {
            for (int i = 0; i < nxy; ++i) {
                int index = z * nxy + i;

                double dx = gx.data[index][0];
                double dy = gy.data[index][0];
                double dz = gz.data[index][0];

                tv.data[index][0] = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
                tv.data[index][1] = 1.0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in computeTV: " << e.what() << std::endl;
    }
}

void CPUDeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) {
    try {
        int nxy = gradX.size.width * gradX.size.height;

        for (int z = 0; z < gradX.size.depth; ++z) {
            for (int i = 0; i < nxy; ++i) {
                int index = z * nxy + i;

                double norm = std::sqrt(
                    gradX.data[index][0] * gradX.data[index][0] + gradX.data[index][1] * gradX.data[index][1] +
                    gradY.data[index][0] * gradY.data[index][0] + gradY.data[index][1] * gradY.data[index][1] +
                    gradZ.data[index][0] * gradZ.data[index][0] + gradZ.data[index][1] * gradZ.data[index][1]
                );

                norm = std::max(norm, epsilon);

                gradX.data[index][0] /= norm;
                gradX.data[index][1] /= norm;
                gradY.data[index][0] /= norm;
                gradY.data[index][1] /= norm;
                gradZ.data[index][0] /= norm;
                gradZ.data[index][1] /= norm;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeTV: " << e.what() << std::endl;
    }
}

extern "C" IDeconvolutionBackend* createDeconvolutionBackend() {
    return new CPUDeconvolutionBackend();
}

extern "C" IBackendMemoryManager* createBackendMemoryManager() {
    return new CPUBackendMemoryManager();
}