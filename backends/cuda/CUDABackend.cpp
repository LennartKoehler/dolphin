#include "CUDABackend.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>

#include <cassert>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

#define CUFFT_CHECK(call) { \
    cufftResult res = call; \
    if (res != CUFFT_SUCCESS) { \
        throw std::runtime_error("cuFFT error code: " + std::to_string(res)); \
    } \
}



extern "C" IDeconvolutionBackend* createDeconvolutionBackend() {
    return new CUDADeconvolutionBackend();
}

extern "C" IBackendMemoryManager* createBackendMemoryManager() {
    return new CUDABackendMemoryManager();
}



bool CUDABackendMemoryManager::isOnDevice(void* ptr) const {
    if (ptr == nullptr) return false;
    
    cudaPointerAttributes attributes;
    cudaError_t result = cudaPointerGetAttributes(&attributes, ptr);
    
    if (result == cudaSuccess) {
        return (attributes.type == cudaMemoryTypeDevice ||
                attributes.type == cudaMemoryTypeManaged);
    } else if (result == cudaErrorInvalidValue) {
        cudaGetLastError(); // Clear the error
        return false;
    } else {
        CUDA_CHECK(result); // This will throw an exception
        return false; // Never reached
    }
}

void CUDABackendMemoryManager::allocateMemoryOnDevice(ComplexData& data) const {
    if (data.data != nullptr && isOnDevice(data.data)) {
        return; // Already on device
    }
    
    // Allocate CUDA memory
    CUDA_CHECK(cudaMalloc((void**)&data.data, data.size.volume * sizeof(complex)));
    data.backend = this;

}

void CUDABackendMemoryManager::memCopy(const ComplexData& srcData, ComplexData& destData) const {
    // Ensure destination has memory allocated
    if (destData.data == nullptr) {
        allocateMemoryOnDevice(destData);
    }
    
    // Check if sizes match
    if (srcData.size.volume != destData.size.volume) {
        std::cerr << "[ERROR] Size mismatch in moveData" << std::endl;
        return;
    }
    
    // Setup cudaMemcpy3D parameters
    cudaMemcpy3DParms copyParams = {0};
    
    // Source parameters
    copyParams.srcPtr = make_cudaPitchedPtr(
        srcData.data,                           // Source pointer
        srcData.size.width * sizeof(complex),  // Pitch (row width in bytes)
        srcData.size.width,                     // Width in elements
        srcData.size.height                     // Height in elements
    );
    copyParams.srcPos = make_cudaPos(0, 0, 0); // Start from origin
    
    // Destination parameters
    copyParams.dstPtr = make_cudaPitchedPtr(
        destData.data,                          // Destination pointer
        destData.size.width * sizeof(complex), // Pitch (row width in bytes)
        destData.size.width,                    // Width in elements
        destData.size.height                    // Height in elements
    );
    copyParams.dstPos = make_cudaPos(0, 0, 0); // Start from origin
    
    // Copy extent (how much to copy)
    copyParams.extent = make_cudaExtent(
        srcData.size.width * sizeof(complex),  // Width in bytes
        srcData.size.height,                    // Height in elements
        srcData.size.depth                      // Depth in elements
    );
    
    // Determine copy direction
    bool srcIsDevice = isOnDevice(srcData.data);
    bool dstIsDevice = isOnDevice(destData.data);
    
    if (srcIsDevice && dstIsDevice) {
        copyParams.kind = cudaMemcpyDeviceToDevice;
    } else if (!srcIsDevice && dstIsDevice) {
        copyParams.kind = cudaMemcpyHostToDevice;
    } else if (srcIsDevice && !dstIsDevice) {
        copyParams.kind = cudaMemcpyDeviceToHost;
    } else {
        copyParams.kind = cudaMemcpyHostToHost;
    }
    
    // Execute the copy
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
    destData.backend = this;

}

ComplexData CUDABackendMemoryManager::allocateMemoryOnDevice(const RectangleShape& shape) const {
    ComplexData result{this, nullptr, shape};
    CUDA_CHECK(cudaMalloc((void**)&result.data, shape.volume * sizeof(complex)));
    return result;
}

ComplexData CUDABackendMemoryManager::copyDataToDevice(const ComplexData& srcdata) const {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromHostToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                               destdata.data, srcdata.data);
    }
    destdata.backend = this;

    return destdata;
}

ComplexData CUDABackendMemoryManager::moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const {
    complex* temp;
    if (&destBackend == this){
        return srcdata;
    }
    else{

        temp = (complex*) fftw_malloc(sizeof(complex) * srcdata.size.volume);

        if (srcdata.data != nullptr){
            CUBE_UTL_COPY::copyDataFromDeviceToHost(srcdata.size.width, srcdata.size.height, srcdata.size.depth, temp, srcdata.data);
        }
    }

    ComplexData destdata{&destBackend, temp, srcdata.size};
    return destBackend.copyDataToDevice(destdata);
}

ComplexData CUDABackendMemoryManager::copyData(const ComplexData& srcdata) const {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                                 destdata.data, srcdata.data);
    }
    return destdata;
}

void CUDABackendMemoryManager::freeMemoryOnDevice(ComplexData& srcdata) const {
    assert((srcdata.data != nullptr) + "trying to free gpu memory that is a nullptr");
    CUDA_CHECK(cudaFree(srcdata.data));
    srcdata.data = nullptr;
}

size_t CUDABackendMemoryManager::getAvailableMemory() const {
    // For CUDA backend, return available GPU memory
    std::unique_lock<std::mutex> lock(backendMutex);
    
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    
    return freeMem;

}







CUDADeconvolutionBackend::CUDADeconvolutionBackend() {
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
}

CUDADeconvolutionBackend::~CUDADeconvolutionBackend() {
    destroyFFTPlans();
}


void CUDADeconvolutionBackend::init(const RectangleShape& shape){
    try {
        initializeFFTPlans(shape);
        
        std::cout << "[STATUS] CUDA backend initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CUDA preprocessing: " << e.what() << std::endl;
    }
}


void CUDADeconvolutionBackend::cleanup(){
    try {
        // Clean up CUDA resources if needed
        destroyFFTPlans();
        std::cout << "[STATUS] CUDA backend postprocessing completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CUDA postprocessing: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::initializeFFTPlans(const RectangleShape& cube){
    std::unique_lock<std::mutex> lock(backendMutex);
    if (plansInitialized_) return;
    
    try {
        // Allocate temporary memory for plan creation
        size_t tempSize = sizeof(complex) * cube.volume;
        // Create forward FFT plan
        CUFFT_CHECK(cufftCreate(&this->forwardPlan));
        CUFFT_CHECK(cufftMakePlan3d(this->forwardPlan, cube.depth, cube.height, cube.width, CUFFT_Z2Z, &tempSize));

        
        // Create backward FFT plan
        CUFFT_CHECK(cufftCreate(&this->backwardPlan));
        CUFFT_CHECK(cufftMakePlan3d(this->backwardPlan, cube.depth, cube.height, cube.width, CUFFT_Z2Z, &tempSize));

        plansInitialized_ = true;
     } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in FFT plan initialization: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::destroyFFTPlans(){
    std::unique_lock<std::mutex> lock(backendMutex);

    if (plansInitialized_) {
        if (forwardPlan) {
            CUFFT_CHECK(cufftDestroy(forwardPlan));
            forwardPlan = CUFFT_PLAN_NULL;
        }
        if (backwardPlan) {
            CUFFT_CHECK(cufftDestroy(backwardPlan));
            backwardPlan = CUFFT_PLAN_NULL;
        }
        plansInitialized_ = false;
    }
}

// FFT Operations
void CUDADeconvolutionBackend::forwardFFT(const ComplexData& in, ComplexData& out) const {
    CUFFT_CHECK(cufftExecZ2Z(this->forwardPlan, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD));

}

void CUDADeconvolutionBackend::backwardFFT(const ComplexData& in, ComplexData& out) const {
    CUFFT_CHECK(cufftExecZ2Z(this->backwardPlan, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_BACKWARD));

}

// Shift Operations
void CUDADeconvolutionBackend::octantFourierShift(ComplexData& data) const {
    try {
        CUBE_FTT::octantFourierShiftFftwComplex(data.size.width, data.size.height, data.size.depth, data.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in octantFourierShift: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::inverseQuadrantShift(ComplexData& data) const {
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

// void CUDADeconvolutionBackend::quadrantShiftMat(cv::Mat& magI) {
//     try {
//         int cx = magI.cols / 2;
//         int cy = magI.rows / 2;

//         cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
//         cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
//         cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
//         cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

//         cv::Mat tmp;
//         q0.copyTo(tmp);
//         q3.copyTo(q0);
//         tmp.copyTo(q3);

//         q1.copyTo(tmp);
//         q2.copyTo(q1);
//         tmp.copyTo(q2);
//     } catch (const std::exception& e) {
//         std::cerr << "[ERROR] Exception in quadrantShiftMat: " << e.what() << std::endl;
//     }
// }

// Complex Arithmetic Operations
void CUDADeconvolutionBackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    try {
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in complexMultiplication" << std::endl;
            return;
        }
        CUBE_MAT::complexElementwiseMatMulFftwComplex(a.size.volume, 1, 1, a.data, b.data, result.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplication: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
    try {
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in complexDivision" << std::endl;
            return;
        }
        CUBE_MAT::complexElementwiseMatDivFftwComplex(a.size.volume, 1, 1, a.data, b.data, result.data, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivision: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    try {
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in complexAddition" << std::endl;
            return;
        }
        for (int i = 0; i < a.size.volume; ++i) {
            result.data[i][0] = a.data[i][0] + b.data[i][0];
            result.data[i][1] = a.data[i][1] + b.data[i][1];
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexAddition: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) const {
    try {
        if (a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in scalarMultiplication" << std::endl;
            return;
        }
        for (int i = 0; i < a.size.volume; ++i) {
            result.data[i][0] = a.data[i][0] * scalar;
            result.data[i][1] = a.data[i][1] * scalar;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in scalarMultiplication: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
    try {
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in complexMultiplicationWithConjugate" << std::endl;
            return;
        }
        CUBE_MAT::complexElementwiseMatMulConjugateFftwComplex(a.size.volume, 1, 1, a.data, b.data, result.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplicationWithConjugate: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
    try {
        if (a.size.volume != b.size.volume || a.size.volume != result.size.volume) {
            std::cerr << "[ERROR] Size mismatch in complexDivisionStabilized" << std::endl;
            return;
        }
        CUBE_MAT::complexElementwiseMatDivStabilizedFftwComplex(a.size.volume, 1, 1, a.data, b.data, result.data, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivisionStabilized: " << e.what() << std::endl;
    }
}





// Specialized Functions
void CUDADeconvolutionBackend::hasNAN(const ComplexData& data) const {
    try {
        // Implementation would go here
        std::cout << "[DEBUG] hasNAN called on CUDA backend" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in hasNAN: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
    try {
        CUBE_REG::calculateLaplacianFftwComplex(psf.size.width, psf.size.height, psf.size.depth, psf.data, laplacian.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in calculateLaplacianOfPSF: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::normalizeImage(ComplexData& resultImage, double epsilon) const {
    try {
        CUBE_FTT::normalizeFftwComplexData(1, 1, 1, resultImage.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeImage: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::rescaledInverse(ComplexData& data, double cubeVolume) const {
    try {
        for (int i = 0; i < data.size.volume; ++i) {
            data.data[i][0] /= cubeVolume;
            data.data[i][1] /= cubeVolume;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in rescaledInverse: " << e.what() << std::endl;
    }
}



// // Layer and Visualization Functions
// void CUDADeconvolutionBackend::reorderLayers(ComplexData& data) {
//     try {
//         int width = data.size.width;
//         int height = data.size.height;
//         int depth = data.size.depth;
//         int layerSize = width * height;
//         int halfDepth = depth / 2;
        
//         complex* temp = (complex*) fftw_malloc(sizeof(complex) * data.size.volume);

//         int destIndex = 0;

//         // Copy the middle layer to the first position
//         std::memcpy(temp + destIndex * layerSize, data.data + halfDepth * layerSize, sizeof(complex) * layerSize);
//         destIndex++;

//         // Copy the layers after the middle layer
//         for (int z = halfDepth + 1; z < depth; ++z) {
//             std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex) * layerSize);
//             destIndex++;
//         }

//         // Copy the layers before the middle layer
//         for (int z = 0; z < halfDepth; ++z) {
//             std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(complex) * layerSize);
//             destIndex++;
//         }

//         // Copy reordered data back to the original array
//         std::memcpy(data.data, temp, sizeof(complex) * data.size.volume);
//         fftw_free(temp);
//     } catch (const std::exception& e) {
//         std::cerr << "[ERROR] Exception in reorderLayers: " << e.what() << std::endl;
//     }
// }

// void CUDABackend::visualizeFFT(const ComplexData& data) {
//     try {
//         int width = data.size.width;
//         int height = data.size.height;
//         int depth = data.size.depth;
        
//         Image3D i;
//         std::vector<cv::Mat> output;
//         for (int z = 0; z < depth; ++z) {
//             cv::Mat result(height, width, CV_32F);
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     int index = z * height * width + y * width + x;
//                     result.at<float>(y, x) = data.data[index][0];
//                 }
//             }
//             output.push_back(result);
//         }
//         i.slices = output;
//         i.show();
//     } catch (const std::exception& e) {
//         std::cerr << "[ERROR] Exception in visualizeFFT: " << e.what() << std::endl;
//     }
// }

// Gradient and TV Functions
void CUDADeconvolutionBackend::gradientX(const ComplexData& image, ComplexData& gradX) const {
    try {
        CUBE_REG::gradXFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradX.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientX: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::gradientY(const ComplexData& image, ComplexData& gradY) const {
    try {
        CUBE_REG::gradYFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradY.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientY: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::gradientZ(const ComplexData& image, ComplexData& gradZ) const {
    try {
        CUBE_REG::gradZFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradZ.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientZ: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
    try {
        CUBE_REG::computeTVFftwComplex(gx.size.width, gx.size.height, gx.size.depth, lambda, gx.data, gy.data, gz.data, tv.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in computeTV: " << e.what() << std::endl;
    }
}

void CUDADeconvolutionBackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const {
    try {
        CUBE_REG::normalizeTVFftwComplex(gradX.size.width, gradX.size.height, gradX.size.depth, gradX.data, gradY.data, gradZ.data, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeTV: " << e.what() << std::endl;
    }
}


