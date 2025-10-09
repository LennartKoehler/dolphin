#include "CUDABackend.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>

#include <cassert>



CUDABackend::CUDABackend() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[ERROR] CUDA not available, falling back to CPU operations" << std::endl;
    }
}

CUDABackend::~CUDABackend() {
    destroyFFTPlans();
}


void CUDABackend::init(const RectangleShape& shape) {
    try {
        initializeFFTPlans(shape);
        
        // Set OpenMP to single thread for CUDA operations
        omp_set_num_threads(1);
        
        std::cout << "[STATUS] CUDA backend initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CUDA preprocessing: " << e.what() << std::endl;
    }
}


void CUDABackend::cleanup() {
    try {
        // Clean up CUDA resources if needed
        destroyFFTPlans();
        std::cout << "[STATUS] CUDA backend postprocessing completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in CUDA postprocessing: " << e.what() << std::endl;
    }
}

bool CUDABackend::isOnDevice(void* ptr) {
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
        std::cerr << "[ERROR] CUDA error checking pointer: " << cudaGetErrorString(result) << std::endl;
        return false;
    }
}

void CUDABackend::allocateMemoryOnDevice(ComplexData& data) {
    if (data.data != nullptr && isOnDevice(data.data)) {
        return; // Already on device
    }
    
    // Allocate CUDA memory
    cudaError_t result = cudaMalloc((void**)&data.data, data.size.volume * sizeof(complex));
    if (result != cudaSuccess) {
        std::cerr << "[ERROR] CUDA malloc failed: " << cudaGetErrorString(result) << std::endl;
        data.data = nullptr;
    }
    data.device = DeviceID::CUDA;

}

void CUDABackend::memCopy(const ComplexData& srcData, ComplexData& destData) {
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
    cudaError_t result = cudaMemcpy3D(&copyParams);
    if (result != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemcpy3D failed: " << cudaGetErrorString(result) << std::endl;
    }
    destData.device = DeviceID::CUDA;

}

ComplexData CUDABackend::allocateMemoryOnDevice(const RectangleShape& shape) {
    ComplexData result;
    result.size = shape;
    result.data = nullptr;
    
    cudaError_t cudaResult = cudaMalloc((void**)&result.data, shape.volume * sizeof(complex));
    if (cudaResult != cudaSuccess) {
        std::cerr << "[ERROR] CUDA malloc failed: " << cudaGetErrorString(cudaResult) << std::endl;
        result.data = nullptr;
    }
    result.device = DeviceID::CUDA;
   
    return result;
}

ComplexData CUDABackend::moveDataToDevice(const ComplexData& srcdata) {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromHostToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth, 
                                               destdata.data, srcdata.data);
    }
    destdata.device = DeviceID::CUDA;

    return destdata;
}

ComplexData CUDABackend::moveDataFromDevice(const ComplexData& srcdata){
    complex* temp = (complex *) fftw_malloc(sizeof(complex) * srcdata.size.volume);
    ComplexData destdata{temp, srcdata.size};

    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToHost(srcdata.size.width, srcdata.size.height, srcdata.size.depth, 
                                               destdata.data, srcdata.data);
    }
    destdata.device = DeviceID::CUDA;

    return destdata;
}

ComplexData CUDABackend::copyData(const ComplexData& srcdata) {
    ComplexData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                                 destdata.data, srcdata.data);
    }
    destdata.device = DeviceID::CUDA;

    return destdata;
}

void CUDABackend::freeMemoryOnDevice(ComplexData& srcdata){
    assert((srcdata.data != nullptr) + "trying to free gpu memory that is a nullptr");
    cudaFree(srcdata.data);
}

void CUDABackend::initializeFFTPlans(const RectangleShape& cube) {
    std::unique_lock<std::mutex> lock(backendMutex);
    if (plansInitialized_) return;
    
    try {
        // Allocate temporary memory for plan creation
        size_t tempSize = sizeof(complex) * cube.volume;
        // Create forward FFT plan
        cufftCreate(&this->forwardPlan);
        cufftMakePlan3d(this->forwardPlan, cube.depth, cube.height, cube.width, CUFFT_Z2Z, &tempSize);

        
        // Create backward FFT plan
        cufftCreate(&this->backwardPlan);
        cufftMakePlan3d(this->backwardPlan, cube.depth, cube.height, cube.width, CUFFT_Z2Z, &tempSize);

     } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in FFT plan initialization: " << e.what() << std::endl;
    }
}

void CUDABackend::destroyFFTPlans() {
    std::unique_lock<std::mutex> lock(backendMutex);

    if (plansInitialized_) {
        if (forwardPlan) {
            cufftDestroy(forwardPlan);
            forwardPlan = CUFFT_PLAN_NULL;
        }
        if (backwardPlan) {
            cufftDestroy(backwardPlan);
            backwardPlan = CUFFT_PLAN_NULL;
        }
        plansInitialized_ = false;
    }
}

// FFT Operations
void CUDABackend::forwardFFT(const ComplexData& in, ComplexData& out) {
    cufftExecZ2Z(this->forwardPlan, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD);

}

void CUDABackend::backwardFFT(const ComplexData& in, ComplexData& out) {
    cufftExecZ2Z(this->backwardPlan, reinterpret_cast<cufftDoubleComplex*>(in.data), reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_BACKWARD);

}

// Shift Operations
void CUDABackend::octantFourierShift(ComplexData& data) {
    try {
        CUBE_FTT::octantFourierShiftFftwComplex(data.size.width, data.size.height, data.size.depth, data.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in octantFourierShift: " << e.what() << std::endl;
    }
}

void CUDABackend::inverseQuadrantShift(ComplexData& data) {
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

void CUDABackend::quadrantShiftMat(cv::Mat& magI) {
    try {
        int cx = magI.cols / 2;
        int cy = magI.rows / 2;

        cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
        cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
        cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        cv::Mat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in quadrantShiftMat: " << e.what() << std::endl;
    }
}

// Complex Arithmetic Operations
void CUDABackend::complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) {
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

void CUDABackend::complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
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

void CUDABackend::complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) {
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

void CUDABackend::scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) {
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

void CUDABackend::complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) {
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

void CUDABackend::complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
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
void CUDABackend::calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) {
    try {
        CUBE_REG::calculateLaplacianFftwComplex(psf.size.width, psf.size.height, psf.size.depth, psf.data, laplacian.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in calculateLaplacianOfPSF: " << e.what() << std::endl;
    }
}

void CUDABackend::normalizeImage(ComplexData& resultImage, double epsilon) {
    try {
        CUBE_FTT::normalizeFftwComplexData(1, 1, 1, resultImage.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeImage: " << e.what() << std::endl;
    }
}

void CUDABackend::rescaledInverse(ComplexData& data, double cubeVolume) {
    try {
        for (int i = 0; i < data.size.volume; ++i) {
            data.data[i][0] /= cubeVolume;
            data.data[i][1] /= cubeVolume;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in rescaledInverse: " << e.what() << std::endl;
    }
}



// Layer and Visualization Functions
void CUDABackend::reorderLayers(ComplexData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        int layerSize = width * height;
        int halfDepth = depth / 2;
        
        complex* temp = (complex*) fftw_malloc(sizeof(complex) * data.size.volume);

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
void CUDABackend::gradientX(const ComplexData& image, ComplexData& gradX) {
    try {
        CUBE_REG::gradXFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradX.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientX: " << e.what() << std::endl;
    }
}

void CUDABackend::gradientY(const ComplexData& image, ComplexData& gradY) {
    try {
        CUBE_REG::gradYFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradY.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientY: " << e.what() << std::endl;
    }
}

void CUDABackend::gradientZ(const ComplexData& image, ComplexData& gradZ) {
    try {
        CUBE_REG::gradZFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradZ.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientZ: " << e.what() << std::endl;
    }
}

void CUDABackend::computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) {
    try {
        CUBE_REG::computeTVFftwComplex(gx.size.width, gx.size.height, gx.size.depth, lambda, gx.data, gy.data, gz.data, tv.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in computeTV: " << e.what() << std::endl;
    }
}

void CUDABackend::normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) {
    try {
        CUBE_REG::normalizeTVFftwComplex(gradX.size.width, gradX.size.height, gradX.size.depth, gradX.data, gradY.data, gradZ.data, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeTV: " << e.what() << std::endl;
    }
}


size_t CUDABackend::getAvailableMemory() {
    // For CUDA backend, return available GPU memory
    std::unique_lock<std::mutex> lock(backendMutex);
    cudaError_t cudaStatus;
    
    size_t freeMem, totalMem;
    cudaStatus = cudaMemGetInfo(&freeMem, &totalMem);
    
    return freeMem;

}
extern "C" IDeconvolutionBackend* create_backend() {
    return new CUDABackend();
}