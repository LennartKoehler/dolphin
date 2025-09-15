#include "deconvolution/backend/GPUBackend.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "Image3D.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cassert>

GPUBackend::GPUBackend() {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "[ERROR] CUDA not available, falling back to CPU operations" << std::endl;
    }
}

GPUBackend::~GPUBackend() {
    destroyFFTPlans();
}

void GPUBackend::preprocess() {
    try {
        // Initialize FFT plans if not already done
        if (!plansInitialized) {
            // Plans will be initialized when needed based on image dimensions
            plansInitialized = true;
        }
        
        // Set OpenMP to single thread for GPU operations
        omp_set_num_threads(1);
        
        std::cout << "[STATUS] GPU backend initialized" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in GPU preprocessing: " << e.what() << std::endl;
    }
}

void GPUBackend::postprocess() {
    try {
        // Clean up GPU resources if needed
        destroyFFTPlans();
        std::cout << "[STATUS] GPU backend postprocessing completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in GPU postprocessing: " << e.what() << std::endl;
    }
}

std::unordered_map<PSFIndex, FFTWData>& GPUBackend::movePSFstoGPU(std::unordered_map<PSFIndex, FFTWData>& psfMap) {
    try {
        for (auto& it : psfMap) {
            FFTWData& psfData = it.second;
            
            if (!isOnDevice(psfData.data)) {
                fftw_complex* d_temp_h;
                cudaError_t cudaStatus = cudaMalloc((void**)&d_temp_h, psfData.size.volume * sizeof(fftw_complex));
                if (cudaStatus != cudaSuccess) {
                    std::cerr << "[ERROR] CUDA malloc failed for PSF" << std::endl;
                    continue;
                }
                
                CUBE_UTL_COPY::copyDataFromHostToDevice(psfData.size.width, psfData.size.height, psfData.size.depth, d_temp_h, psfData.data);
                psfData.data = d_temp_h;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in movePSFstoGPU: " << e.what() << std::endl;
    }
    return psfMap;
}

bool GPUBackend::isOnDevice(void* ptr) {
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

void GPUBackend::allocateMemoryOnDevice(FFTWData& data) {
    if (data.data != nullptr && isOnDevice(data.data)) {
        return; // Already on device
    }
    
    // Allocate GPU memory
    cudaError_t result = cudaMalloc((void**)&data.data, data.size.volume * sizeof(fftw_complex));
    if (result != cudaSuccess) {
        std::cerr << "[ERROR] CUDA malloc failed: " << cudaGetErrorString(result) << std::endl;
        data.data = nullptr;
    }
}

FFTWData GPUBackend::allocateMemoryOnDevice(const RectangleShape& shape) {
    FFTWData result;
    result.size = shape;
    result.data = nullptr;
    
    cudaError_t cudaResult = cudaMalloc((void**)&result.data, shape.volume * sizeof(fftw_complex));
    if (cudaResult != cudaSuccess) {
        std::cerr << "[ERROR] CUDA malloc failed: " << cudaGetErrorString(cudaResult) << std::endl;
        result.data = nullptr;
    }
    
    return result;
}

FFTWData GPUBackend::moveDataToDevice(const FFTWData& srcdata) {
    FFTWData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromHostToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth, 
                                               destdata.data, srcdata.data);
    }
    return destdata;
}

FFTWData GPUBackend::moveDataFromDevice(const FFTWData& srcdata){
    fftw_complex* temp = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * srcdata.size.volume);
    FFTWData destdata{temp, srcdata.size};
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToHost(srcdata.size.width, srcdata.size.height, srcdata.size.depth, 
                                               destdata.data, srcdata.data);
    }
    return destdata;
}

FFTWData GPUBackend::copyData(const FFTWData& srcdata) {
    FFTWData destdata = allocateMemoryOnDevice(srcdata.size);
    if (srcdata.data != nullptr) {
        CUBE_UTL_COPY::copyDataFromDeviceToDevice(srcdata.size.width, srcdata.size.height, srcdata.size.depth,
                                                 destdata.data, srcdata.data);
    }
    return destdata;
}

void GPUBackend::freeMemoryOnDevice(FFTWData& srcdata){
    assert((srcdata.data != nullptr) + "trying to free gpu memory that is a nullptr");
    cudaFree(srcdata.data);
}

void GPUBackend::initializeFFTPlans(const RectangleShape& cube) {
    if (plansInitialized) return;
    
    try {
        // Create forward FFT plan
        this->forwardPlan = fftw_plan_dft_3d(cube.depth, cube.height, cube.width, 
                                            this->planMemory, this->planMemory, FFTW_FORWARD, FFTW_MEASURE);
        
        // Create backward FFT plan
        this->backwardPlan = fftw_plan_dft_3d(cube.depth, cube.height, cube.width,
                                             this->planMemory, this->planMemory, FFTW_BACKWARD, FFTW_MEASURE);
        
        plansInitialized = true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in FFT plan initialization: " << e.what() << std::endl;
    }
}

void GPUBackend::destroyFFTPlans() {
    if (plansInitialized) {
        if (forwardPlan) {
            fftw_destroy_plan(forwardPlan);
            forwardPlan = nullptr;
        }
        if (backwardPlan) {
            fftw_destroy_plan(backwardPlan);
            backwardPlan = nullptr;
        }
        plansInitialized = false;
    }
}

// FFT Operations
void GPUBackend::forwardFFT(const FFTWData& in, FFTWData& out) {
    fftw_execute_dft(this->forwardPlan, in.data, out.data);
}

void GPUBackend::backwardFFT(const FFTWData& in, FFTWData& out) {
    fftw_execute_dft(this->backwardPlan, in.data, out.data);
}

// Shift Operations
void GPUBackend::octantFourierShift(FFTWData& data) {
    try {
        CUBE_FTT::octantFourierShiftFftwComplex(data.size.width, data.size.height, data.size.depth, data.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in octantFourierShift: " << e.what() << std::endl;
    }
}

void GPUBackend::inverseQuadrantShift(FFTWData& data) {
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

void GPUBackend::quadrantShiftMat(cv::Mat& magI) {
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
void GPUBackend::complexMultiplication(const FFTWData& a, const FFTWData& b, FFTWData& result) {
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

void GPUBackend::complexDivision(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) {
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

void GPUBackend::complexAddition(const FFTWData& a, const FFTWData& b, FFTWData& result) {
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

void GPUBackend::scalarMultiplication(const FFTWData& a, double scalar, FFTWData& result) {
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

void GPUBackend::complexMultiplicationWithConjugate(const FFTWData& a, const FFTWData& b, FFTWData& result) {
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

void GPUBackend::complexDivisionStabilized(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) {
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

// Conversion Functions
void GPUBackend::convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, FFTWData& output) {
    try {
        int width = output.size.width;
        int height = output.size.height;
        int depth = output.size.depth;
        
        for (int z = 0; z < depth; ++z) {
            CV_Assert(input[z].type() == CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    output.data[z * height * width + y * width + x][0] = static_cast<double>(input[z].at<float>(y, x));
                    output.data[z * height * width + y * width + x][1] = 0.0;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertCVMatVectorToFFTWComplex: " << e.what() << std::endl;
    }
}

void GPUBackend::convertFFTWComplexToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) {
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    double real_part = input.data[index][0];
                    double imag_part = input.data[index][1];
                    result.at<float>(y, x) = static_cast<float>(sqrt(real_part * real_part + imag_part * imag_part));
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexToCVMatVector: " << e.what() << std::endl;
    }
}

void GPUBackend::convertFFTWComplexRealToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) {
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input.data[index][0];
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexRealToCVMatVector: " << e.what() << std::endl;
    }
}

void GPUBackend::convertFFTWComplexImgToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) {
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input.data[index][1];
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexImgToCVMatVector: " << e.what() << std::endl;
    }
}

// PSF Operations
void GPUBackend::padPSF(const FFTWData& psf, FFTWData& padded_psf, const RectangleShape& target_size) {
    try {
        // Create temporary copy for shifting
        FFTWData temp_psf = copyData(psf);
        octantFourierShift(temp_psf);
        
        // Zero out padded PSF
        for (int i = 0; i < padded_psf.size.volume; ++i) {
            padded_psf.data[i][0] = 0.0;
            padded_psf.data[i][1] = 0.0;
        }

        if (psf.size.depth > target_size.depth) {
            std::cerr << "[ERROR] PSF has more layers than target size" << std::endl;
        }

        int x_offset = (target_size.width - psf.size.width) / 2;
        int y_offset = (target_size.height - psf.size.height) / 2;
        int z_offset = (target_size.depth - psf.size.depth) / 2;

        for (int z = 0; z < psf.size.depth; ++z) {
            for (int y = 0; y < psf.size.height; ++y) {
                for (int x = 0; x < psf.size.width; ++x) {
                    int padded_index = ((z + z_offset) * target_size.height + (y + y_offset)) * target_size.width + (x + x_offset);
                    int psf_index = (z * psf.size.height + y) * psf.size.width + x;

                    padded_psf.data[padded_index][0] = temp_psf.data[psf_index][0];
                    padded_psf.data[padded_index][1] = temp_psf.data[psf_index][1];
                }
            }
        }
        octantFourierShift(padded_psf);
        
        // Clean up temporary data
        cudaFree(temp_psf.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in padPSF: " << e.what() << std::endl;
    }
}

// Specialized Functions
void GPUBackend::calculateLaplacianOfPSF(const FFTWData& psf, FFTWData& laplacian) {
    try {
        CUBE_REG::calculateLaplacianFftwComplex(psf.size.width, psf.size.height, psf.size.depth, psf.data, laplacian.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in calculateLaplacianOfPSF: " << e.what() << std::endl;
    }
}

void GPUBackend::normalizeImage(FFTWData& resultImage, double epsilon) {
    try {
        CUBE_FTT::normalizeFftwComplexData(1, 1, 1, resultImage.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeImage: " << e.what() << std::endl;
    }
}

void GPUBackend::rescaledInverse(FFTWData& data, double cubeVolume) {
    try {
        for (int i = 0; i < data.size.volume; ++i) {
            data.data[i][0] /= cubeVolume;
            data.data[i][1] /= cubeVolume;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in rescaledInverse: " << e.what() << std::endl;
    }
}

void GPUBackend::saveInterimImages(const FFTWData& resultImage, int gridNum, int channel_z, int i) {
    try {
        std::vector<cv::Mat> debugImage;
        convertFFTWComplexToCVMatVector(resultImage, debugImage);
        for (int k = 0; k < debugImage.size(); k++) {
            cv::normalize(debugImage[k], debugImage[k], 0, 255, cv::NORM_MINMAX);
            cv::imwrite(
                "../result/debug/debug_image_" + std::to_string(channel_z) + "_" + std::to_string(gridNum) + "_iter_" +
                std::to_string(i) + "_slice_" + std::to_string(k) + ".png", debugImage[k]);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in saveInterimImages: " << e.what() << std::endl;
    }
}

// Layer and Visualization Functions
void GPUBackend::reorderLayers(FFTWData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        int layerSize = width * height;
        int halfDepth = depth / 2;
        
        fftw_complex* temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * data.size.volume);

        int destIndex = 0;

        // Copy the middle layer to the first position
        std::memcpy(temp + destIndex * layerSize, data.data + halfDepth * layerSize, sizeof(fftw_complex) * layerSize);
        destIndex++;

        // Copy the layers after the middle layer
        for (int z = halfDepth + 1; z < depth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(fftw_complex) * layerSize);
            destIndex++;
        }

        // Copy the layers before the middle layer
        for (int z = 0; z < halfDepth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data.data + z * layerSize, sizeof(fftw_complex) * layerSize);
            destIndex++;
        }

        // Copy reordered data back to the original array
        std::memcpy(data.data, temp, sizeof(fftw_complex) * data.size.volume);
        fftw_free(temp);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in reorderLayers: " << e.what() << std::endl;
    }
}

void GPUBackend::visualizeFFT(const FFTWData& data) {
    try {
        int width = data.size.width;
        int height = data.size.height;
        int depth = data.size.depth;
        
        Image3D i;
        std::vector<cv::Mat> output;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = data.data[index][0];
                }
            }
            output.push_back(result);
        }
        i.slices = output;
        i.show();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in visualizeFFT: " << e.what() << std::endl;
    }
}

// Gradient and TV Functions
void GPUBackend::gradientX(const FFTWData& image, FFTWData& gradX) {
    try {
        CUBE_REG::gradXFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradX.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientX: " << e.what() << std::endl;
    }
}

void GPUBackend::gradientY(const FFTWData& image, FFTWData& gradY) {
    try {
        CUBE_REG::gradYFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradY.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientY: " << e.what() << std::endl;
    }
}

void GPUBackend::gradientZ(const FFTWData& image, FFTWData& gradZ) {
    try {
        CUBE_REG::gradZFftwComplex(image.size.width, image.size.height, image.size.depth, image.data, gradZ.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientZ: " << e.what() << std::endl;
    }
}

void GPUBackend::computeTV(double lambda, const FFTWData& gx, const FFTWData& gy, const FFTWData& gz, FFTWData& tv) {
    try {
        CUBE_REG::computeTVFftwComplex(gx.size.width, gx.size.height, gx.size.depth, lambda, gx.data, gy.data, gz.data, tv.data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in computeTV: " << e.what() << std::endl;
    }
}

void GPUBackend::normalizeTV(FFTWData& gradX, FFTWData& gradY, FFTWData& gradZ, double epsilon) {
    try {
        CUBE_REG::normalizeTVFftwComplex(gradX.size.width, gradX.size.height, gradX.size.depth, gradX.data, gradY.data, gradZ.data, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeTV: " << e.what() << std::endl;
    }
}