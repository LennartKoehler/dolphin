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

GPUBackend::GPUBackend() : plansInitialized(false) {
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

std::unordered_map<PSFIndex, PSFfftw*>& GPUBackend::movePSFstoGPU(std::unordered_map<PSFIndex, PSFfftw*>& psfMap) {
    try {
        for (auto& it : psfMap) {
            fftw_complex* d_temp_h;
            // psf same size as cube
            cudaError_t cudaStatus = cudaMalloc((void**)&d_temp_h, cubeMetaData.cubeVolume * sizeof(PSFfftw));
            if (cudaStatus != cudaSuccess) {
                std::cerr << "[ERROR] CUDA malloc failed for PSF" << std::endl;
                continue;
            }
            
            CUBE_UTL_COPY::copyDataFromHostToDevice(cubeMetaData.cubeWidth, cubeMetaData.cubeHeight, cubeMetaData.cubeDepth, d_temp_h, it.second);
            it.second = d_temp_h;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in GPU preprocessing: " << e.what() << std::endl;
    }
    return psfMap;
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

void GPUBackend::initializeFFTPlans(int width, int height, int depth) {
    if (plansInitialized) return;
    
    try {
        // Create forward FFT plan
        cufftResult result = cufftPlan3d(&forwardPlan, depth, height, width, CUFFT_FORWARD);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[ERROR] Failed to create forward FFT plan" << std::endl;
            return;
        }
        
        // Create backward FFT plan
        result = cufftPlan3d(&backwardPlan, depth, height, width, CUFFT_INVERSE);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[ERROR] Failed to create backward FFT plan" << std::endl;
            cufftDestroy(forwardPlan);
            return;
        }
        
        plansInitialized = true;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in FFT plan initialization: " << e.what() << std::endl;
    }
}

void GPUBackend::destroyFFTPlans() {
    if (plansInitialized) {
        if (forwardPlan) {
            cufftDestroy(forwardPlan);
            forwardPlan = nullptr;
        }
        if (backwardPlan) {
            cufftDestroy(backwardPlan);
            backwardPlan = nullptr;
        }
        plansInitialized = false;
    }
}



// FFT Operations
void GPUBackend::forwardFFT(fftw_complex* in, fftw_complex* out, int imageDepth, int imageHeight, int imageWidth) {
    try {
        // Initialize FFT plans if not already done
        initializeFFTPlans(imageWidth, imageHeight, imageDepth);
        
        // Convert FFTW complex to CUFFT complex on device
        cufftComplex* d_in;
        cudaMalloc((void**)&d_in, imageDepth * imageHeight * imageWidth * sizeof(cufftComplex));
        CUBE_UTL_CONVERT::convertFftwToCufftComplexOnDevice(imageWidth, imageHeight, imageDepth, in, d_in);
        
        // Allocate output on device
        cufftComplex* d_out;
        cudaMalloc((void**)&d_out, imageDepth * imageHeight * imageWidth * sizeof(cufftComplex));
        
        // Execute forward FFT
        cufftResult result = cufftExecForward(forwardPlan, d_in, d_out);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[ERROR] Forward FFT execution failed" << std::endl;
            cudaFree(d_in);
            cudaFree(d_out);
            return;
        }
        
        // Copy result back to host
        CUBE_UTL_CONVERT::convertCufftToFftwComplexOnHost(imageWidth, imageHeight, imageDepth, out, d_out);
        
        // Apply octant shift
        octantFourierShift(out, imageWidth, imageHeight, imageDepth);
        
        // Clean up
        cudaFree(d_in);
        cudaFree(d_out);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in forwardFFT: " << e.what() << std::endl;
    }
}

void GPUBackend::backwardFFT(fftw_complex* in, fftw_complex* out, int imageDepth, int imageHeight, int imageWidth) {
    try {
        // Initialize FFT plans if not already done
        initializeFFTPlans(imageWidth, imageHeight, imageDepth);
        
        // Apply inverse octant shift
        inverseQuadrantShift(in, imageWidth, imageHeight, imageDepth);
        
        // Convert FFTW complex to CUFFT complex on device
        cufftComplex* d_in;
        cudaMalloc((void**)&d_in, imageDepth * imageHeight * imageWidth * sizeof(cufftComplex));
        CUBE_UTL_CONVERT::convertFftwToCufftComplexOnDevice(imageWidth, imageHeight, imageDepth, in, d_in);
        
        // Allocate output on device
        cufftComplex* d_out;
        cudaMalloc((void**)&d_out, imageDepth * imageHeight * imageWidth * sizeof(cufftComplex));
        
        // Execute backward FFT
        cufftResult result = cufftExecInverse(backwardPlan, d_in, d_out);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[ERROR] Backward FFT execution failed" << std::endl;
            cudaFree(d_in);
            cudaFree(d_out);
            return;
        }
        
        // Copy result back to host
        CUBE_UTL_CONVERT::convertCufftToFftwComplexOnHost(imageWidth, imageHeight, imageDepth, out, d_out);
        
        // Clean up
        cudaFree(d_in);
        cudaFree(d_out);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in backwardFFT: " << e.what() << std::endl;
    }
}

// Shift Operations
void GPUBackend::octantFourierShift(fftw_complex* data, int width, int height, int depth) {
    try {
        CUBE_FTT::octantFourierShiftFftwComplex(width, height, depth, data);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in octantFourierShift: " << e.what() << std::endl;
    }
}

void GPUBackend::inverseQuadrantShift(fftw_complex* data, int width, int height, int depth) {
    try {
        // For GPU implementation, we can use the same approach as CPU
        // since this is a relatively small operation
        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int halfDepth = depth / 2;

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < halfHeight; ++y) {
                for (int x = 0; x < halfWidth; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x + halfWidth);

                    std::swap(data[idx1][0], data[idx2][0]);
                    std::swap(data[idx1][1], data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < halfHeight; ++y) {
                for (int x = halfWidth; x < width; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y + halfHeight) * width + (x - halfWidth);

                    std::swap(data[idx1][0], data[idx2][0]);
                    std::swap(data[idx1][1], data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = halfHeight; y < height; ++y) {
                for (int x = 0; x < halfWidth; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x + halfWidth);

                    std::swap(data[idx1][0], data[idx2][0]);
                    std::swap(data[idx1][1], data[idx2][1]);
                }
            }
        }

        for (int z = 0; z < halfDepth; ++z) {
            for (int y = halfHeight; y < height; ++y) {
                for (int x = halfWidth; x < width; ++x) {
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = (z + halfDepth) * height * width + (y - halfHeight) * width + (x - halfWidth);

                    std::swap(data[idx1][0], data[idx2][0]);
                    std::swap(data[idx1][1], data[idx2][1]);
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

        cv::Mat tmp;                               // Swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);                            // Swap quadrants (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in quadrantShiftMat: " << e.what() << std::endl;
    }
}

// Complex Arithmetic Operations
void GPUBackend::complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) {
    try {
        CUBE_MAT::complexElementwiseMatMulFftwComplex(size, 1, 1, a, b, result);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplication: " << e.what() << std::endl;
    }
}

void GPUBackend::complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) {
    try {
        CUBE_MAT::complexElementwiseMatDivFftwComplex(size, 1, 1, a, b, result, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivision: " << e.what() << std::endl;
    }
}

void GPUBackend::complexAddition(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) {
    try {
        // For addition, we can use a simple loop since it's not available in CUBE
        for (int i = 0; i < size; ++i) {
            result[i][0] = a[i][0] + b[i][0];
            result[i][1] = a[i][1] + b[i][1];
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexAddition: " << e.what() << std::endl;
    }
}

void GPUBackend::scalarMultiplication(fftw_complex* a, double scalar, fftw_complex* result, int size) {
    try {
        // For scalar multiplication, we can use a simple loop since it's not available in CUBE
        for (int i = 0; i < size; ++i) {
            result[i][0] = a[i][0] * scalar;
            result[i][1] = a[i][1] * scalar;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in scalarMultiplication: " << e.what() << std::endl;
    }
}

void GPUBackend::complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) {
    try {
        CUBE_MAT::complexElementwiseMatMulConjugateFftwComplex(size, 1, 1, a, b, result);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexMultiplicationWithConjugate: " << e.what() << std::endl;
    }
}

void GPUBackend::complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) {
    try {
        CUBE_MAT::complexElementwiseMatDivStabilizedFftwComplex(size, 1, 1, a, b, result, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in complexDivisionStabilized: " << e.what() << std::endl;
    }
}

// Conversion Functions
void GPUBackend::convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth) {
    try {
        for (int z = 0; z < depth; ++z) {
            CV_Assert(input[z].type() == CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    output[z * height * width + y * width + x][0] = static_cast<double>(input[z].at<float>(y, x));
                    output[z * height * width + y * width + x][1] = 0.0;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertCVMatVectorToFFTWComplex: " << e.what() << std::endl;
    }
}

void GPUBackend::convertFFTWComplexToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) {
    try {
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    double real_part = input[index][0];
                    double imag_part = input[index][1];
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

void GPUBackend::convertFFTWComplexRealToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) {
    try {
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input[index][0];
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexRealToCVMatVector: " << e.what() << std::endl;
    }
}

void GPUBackend::convertFFTWComplexImgToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) {
    try {
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input[index][1];
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
void GPUBackend::padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth) {
    try {
        octantFourierShift(psf, psf_width, psf_height, psf_depth);
        
        for (int i = 0; i < width * height * depth; ++i) {
            padded_psf[i][0] = 0.0;
            padded_psf[i][1] = 0.0;
        }

        if(psf_depth > depth) {
            std::cerr << "[ERROR] PSF has more layers than image" << std::endl;
        }

        int x_offset = (width - psf_width) / 2;
        int y_offset = (height - psf_height) / 2;
        int z_offset = (depth - psf_depth) / 2;

        for (int z = 0; z < psf_depth; ++z) {
            for (int y = 0; y < psf_height; ++y) {
                for (int x = 0; x < psf_width; ++x) {
                    int padded_index = ((z + z_offset) * height + (y + y_offset)) * width + (x + x_offset);
                    int psf_index = (z * psf_height + y) * psf_width + x;

                    padded_psf[padded_index][0] = psf[psf_index][0];
                    padded_psf[padded_index][1] = psf[psf_index][1];
                }
            }
        }
        octantFourierShift(padded_psf, width, height, depth);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in padPSF: " << e.what() << std::endl;
    }
}

// Specialized Functions
void GPUBackend::calculateLaplacianOfPSF(fftw_complex* psf, fftw_complex* laplacian, int width, int height, int depth) {
    try {
        CUBE_REG::calculateLaplacianFftwComplex(width, height, depth, psf, laplacian);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in calculateLaplacianOfPSF: " << e.what() << std::endl;
    }
}

void GPUBackend::normalizeImage(fftw_complex* resultImage, int size, double epsilon) {
    try {
        CUBE_FTT::normalizeFftwComplexData(1, 1, 1, resultImage); // This might need adjustment for proper normalization
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeImage: " << e.what() << std::endl;
    }
}

void GPUBackend::rescaledInverse(fftw_complex* data, double cubeVolume) {
    try {
        for (int i = 0; i < cubeVolume; ++i) {
            data[i][0] /= cubeVolume;
            data[i][1] /= cubeVolume;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in rescaledInverse: " << e.what() << std::endl;
    }
}

void GPUBackend::saveInterimImages(fftw_complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i) {
    try {
        std::vector<cv::Mat> debugImage;
        convertFFTWComplexToCVMatVector(resultImage, debugImage, imageWidth, imageHeight, imageDepth);
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
void GPUBackend::reorderLayers(fftw_complex* data, int width, int height, int depth) {
    try {
        int layerSize = width * height;
        int halfDepth = depth / 2;
        fftw_complex* temp = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * width * height * depth);

        int destIndex = 0;

        // Copy the middle layer to the first position
        std::memcpy(temp + destIndex * layerSize, data + halfDepth * layerSize, sizeof(fftw_complex) * layerSize);
        destIndex++;

        // Copy the layers after the middle layer
        for (int z = halfDepth + 1; z < depth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data + z * layerSize, sizeof(fftw_complex) * layerSize);
            destIndex++;
        }

        // Copy the layers before the middle layer
        for (int z = 0; z < halfDepth; ++z) {
            std::memcpy(temp + destIndex * layerSize, data + z * layerSize, sizeof(fftw_complex) * layerSize);
            destIndex++;
        }

        // Copy reordered data back to the original array
        std::memcpy(data, temp, sizeof(fftw_complex) * width * height * depth);
        fftw_free(temp);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in reorderLayers: " << e.what() << std::endl;
    }
}

void GPUBackend::visualizeFFT(fftw_complex* data, int width, int height, int depth) {
    try {
        // Convert the result FFTW complex array back to OpenCV Mat vector
        Image3D i;
        std::vector<cv::Mat> output;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = data[index][0];
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
void GPUBackend::gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth) {
    try {
        CUBE_REG::gradXFftwComplex(width, height, depth, image, gradX);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientX: " << e.what() << std::endl;
    }
}

void GPUBackend::gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth) {
    try {
        CUBE_REG::gradYFftwComplex(width, height, depth, image, gradY);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientY: " << e.what() << std::endl;
    }
}

void GPUBackend::gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth) {
    try {
        CUBE_REG::gradZFftwComplex(width, height, depth, image, gradZ);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in gradientZ: " << e.what() << std::endl;
    }
}

void GPUBackend::computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth) {
    try {
        CUBE_REG::computeTVFftwComplex(width, height, depth, lambda, gx, gy, gz, tv);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in computeTV: " << e.what() << std::endl;
    }
}

void GPUBackend::normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon) {
    try {
        CUBE_REG::normalizeTVFftwComplex(width, height, depth, gradX, gradY, gradZ, epsilon);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in normalizeTV: " << e.what() << std::endl;
    }
}