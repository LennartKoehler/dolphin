#include "RLDeconvolutionAlgorithm.h"
#ifdef CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>
#include <operations.h>
#include <utl.h>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;
    this->gpu = config.gpu;

    // General
    this->epsilon = config.epsilon;

    // Grid
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    this->secondpsflayers = config.secondpsflayers;
    this->secondpsfcubes = config.secondpsfcubes;
    this->secondPSF = config.secondPSF;

    // Output
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << std::to_string(this->grid) << std::endl;
    std::cout << "[CONFIGURATION] secondPSF: " << std::to_string(this->secondPSF) << std::endl;

    if(this->grid){
        std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
        std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "[CONFIGURATION] cubeSize: " << this->cubeSize << std::endl;
        if(!this->secondpsflayers.empty()){
            std::cout << "[CONFIGURATION] secondpsflayers: ";
            for (const int& layer : secondpsflayers) {
                std::cout << layer << ", ";
            }
            std::cout << std::endl;
        }
        if(!this->secondpsfcubes.empty()){
            std::cout << "[CONFIGURATION] secondpsfcubes: ";
            for (const int& cube : secondpsfcubes) {
                std::cout << cube << ", ";
            }
            std::cout << std::endl;
        }
    }
    if(this->gpu != "") {
        std::cout << "[CONFIGURATION] gpu: " << this->gpu << std::endl;
    }
}

void RLDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    if(this->gpu == "cuda") {
#ifdef CUDA_AVAILABLE

        //TODO CUDA ERROR: invalid argument
        //std::cout << "[ALGORITHM] Richardson-Lucy algorithm with CUDA" << std::endl;
        cufftComplex *d_c, *d_f, *d_g, *d_H;
        cudaMalloc((void**)&d_c, this->cubeVolume * sizeof(cufftComplex));
        cudaMalloc((void**)&d_f, this->cubeVolume * sizeof(cufftComplex));
        cudaMalloc((void**)&d_g, this->cubeVolume * sizeof(cufftComplex));
        cudaMalloc((void**)&d_H, this->cubeVolume * sizeof(cufftComplex));
        convertFftwToCufftComplexOnDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, g, d_g);
        convertFftwToCufftComplexOnDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, g, d_f);
        convertFftwToCufftComplexOnDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, H, d_H);

        for (int n = 0; n < this->iterations; ++n) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                      << this->iterations << " ";

            // a) First transformation:
            // Fn = FFT(fn)
            cufftForward(d_f, d_f, this->cufftPlan);
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f);

            // Fn' = Fn * H
            complexElementwiseMatMulCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_H, d_c);

            // fn' = IFFT(Fn')
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            cufftInverse(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c, d_c, this->cufftPlan);
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            complexElementwiseMatDivCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_c, d_c, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            cufftForward(d_c, d_c, this->cufftPlan);
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // C' = C * conj(H)
            complexElementwiseMatMulConjugateCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c, d_H, d_c);

            // c' = IFFT(C')
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            cufftInverse(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c, d_c, this->cufftPlan);
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            octantFourierShiftCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            cufftInverse(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_f, this->cufftPlan);

            // fn+1 = fn * c
            complexElementwiseMatMulCufftComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_c, d_f);

            std::flush(std::cout);
        }
        cudaFree(d_c);
        convertCufftToFftwComplexOnHost(this->cubeWidth, this->cubeHeight, this->cubeDepth, f, d_f);


#else
        std::cout << "[ERROR] Cuda is not available" << std::endl;
#endif

    }else if(this->gpu == "opencl") {
        std::cout << "[ERROR] OpenCL is not implemented yet" << std::endl;
    }else if(this->gpu == "" || this->gpu == "none") {
        // Allocate memory for intermediate FFTW arrays
        fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        std::memcpy(f, g, sizeof(fftw_complex) * this->cubeVolume);

        for (int n = 0; n < this->iterations; ++n) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                      << this->iterations << " ";

            // a) First transformation:
            // Fn = FFT(fn)
            fftw_execute_dft(this->forwardPlan, f, f);
            UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // Fn' = Fn * H
            UtlFFT::complexMultiplication(f, H, c, this->cubeVolume);

            // fn' = IFFT(Fn')
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            fftw_execute_dft(this->backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            UtlFFT::complexDivision(g, c, c, this->cubeVolume, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            fftw_execute_dft(this->forwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // C' = C * conj(H)
            UtlFFT::complexMultiplicationWithConjugate(c, H, c, this->cubeVolume);

            // c' = IFFT(C')
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            fftw_execute_dft(backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            fftw_execute_dft(this->backwardPlan, f, f);

            // fn+1 = fn * c
            UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

            // Uncomment the following lines for debugging
            // UtlFFT::normalizeImage(f, size, this->epsilon);
            // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
            // Check size of number
            if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                std::cout << "[WARNING] Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
            }
            std::flush(std::cout);
        }
    fftw_free(c);

    }else {
        std::cout << "[ERROR] Please give a specific GPU API" << std::endl;
    }


}

