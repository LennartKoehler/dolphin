#include "RLDeconvolutionAlgorithm.h"
#ifdef CUDA_AVAILABLE
#include <CUBE.h>
#else
#include <fftw3.h>
#endif
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>




void normalizeFFTW(fftw_complex* data, std::size_t sizeX, std::size_t sizeY, std::size_t sizeZ) {
    // Gesamtanzahl der Elemente
    std::size_t totalSize = sizeX * sizeY * sizeZ;
    double normalizationFactor = 1.0 / totalSize;

    // Schleife über alle Elemente und Normalisierung
    for (std::size_t i = 0; i < totalSize; ++i) {
        data[i][0] *= normalizationFactor; // Realteil normalisieren
        data[i][1] *= normalizationFactor; // Imaginärteil normalisieren
    }
}
void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;

    // General
    this->epsilon = config.epsilon;
    this->gpu = config.gpu;

    // Grid
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    this->secondpsflayers = config.secondpsflayers;
    this->secondpsfcubes = config.secondpsfcubes;

    //TODO also in other algo classes!!!
    this->cubeNumVec = config.psfCubeVec;
    this->layerNumVec = config.psfLayerVec;


    // Output
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << std::to_string(this->grid) << std::endl;
    std::cout << "[CONFIGURATION] secondPSF: " << std::to_string(this->secondPSF) << std::endl;

    if(this->grid){
        std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
        std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "[CONFIGURATION] subimageSize: " << this->cubeSize << std::endl;
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
        //INFO H(PSF) memory already allocated on GPU
        fftw_complex *d_c, *d_f, *d_g;
        cudaMalloc((void**)&d_c, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_f, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_g, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        CUBE_UTL_COPY::copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, g);
        CUBE_UTL_COPY::copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, g);

        for (int n = 0; n < this->iterations; ++n) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                      << this->iterations << " ";

            // a) First transformation:
            // Fn = FFT(fn)
            fftw_execute_dft(this->forwardPlan, d_f, d_f);

            // Fn' = Fn * H
            CUBE_MAT::complexElementwiseMatMulFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, H, d_c);

            // fn' = IFFT(Fn')
            fftw_execute_dft(this->backwardPlan, d_c, d_c);
            //CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            CUBE_FTT::octantFourierShiftFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            CUBE_MAT::complexElementwiseMatDivFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_c, d_c, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            fftw_execute_dft(this->forwardPlan, d_c, d_c);

            // C' = C * conj(H)
            CUBE_MAT::complexElementwiseMatMulConjugateFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c, H, d_c);

            // c' = IFFT(C')
            fftw_execute_dft(this->backwardPlan, d_c, d_c);
            //CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            CUBE_FTT::octantFourierShiftFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            fftw_execute_dft(this->backwardPlan, d_f, d_f);
            //CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f);
            // fn+1 = fn * c
            CUBE_MAT::complexElementwiseMatMulFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_c, d_f);

            std::flush(std::cout);
        }
        CUBE_UTL_COPY::copyDataFromDeviceToHost(this->cubeWidth, this->cubeHeight, this->cubeDepth, f, d_f);
        cudaFree(d_c);
        cudaFree(d_f);
        cudaFree(d_g);

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
            // Fn' = Fn * H
            UtlFFT::complexMultiplication(f, H, c, this->cubeVolume);

            // fn' = IFFT(Fn')
            fftw_execute_dft(this->backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            UtlFFT::complexDivision(g, c, c, this->cubeVolume, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            fftw_execute_dft(this->forwardPlan, c, c);

            // C' = C * conj(H)
            UtlFFT::complexMultiplicationWithConjugate(c, H, c, this->cubeVolume);

            // c' = IFFT(C')
            fftw_execute_dft(backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            fftw_execute_dft(this->backwardPlan, f, f);

            // fn+1 = fn * c
            UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

            // Uncomment the following lines for debugging
            // UtlFFT::normalizeImage(f, size, this->epsilon);
            // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
            // Check size of number
            if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                std::cout << "[WARNING] Value of f fftwcomplex(double) is smaller than float" << std::endl;
            }

            std::flush(std::cout);
        }
    fftw_free(c);

    }else {
        std::cout << "[ERROR] Please give a specific GPU API" << std::endl;
    }


}

