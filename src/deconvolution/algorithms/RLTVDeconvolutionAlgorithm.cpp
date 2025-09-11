#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#ifdef CUDA_AVAILABLE
#include <cufft.h>
#include <cufftw.h>
#include <CUBE.h>
#else
#include <fftw3.h>
#endif
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>


void RLTVDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    if(this->gpu == "cuda") {
#ifdef CUDA_AVAILABLE
        //INFO H(PSF) memory already allocated on GPU
        fftw_complex *d_c, *d_f, *d_g;
        cudaMalloc((void**)&d_c, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_f, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_g, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        CUBE_UTL_COPY::copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, g);
        CUBE_UTL_COPY::copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, g);

        fftw_complex *d_gx, *d_gy, *d_gz, *d_tv;
        cudaMalloc((void**)&d_gx, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_gy, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_gz, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));
        cudaMalloc((void**)&d_tv, this->cubeWidth* this->cubeHeight* this->cubeDepth * sizeof(fftw_complex));


        for (int n = 0; n < this->iterations; ++n) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                      << this->iterations << " ";

            // Calculate gradients and the Total Variation
            CUBE_REG::gradXFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_gx);
            CUBE_REG::gradYFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_gy);
            CUBE_REG::gradZFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_gz);
            CUBE_REG::normalizeTVFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth,d_gx, d_gy, d_gz, this->epsilon);
            CUBE_REG::gradXFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_gx, d_gx);
            CUBE_REG::gradYFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_gy, d_gy);
            CUBE_REG::gradZFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_gz, d_gz);
            CUBE_REG::computeTVFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth,  this->lambda, d_gx, d_gy, d_gz, d_tv);

            // a) First transformation:
            // Fn = FFT(fn)
            fftw_execute_dft(this->forwardPlan, d_f, d_f);

            // Fn' = Fn * H
            CUBE_MAT::complexElementwiseMatMulFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, H, d_c);

            // fn' = IFFT(Fn')
            fftw_execute_dft(this->backwardPlan, d_c, d_c);
            CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            CUBE_FTT::octantFourierShiftFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            CUBE_MAT::complexElementwiseMatDivStabilizedFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_g, d_c, d_c, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            fftw_execute_dft(this->forwardPlan, d_c, d_c);

            // C' = C * conj(H)
            CUBE_MAT::complexElementwiseMatMulConjugateFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c, H, d_c);

            // c' = IFFT(C')
            fftw_execute_dft(this->backwardPlan, d_c, d_c);
            CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);
            CUBE_FTT::octantFourierShiftFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_c);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            fftw_execute_dft(this->backwardPlan, d_f, d_f);
            CUBE_FTT::normalizeFftwComplexData(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f);
            // fn+1 = fn * c
            CUBE_MAT::complexElementwiseMatMulFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_c, d_f);

            // fn+1 = fn+1' * tv
            CUBE_MAT::complexElementwiseMatMulFftwComplex(this->cubeWidth, this->cubeHeight, this->cubeDepth, d_f, d_tv, d_f);

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

        fftw_complex *gx = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex *gy = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex *gz = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex *tv = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);

        // Calculate gradients and the Total Variation
        UtlFFT::gradientX(g, gx, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientY(g, gy, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientZ(g, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::normalizeTV(gx, gy, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth, this->epsilon);
        UtlFFT::gradientX(gx, gx, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientY(gy, gy, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientZ(gz, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::computeTV(this->lambda, gx, gy, gz, tv, this->cubeWidth, this->cubeHeight, this->cubeDepth);

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

            // fn+1' = fn * c
            UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

            // fn+1 = fn+1' * tv
            UtlFFT::complexMultiplication(f, tv, f, this->cubeVolume);

            // Uncomment the following lines for debugging
            // UtlFFT::normalizeImage(f, size, this->epsilon);
            // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
            // Überprüfung
            if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                std::cout << "[WARNING] Value of f fftcomplex(double) is smaller than float" << std::endl;
            }
            std::flush(std::cout);
        }
        fftw_free(c);
        fftw_free(gx);
        fftw_free(gy);
        fftw_free(gz);
        fftw_free(tv);
    }else {
        std::cout << "[ERROR] Please give a specific GPU API" << std::endl;
    }
}
