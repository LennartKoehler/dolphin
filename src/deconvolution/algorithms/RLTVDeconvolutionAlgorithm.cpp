#include "RLTVDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#ifdef CUDA_AVAILABLE
#include <cufft.h>
#include <cufftw.h>
#else
#include <fftw3.h>
#endif
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>

void RLTVDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;
    this->lambda = config.lambda;

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
    std::cout << "[CONFIGURATION] Richardson-Lucy with Total Variation algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] lambda: " << this->lambda << std::endl;
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
}

void RLTVDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {

    // Allocate memory for intermediate FFTW arrays
    fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    std::memcpy(f, g, sizeof(fftw_complex) * this->cubeVolume);

    fftw_complex *gx = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    fftw_complex *gy = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    fftw_complex *gz = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    // Memory reuse
    //fftw_complex *ggx = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    //fftw_complex *ggy = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    //fftw_complex *ggz = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    fftw_complex *tv = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);

    for (int n = 0; n < this->iterations; ++n) {
        std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                  << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                  << this->iterations << " ";

        // Calculate gradients and the Total Variation
        UtlFFT::gradientX(g, gx, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientY(g, gy, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientZ(g, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::normalizeTV(gx, gy, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth, this->epsilon);
        UtlFFT::gradientX(gx, gx, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientY(gy, gy, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::gradientZ(gz, gz, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::computeTV(this->lambda, gx, gy, gz, tv, this->cubeWidth, this->cubeHeight, this->cubeDepth);

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

        // fn+1' = fn * c
        UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

        // fn+1 = fn+1' * tv
        UtlFFT::complexMultiplication(f, tv, f, this->cubeVolume);

        // Uncomment the following lines for debugging
        // UtlFFT::normalizeImage(f, size, this->epsilon);
        // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
        // Überprüfung
        if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
            std::cout << "[WARNING] Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
        }
        std::flush(std::cout);
    }
    fftw_free(c);
    fftw_free(gx);
    fftw_free(gy);
    fftw_free(gz);
    //fftw_free(ggx);
    //fftw_free(ggy);
    //fftw_free(ggz);
    fftw_free(tv);


}
