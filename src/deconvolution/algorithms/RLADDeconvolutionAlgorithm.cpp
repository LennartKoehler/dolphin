#include "RLADDeconvolutionAlgorithm.h"
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

void RLADDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;
    //this->dampingDecrease = config.dampingDecrease;
    //this->alpha = config.alpha;
    //this->beta = config.beta;
    this->dampingDecrease = 0;
    this->alpha = 0.9;
    this->beta = 0.01;

    // General
    this->epsilon = config.epsilon;
    this->time = config.time;
    this->saveSubimages = config.saveSubimages;



    // Grid
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    this->secondpsflayers = config.secondpsflayers;
    this->secondpsfcubes = config.secondpsfcubes;

    // Output
    std::cout << "[CONFIGURATION] Richardson-Lucy with Adaptive Damping algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] dampingDecrease: " << this->dampingDecrease << std::endl;
    std::cout << "[CONFIGURATION] alpha: " << this->alpha << std::endl;
    std::cout << "[CONFIGURATION] beta: " << this->beta << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << std::to_string(this->grid) << std::endl;

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
}

void RLADDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {

    // Allocate memory for intermediate FFTW arrays
    fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
    std::memcpy(f, g, sizeof(fftw_complex) * this->cubeVolume);

    double a;

    for (int n = 0; n < this->iterations; ++n) {
        std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                  << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                  << this->iterations << " ";

        if(true) {
            a = this->alpha*exp(-this->beta*n);
        }else if(this->dampingDecrease = 1) {
            a = this->alpha-this->beta*n;
        }else {
            a = this->alpha*exp(-this->beta*n);
        }

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

        // c = c * a
        UtlFFT::scalarMultiplication(c, a, c, this->cubeVolume);

        // fn+1' = fn * c
        UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

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
}
