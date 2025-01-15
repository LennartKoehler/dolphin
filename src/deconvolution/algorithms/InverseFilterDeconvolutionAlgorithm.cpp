
#include "InverseFilterDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstring>
#include "UtlGrid.h"
#include "UtlFFT.h"
#include "UtlImage.h"



// Configure the deconvolution parameters
void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // General
    this->epsilon = config.epsilon;

    // Grid
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;


    // Output
    std::cout << "[CONFIGURATION] Naive Inverse Filter" << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << this->grid << std::endl;
    std::cout << "[CONFIGURATION] secondPSF: " << std::to_string(this->secondPSF) << std::endl;

    if(this->grid){
        std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
        std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "[CONFIGURATION] subimageSize: " << this->cubeSize << std::endl;
    }
}

void InverseFilterDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
        // Forward FFT on image
        fftw_execute_dft(forwardPlan, g, g);

        // Division in frequency domain
        UtlFFT::complexDivisionStabilized(g, H, f, this->cubeVolume, this->epsilon);

        // Inverse FFT
        fftw_execute_dft(backwardPlan, f, f);
        UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);
}
