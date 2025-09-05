#include <opencv2/core.hpp>
#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "UtlImage.h"
#include "UtlFFT.h"
#include "UtlGrid.h"



// Configure the deconvolution parameters
void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->lambda = config.lambda;

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
    std::cout << "[CONFIGURATION] Regularized Inverse Filter" << std::endl;
    std::cout << "[CONFIGURATION] lambda: " << this->lambda << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << this->grid << std::endl;

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

void RegularizedInverseFilterDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
        // Allocate memory for intermediate FFTW arrays
        fftw_complex* H2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex* L = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex* L2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex* FA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex* FP = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);


        // Forward FFT on image
        fftw_execute_dft(forwardPlan, g, g);

        // H*H
        UtlFFT::complexMultiplication(H, H, H2, this->cubeVolume);
        // Laplacian L
        UtlFFT::calculateLaplacianOfPSF(H, L, this->cubeWidth, this->cubeHeight, this->cubeDepth);
        UtlFFT::complexMultiplication(L, L, L2, this->cubeVolume);
        UtlFFT::scalarMultiplication(L2, this->lambda, L2, this->cubeVolume);

        UtlFFT::complexAddition(H2, L2, FA, this->cubeVolume);
        UtlFFT::complexDivisionStabilized(H, FA, FP, this->cubeVolume, this->epsilon);
        UtlFFT::complexMultiplication(g, FP, f, this->cubeVolume);

        // Inverse FFT
        fftw_execute_dft(backwardPlan, f, f);
        UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);
}
