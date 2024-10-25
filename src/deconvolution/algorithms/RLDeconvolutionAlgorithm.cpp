#include "RLDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;

    // General
    this->epsilon = config.epsilon;

    // Grid
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    this->secondpsflayers = config.secondpsflayers;
    this->secondpsfcubes = config.secondpsfcubes;

    // Output
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << std::to_string(this->grid) << std::endl;
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
        std::cout << "[CONFIGURATION] secondpsfcubes: ";
        if(!this->secondpsfcubes.empty()){
            std::cout << "[CONFIGURATION] secondpsfcubes: ";
            for (const int& cube : secondpsfcubes) {
                std::cout << cube << ", ";
            }
            std::cout << std::endl;
        }
    }
}

void RLDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num) {
// Parallelization of grid for
// Using static scheduling because the execution time for each iteration is similar, which reduces overhead costs by minimizing task assignment.
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->gridImages.size(); ++i) {
        int gridNum = static_cast<int>(i);
        // Allocate memory for intermediate FFTW arrays
        // INFO
        // if allocations takes to much memory put outside the loop (dont forget the free lines at the end)
        fftw_complex *g = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex *f = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        fftw_complex* H = nullptr;

        // Check if second PSF has to be applied
        int currentCubeLayer = static_cast<int>(std::ceil(static_cast<double>((i+1)) / this->cubesPerLayer));
        auto useSecondPsfForThisLayer = std::find(secondpsflayers.begin(), secondpsflayers.end(), currentCubeLayer);
        auto useSecondPsfForThisCube = std::find(secondpsfcubes.begin(), secondpsfcubes.end(), gridNum+1);
        // Load the correct PSF
        if (useSecondPsfForThisLayer != secondpsflayers.end() ||  useSecondPsfForThisCube != secondpsfcubes.end()) {
            //std::cout << "[DEBUG] first PSF" << std::endl;
            H = this->paddedH;
        } else {
            //std::cout << "[DEBUG] second PSF" << std::endl;
            H = this->paddedH_2;
        }

        // Convert image to fftcomplex
        UtlFFT::convertCVMatVectorToFFTWComplex(this->gridImages[i], g, this->cubeWidth, this->cubeHeight, this->cubeDepth);
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
            // Überprüfung
            if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                std::cout << "[WARNING] Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
            }
            std::flush(std::cout);
        }
        // Convert the result FFTW complex array back to OpenCV Mat vector
        UtlFFT::convertFFTWComplexToCVMatVector(f, this->gridImages[i], this->cubeWidth, this->cubeHeight, this->cubeDepth);

        gridNum++;
        this->totalGridNum++;
        fftw_free(g);
        fftw_free(c);
        fftw_free(f);
    }
    // this->girdImages of BaseDeconvolutionAlgorithm deconvolution complete
}
