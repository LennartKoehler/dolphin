
#include "InverseFilterDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include <cstring>
#include "UtlGrid.h"
#include "UtlFFT.h"
#include "UtlImage.h"


// Perform deconvolution on the given data using the PSF
Hyperstack InverseFilterDeconvolutionAlgorithm::deconvolve(Hyperstack& data, std::vector<PSF>& psf) {

    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};

    // Deconvolve every channel
    int channel_z = 0;
    for (auto& channel : data.channels) {
        if(preprocess(channel, psf)){
            std::cout << "[STATUS] Preprocessing channel " << channel_z + 1 << " finished" << std::endl;
        }else{
            std::cerr << "[ERROR] Preprocessing channel " << channel_z + 1 << " failed" << std::endl;
            return deconvHyperstack;
        }

        std::cout << "[STATUS] Running Deconvolution..." << std::endl;
        int totalGridNum = 1;
        int cubesPerX = static_cast<int>(std::ceil(static_cast<double>(this->originalImageWidth) / this->cubeSize));
        int cubesPerY = static_cast<int>(std::ceil(static_cast<double>(this->originalImageHeight) / this->cubeSize));
        int cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(this->originalImageDepth) / this->cubeSize));
        int cubesPerLayer = cubesPerX * cubesPerY;
        std::cout << "[INFO] Cubes per Layer(" << cubesPerZ<< "):" << cubesPerX << "x" << cubesPerY << " (" << cubesPerLayer << ")" << std::endl;

        // Parallelization of grid for
        // Using static scheduling because the execution time for each iteration is similar, which reduces overhead costs by minimizing task assignment.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < this->gridImages.size(); ++i) {
            int gridNum = static_cast<int>(i);
            // Allocate memory for intermediate FFTW arrays
            fftw_complex *image = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
            fftw_complex* H = nullptr;
            int currentCubeLayer = static_cast<int>(std::ceil(static_cast<double>((i+1)) / cubesPerLayer));
            // Verwende std::find, um den Wert zu suchen
            auto useSecondPsfForThisLayer = std::find(secondpsflayers.begin(), secondpsflayers.end(), currentCubeLayer);
            auto useSecondPsfForThisCube = std::find(secondpsfcubes.begin(), secondpsfcubes.end(), gridNum+1);

            // Überprüfen, ob der Wert gefunden wurde
            if (useSecondPsfForThisLayer != secondpsflayers.end() ||  useSecondPsfForThisCube != secondpsfcubes.end()) {
                //std::cout << "[DEBUG] first PSF" << std::endl;
                H = this->paddedH;
            } else {
                //std::cout << "[DEBUG] second PSF" << std::endl;
                H = this->paddedH_2;
            }

            std::flush(std::cout);

            std::cout << "\r[STATUS] Channel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                      << totalGridNum << "/" << this->gridImages.size() << " ";
            //Convert image to fftcomplex
            UtlFFT::convertCVMatVectorToFFTWComplex(this->gridImages[i], image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // Forward FFT on image
            fftw_execute_dft(forwardPlan, image, image);
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // Division in frequency domain
            UtlFFT::complexDivisionStabilized(image, H, image, this->cubeVolume, this->epsilon);

            // Inverse FFT
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            fftw_execute_dft(backwardPlan, image, image);
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // Convert the result FFTW complex array back to OpenCV Mat vector
            UtlFFT::convertFFTWComplexToCVMatVector(image, this->gridImages[i], this->cubeWidth, this->cubeHeight, this->cubeDepth);

            gridNum++;
            std::flush(std::cout);

        }

        if(postprocess(data, this->epsilon)){
            std::cout << "[STATUS] Postprocessing channel " << channel_z + 1 << " finished" << std::endl;
        }else{
            std::cerr << "[ERROR] Postprocessing channel " << channel_z + 1 << " failed" << std::endl;
            return deconvHyperstack;
        }
        // Save the result
        std::cout << "[STATUS] Saving result of channel " << channel_z + 1 << std::endl;
        Image3D deconvolutedImage;
        deconvolutedImage.slices = this->mergedVolume;
        deconvHyperstack.channels[channel.id].image = deconvolutedImage;
        channel_z++;
        this->mergedVolume.clear();
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return deconvHyperstack;
}

// Configure the deconvolution parameters
void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    this->epsilon = config.epsilon;
    this->grid = config.grid;
    std::cout << "[CONFIGURATION] Naive Inverse Filter" << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << this->grid << std::endl;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    if(this->grid){
        std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
        std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "[CONFIGURATION] cubeSize: " << this->cubeSize << std::endl;
    }
}
