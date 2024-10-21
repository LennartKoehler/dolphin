#include <fftw3.h>
#include <opencv2/core.hpp>
#include "RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "UtlImage.h"
#include "UtlFFT.h"
#include "UtlGrid.h"


// Perform deconvolution on the given data using the PSF
Hyperstack RegularizedInverseFilterDeconvolutionAlgorithm::deconvolve(Hyperstack& data, std::vector<PSF>& psf) {

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
            fftw_complex *image = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            fftw_complex* H2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            fftw_complex* L = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            fftw_complex* L2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            fftw_complex* FA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            fftw_complex* FP = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);

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

            std::cout << "\rChannel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                      << totalGridNum << "/" << this->gridImages.size() << " ";
            //Convert image to fftcomplex
            UtlFFT::convertCVMatVectorToFFTWComplex(this->gridImages[i], image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // Forward FFT on image
            fftw_execute_dft(forwardPlan, image, image);
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // H*H
            UtlFFT::complexMultiplication(H, H, H2, this->cubeVolume);
            // Laplacian L
            UtlFFT::calculateLaplacianOfPSF(H, L, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            UtlFFT::complexMultiplication(L, L, L2, this->cubeVolume);
            UtlFFT::scalarMultiplication(L2, this->lambda, L2, this->cubeVolume);

            UtlFFT::complexAddition(H2, L2, FA, this->cubeVolume);
            UtlFFT::complexDivisionStabilized(H, FA, FP, this->cubeVolume, this->epsilon);
            UtlFFT::complexMultiplication(image, FP, image, this->cubeVolume);

            // Inverse FFT
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            fftw_execute_dft(backwardPlan, image, image);
            UtlFFT::octantFourierShift(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);
            //TODO
            UtlFFT::reorderLayers(image, this->cubeWidth, this->cubeHeight, this->cubeDepth);

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
void RegularizedInverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    this->epsilon = config.epsilon;
    this->grid = config.grid;
    this->lambda = config.lambda;
    std::cout << "[CONFIGURATION] Regularized Inverse Filter" << std::endl;
    std::cout << "[CONFIGURATION] lambda: " << this->lambda << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << this->grid << std::endl;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    this->secondpsflayers = config.secondpsflayers;
    this->secondpsfcubes = config.secondpsfcubes;
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
