#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmDerived.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include "UtlFFT.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#ifdef CUDA_AVAILABLE
#include <cufftw.h>
#include <CUBE.h>
#else
#include <fftw3.h>
#endif

using namespace std;

Hyperstack BaseDeconvolutionAlgorithmDerived::run(Hyperstack& data, std::vector<PSF>& psfs) {
    return deconvolve(data, psfs);
}

void BaseDeconvolutionAlgorithmDerived::configure(DeconvolutionConfig config) {
    // Store the config in base class
    this->config = config;
    
    // Initialize algorithm-specific parameters
    configureAlgorithmSpecific(config);
    
    // Extract common configuration parameters
    epsilon = config.epsilon;
    time = config.time;
    saveSubimages = config.saveSubimages;
    grid = config.grid;
    gpu = config.gpu;
    borderType = config.borderType;
    psfSafetyBorder = config.psfSafetyBorder;
    cubeSize = config.cubeSize;
}

void BaseDeconvolutionAlgorithmDerived::algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {
    // Call the backend-specific implementation
    algorithmBackendSpecific(channel_num, H, g, f);
}

void BaseDeconvolutionAlgorithmDerived::preprocessPSFS(const std::vector<PSF>& psfs,
    const std::unordered_map<size_t, PSFIndex>& layerPSFMap,
    const std::unordered_map<size_t, PSFIndex>& cubePSFMap){
        std::vector<PSFfftw*> preparedPSFS = preparePSFs(psfs);
        configureLayerMap(preparedPSFS, layerPSFMap);
        configureCubeMap(preparedPSFS, cubePSFMap);    
}

void BaseDeconvolutionAlgorithmDerived::configureGridProcessing(int cubeSize) {
    this->config.cubeSize = cubeSize;
    
    // Validate cube parameters
    if (cubeSize <= 0) {
        cerr << "[ERROR] Invalid cube size: " << cubeSize << endl;
        return;
    }
    
    // Cube size should include safety border
    if (cubeSize < psfSafetyBorder * 2) {
        cout << "[WARNING] Cube size smaller than PSF safety border * 2" << endl;
    }
}

void BaseDeconvolutionAlgorithmDerived::configureLayerMap(const std::vector<PSFfftw*>& psfs, const std::unordered_map<size_t, PSFIndex> layerPSFMap){
    for (auto it : layerPSFMap){
        this->layerPSFMap[it.first] = psfs[it.second];
    }
}

void BaseDeconvolutionAlgorithmDerived::configureCubeMap(const std::vector<PSFfftw*>& psfs, const std::unordered_map<size_t, PSFIndex> cubePSFMap){
    for (auto it : cubePSFMap){
        this->cubePSFMap[it.first] = psfs[it.second];
    }
}


std::vector<PSFfftw*> BaseDeconvolutionAlgorithmDerived::preparePSFs(const std::vector<PSF>& psfs) {
    std::vector<PSFfftw*> preparedPSFS;
    preparedPSFS.reserve(psfs.size());
    cout << "[STATUS] Creating FFTW plans for PSFs..." << endl;
    
    cubeMetaData.originPsfWidth = psfs[0].image.slices[0].cols;
    cubeMetaData.originPsfHeight = psfs[0].image.slices[0].rows;
    cubeMetaData.originPsfDepth = psfs[0].image.slices.size();
    cubeMetaData.originPsfVolume = cubeMetaData.originPsfWidth * cubeMetaData.originPsfHeight * cubeMetaData.originPsfDepth;

    // Calculate safety border for PSF padding
    cubeMetaData.cubeWidth = psfs[0].image.slices[0].cols + (2 * config.psfSafetyBorder);
    cubeMetaData.cubeHeight = psfs[0].image.slices[0].rows + (2 * config.psfSafetyBorder);
    cubeMetaData.cubeDepth = psfs[0].image.slices.size() + (2 * config.psfSafetyBorder);
    cubeMetaData.cubeVolume = cubeMetaData.cubeWidth * cubeMetaData.cubeHeight * cubeMetaData.cubeDepth;

    // Set cube dimensions if not already set
    if (cubeWidth == 0 || cubeHeight == 0 || cubeDepth == 0) {
        cubeWidth = cubeMetaData.cubeWidth;
        cubeHeight = cubeMetaData.cubeHeight;
        cubeDepth = cubeMetaData.cubeDepth;
        cubeVolume = cubeWidth * cubeHeight * cubeDepth;
    }
    
    // Create temporary memory for PSF FFT
    int cubeVolume = cubeWidth * cubeHeight * cubeDepth;
    fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeMetaData.originPsfVolume);
    fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeMetaData.originPsfVolume);
    fftw_plan forwardPSFPlan = fftw_plan_dft_3d(cubeMetaData.originPsfDepth, cubeMetaData.originPsfHeight, cubeMetaData.originPsfWidth, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);

    for (int p = 0; p < psfs.size(); p++) {
        cout << "[STATUS] Performing Fourier Transform on PSF" << p + 1 << "..." << endl;
        
        // Convert PSF to FFTW complex format and execute FFT
        UtlFFT::convertCVMatVectorToFFTWComplex(psfs[p].image.slices, h, cubeMetaData.originPsfWidth, cubeMetaData.originPsfHeight, cubeMetaData.originPsfDepth);
        fftw_execute_dft(forwardPSFPlan, h, h);

        cout << "[STATUS] Padding PSF" << p + 1 << "..." << endl;
        
        // Pad PSF to cube size
        fftw_complex *temp_h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeMetaData.cubeVolume);
        UtlFFT::padPSF(h, cubeMetaData.originPsfWidth, cubeMetaData.originPsfHeight, cubeMetaData.originPsfDepth, temp_h, cubeWidth, cubeHeight, cubeDepth);
        preparedPSFS.push_back(temp_h);
//         // Store padded PSF based on GPU configuration
//         if (config.gpu == "cuda" && paddedHs.size() > 0) {
// #ifdef CUDA_AVAILABLE
//             fftw_complex *d_temp_h;
//             cudaMalloc((void**)&d_temp_h, cubeVolume * sizeof(fftw_complex));
//             CUBE_UTL_COPY::copyDataFromHostToDevice(cubeWidth, cubeHeight, cubeDepth, d_temp_h, temp_h);
//             paddedHs[p] = d_temp_h;
// #endif
//         } else {
//             paddedHs[p] = temp_h;
//         }
    }

    // Clean up temporary resources
    fftw_free(h);
    fftw_free(fftwPSFPlanMem);
    fftw_destroy_plan(forwardPSFPlan);
    
    return preparedPSFS;
}

PSFfftw* BaseDeconvolutionAlgorithmDerived::selectPSFForGridImage(int layerIndex, int cubeIndex) const{
    PSFfftw* psf;
    psf = getPSFForLayer(layerIndex);
    psf = getPSFForCube(cubeIndex);
    return psf;
}

PSFfftw* BaseDeconvolutionAlgorithmDerived::getPSFForLayer(int layerIndex) const {
    PSFfftw* psf;
    for (int v = 1; v < layerPSFMap.size(); ++v) {
        auto itOfLayer = layerPSFMap.find(layerIndex);
        if (itOfLayer != layerPSFMap.end()) {
            psf = itOfLayer->second;
        }
    }
    psf = layerPSFMap.find(0)->second;
    return psf; // Return default PSF index
}

// dont forget to do +1 when calling
PSFfftw* BaseDeconvolutionAlgorithmDerived::getPSFForCube(int cubeIndex) const {
    PSFfftw* psf;
    for (int v = 1; v < cubePSFMap.size(); ++v) {
        auto itOfLayer = cubePSFMap.find(cubeIndex);
        if (itOfLayer != cubePSFMap.end()) {
            psf = itOfLayer->second;
        }
    }
    psf = cubePSFMap.find(0)->second;
    return psf; // Return default PSF index
}


void BaseDeconvolutionAlgorithmDerived::setupCubeArrangement() {
    if (!config.grid) {
        cubesPerX = 1;
        cubesPerY = 1;
        cubesPerZ = 1;
        cubesPerLayer = 1;
    } else {
        cubesPerX = static_cast<int>(std::ceil(static_cast<double>(this->originalImageWidth) / config.cubeSize));
        cubesPerY = static_cast<int>(std::ceil(static_cast<double>(this->originalImageHeight) / config.cubeSize));
        cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(this->originalImageDepth) / config.cubeSize));
        cubesPerLayer = cubesPerX * cubesPerY;
    }
    
    totalGridNum = cubesPerX * cubesPerY * cubesPerZ;
}

bool BaseDeconvolutionAlgorithmDerived::validateImageAndPsfSizes() {
    // Basic validation will be performed during preprocessing
    return true;
}

void BaseDeconvolutionAlgorithmDerived::printConfigurationSummary() const {
    cout << "[CONFIGURATION] Base algorithm configuration" << endl;
    cout << "[CONFIGURATION] epsilon: " << epsilon << endl;
    cout << "[CONFIGURATION] grid: " << (grid ? "true" : "false") << endl;
    cout << "[CONFIGURATION] saveSubimages: " << (saveSubimages ? "true" : "false") << endl;
    cout << "[CONFIGURATION] gpu: " << (gpu.empty() ? "none" : gpu) << endl;
    
    if (grid) {
        cout << "[CONFIGURATION] borderType: " << borderType << endl;
        cout << "[CONFIGURATION] psfSafetyBorder: " << psfSafetyBorder << endl;
        cout << "[CONFIGURATION] cubeSize: " << cubeSize << endl;
    }
    
    cout << "[CONFIGURATION] cubes per layer: " << cubesPerLayer << endl;
    cout << "[CONFIGURATION] layers: " << cubesPerZ << endl;
    cout << "[CONFIGURATION] total cubes: " << totalGridNum << endl;
}

bool BaseDeconvolutionAlgorithmDerived::setupFFTWPlans() {
    cout << "[STATUS] Creating FFTW plans..." << endl;
    
    // Allocate memory for FFTW plans
    fftwPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
    if (fftwPlanMem == nullptr) {
        cerr << "[ERROR] Failed to allocate memory for FFTW plans" << endl;
        return false;
    }
    
    // Create forward and backward FFT plans
    forwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_FORWARD, FFTW_MEASURE);
    backwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_BACKWARD, FFTW_MEASURE);
    
    if (forwardPlan == nullptr || backwardPlan == nullptr) {
        cerr << "[ERROR] Failed to create FFTW plans" << endl;
        return false;
    }
    
    return true;
}
