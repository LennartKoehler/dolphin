#include "deconvolution/DeconvolutionProcessor.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include "UtlFFT.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <fftw3.h>

using namespace std;

#include "backend/CPUBackend.h" // replace with backend factory

Hyperstack DeconvolutionProcessor::run(Hyperstack& input, const std::unordered_map<size_t, std::shared_ptr<PSF>>& psfs){
    preprocess(input, psfs);
    for (auto channel : input.channels){
        std::vector<std::vector<cv::Mat>> gridImages = preprocessChannel(channel);
        for (int gridIndex = 0; gridIndex < gridImages.size(); gridIndex++){ // TODO this loop can be concurrent
            deconvolveSingle(gridIndex, gridImages[gridIndex]);
        }

        std::vector<cv::Mat> deconvolutedChannel = postprocessChannel(input.metaData, gridImages);
        // Save the result
        // TODO clean up:
        std::cout << "[STATUS] Saving result of channel " << std::endl;
        Image3D deconvolutedImage;
        deconvolutedImage.slices = deconvolutedChannel;
        input.channels[channel.id].image = deconvolutedImage;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return input;
}

// refactor
std::vector<cv::Mat> DeconvolutionProcessor::deconvolveSingle(int gridIndex, std::vector<cv::Mat>& gridImages){

    int layerIndex = static_cast<int>(std::ceil(static_cast<double>((gridIndex+1)) / cubes.cubesPerLayer));
    int cubeIndex = gridIndex + 1;
    // PSF
    fftw_complex* tempH = selectPSFForGridImage(layerIndex, cubeIndex);
    FFTWData H = {tempH, cubeShape};
    // Observed image
    fftw_complex* tempg = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
    FFTWData g = {tempg, cubeShape};

    if (!(UtlImage::isValidForFloat(g.data, cubeShape.volume))) {
        std::cout << "[WARNING] Value fftwPlanMem fftwcomplex(double) is smaller than float" << std::endl;
    }
    UtlFFT::convertCVMatVectorToFFTWComplex(gridImages, g.data, cubeShape.width, cubeShape.height, cubeShape.depth);

    FFTWData H_device = backend_->moveDataToDevice(H);
    FFTWData g_device = backend_->moveDataToDevice(g);
    FFTWData f_device = backend_->allocateMemoryOnDevice(cubeShape);

    algorithm_->deconvolve(H_device, g_device, f_device);
    FFTWData f = backend_->moveDataFromDevice(f_device);
    backend_->freeMemoryOnDevice(H_device);
    backend_->freeMemoryOnDevice(g_device);
    backend_->freeMemoryOnDevice(f_device);

    // Convert the result FFTW complex array back to OpenCV Mat vector
    UtlFFT::convertFFTWComplexToCVMatVector(f.data, gridImages, cubeShape.width, cubeShape.height, cubeShape.depth);
    fftw_free(g.data);
    return gridImages;
}

// TODO refactor
std::vector<cv::Mat> DeconvolutionProcessor::postprocessChannel(ImageMetaData& metaData, std::vector<std::vector<cv::Mat>>& gridImages){

    if(gridImages.empty()){
        std::cerr << "[ERROR] No subimages processed" << std::endl;
    }

    std::vector<cv::Mat> result;
    ////////////////////////////////////////
    if(config.saveSubimages){

        std::vector<std::vector<cv::Mat>> tiles(gridImages); // Kopie erstellen
        UtlGrid::cropCubePadding(tiles, cubes.cubePadding);

        double global_max_val_tile= 0.0;
        double global_min_val_tile = MAXFLOAT;
        for(auto gridImage : tiles){
            // Global normalization of the merged volume
            for (const auto& slice : gridImage) {
                cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
                double min_val, max_val;
                cv::minMaxLoc(slice, &min_val, &max_val);
                global_max_val_tile = std::max(global_max_val_tile, max_val);
                global_min_val_tile = std::min(global_min_val_tile, min_val);
            }
        }
        int num = 1;
        for(auto gridImage : tiles){
        // Global normalization of the merged volume

        for (auto& slice : gridImage) {
            slice.convertTo(slice, CV_32F, 1.0 / (global_max_val_tile - global_min_val_tile), -global_min_val_tile * (1 / (global_max_val_tile - global_min_val_tile)));  // Add epsilon to avoid division by zero
            cv::threshold(slice, slice, config.epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        }
            Hyperstack gridImageHyperstack;
            gridImageHyperstack.metaData = metaData;
            gridImageHyperstack.metaData.imageWidth = cubeShape.width-(2*this->cubePadding);
            gridImageHyperstack.metaData.imageLength = cubeShape.width-(2*this->cubePadding);
            gridImageHyperstack.metaData.slices = cubeShape.width-(2*this->cubePadding);

            Image3D image3d;
            image3d.slices = gridImage;
            Channel channel;
            channel.image = image3d;
            gridImageHyperstack.channels.push_back(channel);
            gridImageHyperstack.saveAsTifFile("../result/tiles/deconv_"+std::to_string(num)+".tif");
            num++;
        }
    }
    //////////////////////////////////////////////
    if(config.grid){
        //TODO no effect
        //UtlGrid::adjustCubeOverlap(gridImages,this->cubePadding);

        UtlGrid::cropCubePadding(gridImages, this->cubePadding);
        std::cout << "[STATUS] Merging Grid back to Image..." << std::endl;
        result = UtlGrid::mergeCubes(gridImages, imageOriginalShape.width, imageOriginalShape.height, imageOriginalShape.depth, config.cubeSize);
        std::cout << "[INFO] Image size: " << result[0].rows << "x" << result[0].cols << "x" << result.size()<< std::endl;
    }else{
        result = gridImages[0];
    }

    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    for (const auto& slice : result) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }

    for (auto& slice : result) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));  // Add epsilon to avoid division by zero
        cv::threshold(slice, slice, config.epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    }
    return result;
}

void DeconvolutionProcessor::preprocess(const Hyperstack& input, const std::unordered_map<size_t, std::shared_ptr<PSF>>& psfs){
    if (!configured){
        std::__throw_runtime_error("Processor not configured");
    }
    preprocessPSF(psfs);
    setPSFShape(*(psfs.begin()->second));
    setImageOriginalShape(input.channels[0]); // i tihnk every channel should be same size
    setCubeShape(imageOriginalShape, config.grid, config.cubeSize, config.psfSafetyBorder);
    setupCubeArrangement();
    backend_->preprocess();
    // algorithm_->preprocess();

}

void DeconvolutionProcessor::configure(DeconvolutionConfig config) {
    // Store the config in base class
    this->config = config;
    DeconvolutionAlgorithmFactory fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = std::make_unique<DeconvolutionAlgorithm>(fact.create(config));

    this->backend_ = std::make_unique<CPUBackend>(); // TODO LK  to use the config and a factory
    configured = true;
}


void DeconvolutionProcessor::setPSFShape(const PSF& psf) {
    // Original PSF dimensions
    psfOriginalShape.width = psf.image.slices[0].cols;
    psfOriginalShape.height = psf.image.slices[0].rows;
    psfOriginalShape.depth = psf.image.slices.size();
    psfOriginalShape.volume = psfOriginalShape.width * psfOriginalShape.height * psfOriginalShape.depth;
}


void DeconvolutionProcessor::setImageOriginalShape(const Channel& channel) {
    imageOriginalShape.width = channel.image.slices[0].cols;
    imageOriginalShape.height = channel.image.slices[0].rows;
    imageOriginalShape.depth = channel.image.slices.size();
    imageOriginalShape.volume = imageOriginalShape.width * imageOriginalShape.height * imageOriginalShape.depth;
}

void DeconvolutionProcessor::setCubeShape(
    const RectangleShape& imageOriginalShape,
    bool configgrid,
    int configcubeSize,
    int configpsfSafetyBorder
) {
    if (!configgrid) {
        // Non-grid processing - cube equals entire image
        cubeShape.width = imageOriginalShape.width;
        cubeShape.height = imageOriginalShape.height;
        cubeShape.depth = imageOriginalShape.depth;
        cubeShape.volume = imageOriginalShape.volume;
    } else {
        // Grid processing - calculate cube size with padding
        // This logic mirrors BaseDeconvolutionAlgorithm::preprocess
        cubePadding = configpsfSafetyBorder;
        
        // Adjust padding based on PSF size (from original logic)
        int safetyBorderPsfWidth = psfOriginalShape.width + (2 * configpsfSafetyBorder);
        
        if (safetyBorderPsfWidth < configcubeSize) {
            cubePadding = 10;
        }
        if (configcubeSize + 2 * cubePadding < safetyBorderPsfWidth) {
            cubePadding = (safetyBorderPsfWidth - configcubeSize) / 2;
        }
        
        cubeShape.width = configcubeSize + (2 * cubePadding);
        cubeShape.height = configcubeSize + (2 * cubePadding);
        cubeShape.depth = configcubeSize + (2 * cubePadding);
        cubeShape.volume = cubeShape.width * cubeShape.height * cubeShape.depth;
    }
}

std::vector<std::vector<cv::Mat>> DeconvolutionProcessor::preprocessChannel(Channel& channel){
    int imagepadding = imageOriginalShape.width/2;
    UtlGrid::extendImage(channel.image.slices, imagepadding, config.borderType);
    return UtlGrid::splitWithCubePadding(channel.image.slices, config.cubeSize, imagepadding, cubePadding);
}

void DeconvolutionProcessor::preprocessPSF(
    const std::unordered_map<size_t, std::shared_ptr<PSF>>& inputPSFs
    ) {
        std::unordered_map<size_t, PSFfftw*>  preparedPSFS;
        preparedPSFS.reserve(inputPSFs.size());
        cout << "[STATUS] Creating FFTW plans for PSFs..." << endl;
        

        fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psfOriginalShape.volume);
        fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psfOriginalShape.volume);
        fftw_plan forwardPSFPlan = fftw_plan_dft_3d(psfOriginalShape.depth, psfOriginalShape.height, psfOriginalShape.width, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);

        int counter = 0;
        for (auto psfit : inputPSFs) {
            counter ++;
            cout << "[STATUS] Performing Fourier Transform on PSF" << counter<< "..." << endl;
            
            // Convert PSF to FFTW complex format and execute FFT
            UtlFFT::convertCVMatVectorToFFTWComplex(psfit.second->image.slices, h, psfOriginalShape.width, psfOriginalShape.height, psfOriginalShape.depth);
            fftw_execute_dft(forwardPSFPlan, h, h);

            cout << "[STATUS] Padding PSF" << counter << "..." << endl;
            
            // Pad PSF to cube size
            fftw_complex *temp_h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
            UtlFFT::padPSF(h, psfOriginalShape.width, psfOriginalShape.height, psfOriginalShape.depth, temp_h, cubeShape.width, cubeShape.height, cubeShape.depth);
            preparedPSFS[psfit.first] = temp_h;
        }

        // Clean up temporary resources
        fftw_free(h);
        fftw_free(fftwPSFPlanMem);
        fftw_destroy_plan(forwardPSFPlan);
        
        this->layerPSFMap = preparedPSFS;
    }

PSFfftw* DeconvolutionProcessor::selectPSFForGridImage(int layerIndex, int cubeIndex) const{
    PSFfftw* psf;
    psf = getPSFForLayer(layerIndex);
    psf = getPSFForCube(cubeIndex);
    return psf;
}


PSFfftw* DeconvolutionProcessor::getPSFForLayer(int layerIndex) const {
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
PSFfftw* DeconvolutionProcessor::getPSFForCube(int cubeIndex) const {
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


void DeconvolutionProcessor::setupCubeArrangement() {
    if (!config.grid) {
        cubes.cubesPerX = 1;
        cubes.cubesPerY = 1;
        cubes.cubesPerZ = 1;
        cubes.cubesPerLayer = 1;
    } else {
        cubes.cubesPerX = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.width) / config.cubeSize));
        cubes.cubesPerY = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.height) / config.cubeSize));
        cubes.cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.depth) / config.cubeSize));
        cubes.cubesPerLayer = cubes.cubesPerX * cubes.cubesPerY;
    }
    
    cubes.totalGridNum = cubes.cubesPerX * cubes.cubesPerY * cubes.cubesPerZ;
}

bool DeconvolutionProcessor::validateImageAndPsfSizes() {
    // Basic validation will be performed during preprocessing
    return true;
}

void DeconvolutionProcessor::cleanup() {
    // Clean up FFTW resources

    if (fftwPlanMem) {
        fftw_free(fftwPlanMem);
        fftwPlanMem = nullptr;
    }
    if (forwardPlan) {
        fftw_destroy_plan(forwardPlan);
        forwardPlan = nullptr;
    }
    if (backwardPlan) {
        fftw_destroy_plan(backwardPlan);
        backwardPlan = nullptr;
    }
    
    // Clean up PSF maps
    for (auto& psf : layerPSFMap) {
        if (psf.second) {
            fftw_free(psf.second);
            psf.second = nullptr;
        }
    }
    layerPSFMap.clear();
    
    for (auto& psf : cubePSFMap) {
        if (psf.second) {
            fftw_free(psf.second);
            psf.second = nullptr;
        }
    }
    cubePSFMap.clear();
    
}

// void DeconvolutionProcessor::printConfigurationSummary() const {
//     cout << "[CONFIGURATION] Base algorithm configuration" << endl;
//     cout << "[CONFIGURATION] epsilon: " << epsilon << endl;
//     cout << "[CONFIGURATION] grid: " << (grid ? "true" : "false") << endl;
//     cout << "[CONFIGURATION] saveSubimages: " << (saveSubimages ? "true" : "false") << endl;
//     cout << "[CONFIGURATION] gpu: " << (gpu.empty() ? "none" : gpu) << endl;
    
//     if (grid) {
//         cout << "[CONFIGURATION] borderType: " << borderType << endl;
//         cout << "[CONFIGURATION] psfSafetyBorder: " << psfSafetyBorder << endl;
//         cout << "[CONFIGURATION] cubeSize: " << cubeSize << endl;
//     }
    
//     cout << "[CONFIGURATION] cubes per layer: " << cubesPerLayer << endl;
//     cout << "[CONFIGURATION] layers: " << cubesPerZ << endl;
//     cout << "[CONFIGURATION] total cubes: " << totalGridNum << endl;
// }

bool DeconvolutionProcessor::setupFFTWPlans() {
    cout << "[STATUS] Creating FFTW plans..." << endl;
    
    // Allocate memory for FFTW plans
    fftwPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
    if (fftwPlanMem == nullptr) {
        cerr << "[ERROR] Failed to allocate memory for FFTW plans" << endl;
        return false;
    }
    
    // Create forward and backward FFT plans
    forwardPlan = fftw_plan_dft_3d(cubeShape.depth, cubeShape.height, cubeShape.width, fftwPlanMem, fftwPlanMem, FFTW_FORWARD, FFTW_MEASURE);
    backwardPlan = fftw_plan_dft_3d(cubeShape.depth, cubeShape.height, cubeShape.width, fftwPlanMem, fftwPlanMem, FFTW_BACKWARD, FFTW_MEASURE);
    
    if (forwardPlan == nullptr || backwardPlan == nullptr) {
        cerr << "[ERROR] Failed to create FFTW plans" << endl;
        return false;
    }
    
    return true;
}
