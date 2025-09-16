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
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "deconvolution/backend/CPUBackend.h" // replace with backend factory

Hyperstack DeconvolutionProcessor::run(Hyperstack& input, const std::vector<PSF>& psfs){
    preprocess(input, psfs);
    for (auto channel : input.channels){
        std::vector<std::vector<cv::Mat>> cubeImages = preprocessChannel(channel);
        for (int cubeIndex = 0; cubeIndex < cubeImages.size(); cubeIndex++){ // TODO this loop can be concurrent
            deconvolveSingleCube(cubeIndex, cubeImages[cubeIndex]);
        }

        std::vector<cv::Mat> deconvolutedChannel = postprocessChannel(input.metaData, cubeImages);
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



void DeconvolutionProcessor::deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage){

    int cubeIndex = cubeIndex + 1;// because not 0 based
    // PSF
    std::vector<fftw_complex*> psfs = selectPSFsForCube(cubeIndex);
    for (auto psf: psfs){
        deconvolveSingleCubePSF(psf, cubeImage);
    }
}



void DeconvolutionProcessor::deconvolveSingleCubePSF(fftw_complex* psf, std::vector<cv::Mat>& cubeImage){
    FFTWData H = {psf, cubeShape};
    // Observed image
    fftw_complex* tempg = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
    FFTWData g = {tempg, cubeShape};

    if (!(UtlImage::isValidForFloat(g.data, cubeShape.volume))) {
        std::cout << "[WARNING] Value fftwPlanMem fftwcomplex(double) is smaller than float" << std::endl;
    }
    UtlFFT::convertCVMatVectorToFFTWComplex(cubeImage, g.data, cubeShape.width, cubeShape.height, cubeShape.depth);

    FFTWData H_device = backend_->moveDataToDevice(H);
    FFTWData g_device = backend_->moveDataToDevice(g);
    FFTWData f_device = backend_->allocateMemoryOnDevice(cubeShape);

    algorithm_->deconvolve(H_device, g_device, f_device);
    FFTWData f = backend_->moveDataFromDevice(f_device);
    backend_->freeMemoryOnDevice(H_device);
    backend_->freeMemoryOnDevice(g_device);
    backend_->freeMemoryOnDevice(f_device);

    // Convert the result FFTW complex array back to OpenCV Mat vector
    UtlFFT::convertFFTWComplexToCVMatVector(f.data, cubeImage, cubeShape.width, cubeShape.height, cubeShape.depth);
    fftw_free(g.data);
}



// TODO refactor
std::vector<cv::Mat> DeconvolutionProcessor::postprocessChannel(ImageMetaData& metaData, std::vector<std::vector<cv::Mat>>& cubeImages){

    if(cubeImages.empty()){
        std::cerr << "[ERROR] No subimages processed" << std::endl;
    }

    std::vector<cv::Mat> result;
    ////////////////////////////////////////
    if(config.saveSubimages){

        std::vector<std::vector<cv::Mat>> tiles(cubeImages); // Kopie erstellen
        UtlGrid::cropCubePadding(tiles, cubes.cubePadding);

        double global_max_val_tile= 0.0;
        double global_min_val_tile = MAXFLOAT;
        for(auto cubeImage : tiles){
            // Global normalization of the merged volume
            for (const auto& slice : cubeImage) {
                cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
                double min_val, max_val;
                cv::minMaxLoc(slice, &min_val, &max_val);
                global_max_val_tile = std::max(global_max_val_tile, max_val);
                global_min_val_tile = std::min(global_min_val_tile, min_val);
            }
        }
        int num = 1;
        for(auto cubeImage : tiles){
        // Global normalization of the merged volume

        for (auto& slice : cubeImage) {
            slice.convertTo(slice, CV_32F, 1.0 / (global_max_val_tile - global_min_val_tile), -global_min_val_tile * (1 / (global_max_val_tile - global_min_val_tile)));  // Add epsilon to avoid division by zero
            cv::threshold(slice, slice, config.epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        }
            Hyperstack cubeImageHyperstack;
            cubeImageHyperstack.metaData = metaData;
            cubeImageHyperstack.metaData.imageWidth = cubeShape.width-(2*this->cubePadding);
            cubeImageHyperstack.metaData.imageLength = cubeShape.width-(2*this->cubePadding);
            cubeImageHyperstack.metaData.slices = cubeShape.width-(2*this->cubePadding);

            Image3D image3d;
            image3d.slices = cubeImage;
            Channel channel;
            channel.image = image3d;
            cubeImageHyperstack.channels.push_back(channel);
            cubeImageHyperstack.saveAsTifFile("../result/tiles/deconv_"+std::to_string(num)+".tif");
            num++;
        }
    }
    //////////////////////////////////////////////
    if(config.grid){
        //TODO no effect
        //UtlGrid::adjustCubeOverlap(cubeImages,this->cubePadding);

        UtlGrid::cropCubePadding(cubeImages, this->cubePadding);
        std::cout << "[STATUS] Merging Grid back to Image..." << std::endl;
        result = UtlGrid::mergeCubes(cubeImages, imageOriginalShape.width, imageOriginalShape.height, imageOriginalShape.depth, config.cubeSize);
        std::cout << "[INFO] Image size: " << result[0].rows << "x" << result[0].cols << "x" << result.size()<< std::endl;
    }else{
        result = cubeImages[0];
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

void DeconvolutionProcessor::preprocess(const Hyperstack& input, const std::vector<PSF>& psfs){
    if (!configured){
        std::__throw_runtime_error("Processor not configured");
    }

    setPSFShape((*psfs.front()));
    setImageOriginalShape(input.channels[0]); // i tihnk every channel should be same size
    setCubeShape(imageOriginalShape, config.grid, config.cubeSize, config.psfSafetyBorder);
    setupCubeArrangement();
    
    preprocessPSF(psfs);

    backend_->preprocess();
    // algorithm_->preprocess();

}

void DeconvolutionProcessor::configure(DeconvolutionConfig config) {
    // Store the config in base class
    this->config = config;
    DeconvolutionAlgorithmFactory fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = std::make_unique<DeconvolutionAlgorithm>(fact.create(config));

    this->backend_ = std::make_unique<CPUBackend>(); // TODO LK  to use the config and afactory
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
    const std::vector<PSF>& inputPSFs
    ) {

        cout << "[STATUS] Creating FFTW plans for PSFs..." << endl;
        
        fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psfOriginalShape.volume);
        fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psfOriginalShape.volume);
        fftw_plan forwardPSFPlan = fftw_plan_dft_3d(psfOriginalShape.depth, psfOriginalShape.height, psfOriginalShape.width, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);

        int counter = 0;
        for (int i = 0; i < inputPSFs.size(); i++) {
            counter ++;
            cout << "[STATUS] Performing Fourier Transform on PSF" << counter<< "..." << endl;
            
            // Convert PSF to FFTW complex format and execute FFT
            UtlFFT::convertCVMatVectorToFFTWComplex(inputPSFs[i].image.slices, h, psfOriginalShape.width, psfOriginalShape.height, psfOriginalShape.depth);
            fftw_execute_dft(forwardPSFPlan, h, h);

            cout << "[STATUS] Padding PSF" << counter << "..." << endl;
            
            // Pad PSF to cube size
            fftw_complex *temp_h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
            UtlFFT::padPSF(h, psfOriginalShape.width, psfOriginalShape.height, psfOriginalShape.depth, temp_h, cubeShape.width, cubeShape.height, cubeShape.depth);
            preparedpsfs[i] = temp_h;
        }

        initPSFMaps(inputPSFs);

        // Clean up temporary resources
        fftw_free(h);
        fftw_free(fftwPSFPlanMem);
        fftw_destroy_plan(forwardPSFPlan);
}

std::vector<fftw_complex*> DeconvolutionProcessor::selectPSFsForCube(int cubeIndex){
    int layerIndex = getLayerIndex(cubeIndex, cubes.cubesPerLayer);

    std::vector<fftw_complex*> psfs;
    psfs.emplace_back(layerPreparedPSFMap[layerIndex]);
    psfs.emplace_back(cubePreparedPSFMap[cubeIndex]);

    return psfs;
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
    
    // Clean up PSFs
    for (auto& psf : preparedpsfs) {
        if (psf) {
            fftw_free(psf);
            psf = nullptr;
        }
    }
    preparedpsfs.clear();
    layerPreparedPSFMap.clear();
    cubePreparedPSFMap.clear();
    
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

// find the psf of which the id corresopnds to the one in the config.layerPSFMap
// place the corresponding prepared psf into the layerPreparedPSFMap
// basically using the raw psf and raw ranges to map to the prepared psfs
void DeconvolutionProcessor::initPSFMaps(const std::vector<PSF>& psfs){
    for (const auto& [key, psfids] : config.layerPSFMap){
        for (auto psfid : psfids){
            for (int psfindex = 0; psfindex < psfs.size(); psfindex++){
                if (psfs[psfindex].ID == psfid){
                    layerPreparedPSFMap[key].push_back(preparedpsfs[psfindex]);
                }
            }
        }
    }
}


int DeconvolutionProcessor::getLayerIndex(int cubeIndex, int cubesPerLayer){
    return static_cast<int>(std::ceil(static_cast<double>((cubeIndex+1)) / cubesPerLayer));
}