#include "deconvolution/DeconvolutionProcessor.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <dlfcn.h>










//temp


// Conversion Functions
ComplexData convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, const RectangleShape& shape) {
    ComplexData result;

    try {
        result.size = shape;
        result.data = (complex*)malloc(sizeof(complex) * shape.volume);

        if (result.data == nullptr) {
            std::cerr << "[ERROR] FFTW malloc failed" << std::endl;
        }
    
         int width = shape.width;
        int height = shape.height;
        int depth = shape.depth;
        
        for (int z = 0; z < depth; ++z) {
            CV_Assert(input[z].type() == CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    result.data[z * height * width + y * width + x][0] = static_cast<double>(input[z].at<float>(y, x));
                    result.data[z * height * width + y * width + x][1] = 0.0;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertCVMatVectorToFFTWComplex: " << e.what() << std::endl;
    }
    return result;
}

std::vector<cv::Mat> convertFFTWComplexToCVMatVector(const ComplexData& input) {
    
    std::vector<cv::Mat> output;
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    double real_part = input.data[index][0];
                    double imag_part = input.data[index][1];
                    result.at<float>(y, x) = static_cast<float>(sqrt(real_part * real_part + imag_part * imag_part));
                }
            }
            output.push_back(result);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexToCVMatVector: " << e.what() << std::endl;
    }
    return output;
}

void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) {
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input.data[index][0];
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexRealToCVMatVector: " << e.what() << std::endl;
    }
}

void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) {
    try {
        int width = input.size.width;
        int height = input.size.height;
        int depth = input.size.depth;
        
        std::vector<cv::Mat> tempOutput;
        for (int z = 0; z < depth; ++z) {
            cv::Mat result(height, width, CV_32F);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    int index = z * height * width + y * width + x;
                    result.at<float>(y, x) = input.data[index][1];
                }
            }
            tempOutput.push_back(result);
        }
        output = tempOutput;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in convertFFTWComplexImgToCVMatVector: " << e.what() << std::endl;
    }
}







Hyperstack DeconvolutionProcessor::run(Hyperstack& input, const std::vector<PSF>& psfs){
    preprocess(input, psfs);
    Image3D deconvolutedImage;

    for (auto& channel : input.channels){
        std::vector<std::vector<cv::Mat>> cubeImages = preprocessChannel(channel);
        for (int cubeIndex = 0; cubeIndex < cubeImages.size(); cubeIndex++){
            std::cerr << "[STATUS] cubeIndex :" << cubeIndex << "\t out of " << cubeImages.size() << std::endl;
            deconvolveSingleCube(cubeIndex, cubeImages[cubeIndex]);
        }

        std::vector<cv::Mat> deconvolutedChannel = postprocessChannel(input.metaData, cubeImages);
        // Save the result
        // TODO clean up:
        std::cout << "[STATUS] Saving result of channel " << std::endl;
        deconvolutedImage.slices = deconvolutedChannel;
        input.channels[channel.id].image = deconvolutedImage;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return input;
}



void DeconvolutionProcessor::deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage){

    std::vector<complex*> psfs = selectPSFsForCube(cubeIndex);
    if (psfs.size() == 0){
        std::cout << "[INFO] using no psf for cube " + cubeIndex << std::endl;
    }
    for (auto psf: psfs){
        deconvolveSingleCubePSF(psf, cubeImage);
    }
}


void DeconvolutionProcessor::deconvolveSingleCubePSF(complex* psf, std::vector<cv::Mat>& cubeImage){

    // Observed image
    ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage, cubeShape);

    ComplexData H_device = backend_->moveDataToDevice({psf, cubeShape});
    ComplexData g_device = backend_->moveDataToDevice(g_host);
    ComplexData f_device = backend_->allocateMemoryOnDevice(cubeShape);

    algorithm_->deconvolve(H_device, g_device, f_device);
    ComplexData f_host = backend_->moveDataFromDevice(f_device);

    backend_->freeMemoryOnDevice(H_device);
    backend_->freeMemoryOnDevice(g_device);
    backend_->freeMemoryOnDevice(f_device);

    // Convert the result FFTW complex array back to OpenCV Mat vector
    cubeImage = convertFFTWComplexToCVMatVector(f_host);
    cpu_backend_->freeMemoryOnDevice(g_host);
    cpu_backend_->freeMemoryOnDevice(f_host);
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
        result = UtlGrid::mergeCubes(cubeImages, imageOriginalShape.width, imageOriginalShape.height, imageOriginalShape.depth, config.subimageSize);
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

    setPSFShape(psfs.front());
    setImageOriginalShape(input.channels[0]); // i tihnk every channel should be same size
    setCubeShape(imageOriginalShape, config.grid, config.subimageSize, config.psfSafetyBorder);
    setupCubeArrangement();
    

    backend_->init(cubeShape);
    cpu_backend_->init(psfOriginalShape); // for psf preprocessing
    preprocessPSF(psfs);

    // algorithm_->init();

}

void DeconvolutionProcessor::configure(DeconvolutionConfig config) {
    this->config = config;
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = fact.create(config);

    this->backend_ = loadBackend(config.backend);
    this->cpu_backend_ = loadBackend("cpu");
    std::shared_ptr<IDeconvolutionBackend> backendalgo = this->backend_->clone();
    backendalgo->init(cubeShape);
    this->algorithm_->setBackend(backendalgo);

    configured = true;
}

std::shared_ptr<IDeconvolutionBackend> DeconvolutionProcessor::loadBackend(const std::string& backendName){

    std::string libpath = std::string("backends/") + backendName + "/lib" + backendName + "_backend.so";
    void* handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        std::cerr << "dlopen failed: " << (err ? err : "unknown error") << std::endl;
        throw std::runtime_error(err ? err : "dlopen failed");    }

    using create_fn = IDeconvolutionBackend*();
    auto create_backend = reinterpret_cast<create_fn*>(dlsym(handle, "create_backend"));
    if (!create_backend) {
        throw std::runtime_error(dlerror());
    }
    return std::shared_ptr<IDeconvolutionBackend>(create_backend());
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
    int configsubimageSize,
    int configpsfSafetyBorder
) {
    if (!configgrid) {
        // Non-grid processing - cube equals entire image
        cubeShape.width = imageOriginalShape.width;
        cubeShape.height = imageOriginalShape.height;
        cubeShape.depth = imageOriginalShape.depth;
        cubeShape.volume = imageOriginalShape.volume;
    } else {
        if (configsubimageSize < 1){
            std::__throw_runtime_error("Can't use grid with a grid size of smaller than 0");
        }
        // Grid processing - calculate cube size with padding
        // This logic mirrors BaseDeconvolutionAlgorithm::preprocess
        cubePadding = configpsfSafetyBorder;
        
        // Adjust padding based on PSF size (from original logic)
        int safetyBorderPsfWidth = psfOriginalShape.width + (2 * configpsfSafetyBorder);
        
        if (safetyBorderPsfWidth < configsubimageSize) {
            cubePadding = 10;
        }
        if (configsubimageSize + 2 * cubePadding < safetyBorderPsfWidth) {
            cubePadding = (safetyBorderPsfWidth - configsubimageSize) / 2;
        }
        
        cubeShape.width = configsubimageSize + (2 * cubePadding);
        cubeShape.height = configsubimageSize + (2 * cubePadding);
        cubeShape.depth = configsubimageSize + (2 * cubePadding);
        cubeShape.volume = cubeShape.width * cubeShape.height * cubeShape.depth;
    }
}

std::vector<std::vector<cv::Mat>> DeconvolutionProcessor::preprocessChannel(Channel& channel){
    if (config.grid){
        int imagepadding= imageOriginalShape.width/2;
        UtlGrid::extendImage(channel.image.slices, imagepadding, config.borderType);
        return UtlGrid::splitWithCubePadding(channel.image.slices, config.subimageSize, imagepadding, cubePadding);}
    else{
        return std::vector<std::vector<cv::Mat>>{channel.image.slices};
    }
}

void DeconvolutionProcessor::preprocessPSF(
    const std::vector<PSF>& inputPSFs
    ) {
        assert(backend_->isInitialized() + "backend not initialized");
        std::cout << "[STATUS] Creating FFTW plans for PSFs..." << std::endl;
        
       
        for (int i = 0; i < inputPSFs.size(); i++) {
            std::cout << "[STATUS] Performing Fourier Transform on PSF" << i << "..." << std::endl;
            
            // Convert PSF to FFTW complex format and execute FFT
            ComplexData h = convertCVMatVectorToFFTWComplex(inputPSFs[i].image.slices, psfOriginalShape);
            cpu_backend_->forwardFFT(h, h);

            std::cout << "[STATUS] Padding PSF" << i << "..." << std::endl;
            
            // Pad PSF to cube size
            ComplexData temp_h = cpu_backend_->allocateMemoryOnDevice(cubeShape);
            padPSF(h, temp_h);
            preparedpsfs.push_back(temp_h.data);
            cpu_backend_->freeMemoryOnDevice(h);
        }
        initPSFMaps(inputPSFs);

        // Clean up temporary resources
}

std::vector<complex*> DeconvolutionProcessor::selectPSFsForCube(int cubeIndex) {
    int layerIndex = getLayerIndex(cubeIndex, cubes.cubesPerLayer);

    std::vector<complex*> psfs;
    
    std::vector<complex*> layerPSFs = layerPreparedPSFMap.get(layerIndex);
    if (!layerPSFs.empty()) {
        psfs.insert(psfs.end(), layerPSFs.begin(), layerPSFs.end());
    }
    
    std::vector<complex*> cubePSFs = cubePreparedPSFMap.get(cubeIndex);
    if (!cubePSFs.empty()) {
        psfs.insert(psfs.end(), cubePSFs.begin(), cubePSFs.end());
    }

    return psfs;
}



void DeconvolutionProcessor::setupCubeArrangement() {
    if (!config.grid) {
        cubes.cubesPerX = 1;
        cubes.cubesPerY = 1;
        cubes.cubesPerZ = 1;
        cubes.cubesPerLayer = 1;
    } else {
        cubes.cubesPerX = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.width) / config.subimageSize));
        cubes.cubesPerY = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.height) / config.subimageSize));
        cubes.cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.depth) / config.subimageSize));
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

    
    // Clean up PSFs
    for (auto& psf : preparedpsfs) {
        if (psf) {
            ComplexData temp{psf, cubeShape};
            backend_->freeMemoryOnDevice(temp);
            psf = nullptr;
        }
    }
    preparedpsfs.clear();
    layerPreparedPSFMap.clear();
    cubePreparedPSFMap.clear();
    
}



// find the psf of which the id corresopnds to the one in the config.layerPSFMap
// place the corresponding prepared psf into the layerPreparedPSFMap
// basically using the raw psf and raw ranges to map to the prepared psfs
// refactor
void DeconvolutionProcessor::initPSFMaps(const std::vector<PSF>& psfs){
    for (const auto& range : config.layerPSFMap) {
        for (int psfindex = 0; psfindex < psfs.size(); psfindex++) {
            std::string ID = psfs[psfindex].ID;
            for (const std::string& mapID : range.get()){
                if (ID == mapID){
                    layerPreparedPSFMap.addRange(range.start, range.end, preparedpsfs[psfindex]);
                }
            }
        }
    }
}


int DeconvolutionProcessor::getLayerIndex(int cubeIndex, int cubesPerLayer){
    return static_cast<int>(std::ceil(static_cast<double>((cubeIndex)) / cubesPerLayer));
}


void DeconvolutionProcessor::padPSF(const ComplexData& psf, ComplexData& padded_psf) {
    try {
        // Create temporary copy for shifting
        ComplexData temp_psf = cpu_backend_->copyData(psf);
        cpu_backend_->octantFourierShift(temp_psf);
        
        // Zero out padded PSF
        for (int i = 0; i < padded_psf.size.volume; ++i) {
            padded_psf.data[i][0] = 0.0;
            padded_psf.data[i][1] = 0.0;
        }

        if (psf.size.depth > padded_psf.size.depth) {
            std::cerr << "[ERROR] PSF has more layers than target size" << std::endl;
        }

        int x_offset = (padded_psf.size.width - psf.size.width) / 2;
        int y_offset = (padded_psf.size.height - psf.size.height) / 2;
        int z_offset = (padded_psf.size.depth - psf.size.depth) / 2;

        for (int z = 0; z < psf.size.depth; ++z) {
            for (int y = 0; y < psf.size.height; ++y) {
                for (int x = 0; x < psf.size.width; ++x) {
                    int padded_index = ((z + z_offset) * padded_psf.size.height + (y + y_offset)) * padded_psf.size.width + (x + x_offset);
                    int psf_index = (z * psf.size.height + y) * psf.size.width + x;

                    padded_psf.data[padded_index][0] = temp_psf.data[psf_index][0];
                    padded_psf.data[padded_index][1] = temp_psf.data[psf_index][1];
                }
            }
        }
        cpu_backend_->octantFourierShift(padded_psf);
        
        // Clean up temporary data
        cpu_backend_->freeMemoryOnDevice(temp_psf);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception in padPSF: " << e.what() << std::endl;
    }
}




// std::shared_ptr<IDeconvolutionBackend> getThreadLocalBackend() {
//     if (!thread_backend_) {
//         // Create and initialize backend for this thread
//         thread_backend_ = backend_->clone();
//         thread_backend_->preprocess(); // Initialize memory pools, FFTW plans, etc.
//     }
//     return thread_backend_;
// }


// // Define the thread_local static member
// thread_local std::shared_ptr<IDeconvolutionBackend> DeconvolutionProcessorParallel::thread_backend_;

// void DeconvolutionProcessorParallel::deconvolveSingleCubePSF(fftw_complex* psf, std::vector<cv::Mat>& cubeImage){
//     ComplexData H = {psf, cubeShape};
//     fftw_complex* tempg = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeShape.volume);
//     ComplexData g = {tempg, cubeShape};

//     if (!(UtlImage::isValidForFloat(g.data, cubeShape.volume))) {
//         std::cout << "[WARNING] Value fftwPlanMem fftwcomplex(double) is smaller than float" << std::endl;
//     }
//     UtlFFT::convertCVMatVectorToFFTWComplex(cubeImage, g.data, cubeShape.width, cubeShape.height, cubeShape.depth);

//     // Get thread-local backend (created once per thread)
//     auto thread_backend = getThreadLocalBackend();
    
//     ComplexData H_device = thread_backend->moveDataToDevice(H);
//     ComplexData g_device = thread_backend->moveDataToDevice(g);
//     ComplexData f_device = thread_backend->allocateMemoryOnDevice(cubeShape);

//     // Clone algorithm (lightweight, just copies parameters)
//     std::unique_ptr<DeconvolutionAlgorithm> algorithmClone = algorithm_->clone();
//     algorithmClone->setBackend(thread_backend);

//     algorithmClone->deconvolve(H_device, g_device, f_device);
    
//     ComplexData f = thread_backend->moveDataFromDevice(f_device);
//     thread_backend->freeMemoryOnDevice(H_device);
//     thread_backend->freeMemoryOnDevice(g_device);
//     thread_backend->freeMemoryOnDevice(f_device);

//     UtlFFT::convertFFTWComplexToCVMatVector(f.data, cubeImage, cubeShape.width, cubeShape.height, cubeShape.depth);
//     fftw_free(g.data);   
// }

