#include "deconvolution/DeconvolutionProcessor.h"
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include <dlfcn.h>
#include "algorithms/TestAlgorithm.cpp"
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"

ImageMetaData globalmetadata;

void savecubeDebug(const std::vector<cv::Mat> cubeImage, const char* name, ImageMetaData metaData = globalmetadata){
    // double global_max_val_tile= 0.0;
    // double global_min_val_tile = MAXFLOAT;
    // // Global normalization of the merged volume
    // for (const auto& slice : cubeImage) {
    //     cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    //     double min_val, max_val;
    //     cv::minMaxLoc(slice, &min_val, &max_val);
    //     global_max_val_tile = std::max(global_max_val_tile, max_val);
    //     global_min_val_tile = std::min(global_min_val_tile, min_val);
    //     }
    
    // // Global normalization of the merged volume

    // for (auto& slice : cubeImage) {
    //     slice.convertTo(slice, CV_32F, 1.0 / (global_max_val_tile - global_min_val_tile), -global_min_val_tile * (1 / (global_max_val_tile - global_min_val_tile)));  // Add epsilon to avoid division by zero
    //     cv::threshold(slice, slice, 0.0005, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    // }
    Hyperstack cubeImageHyperstack;
    cubeImageHyperstack.metaData = metaData;
    cubeImageHyperstack.metaData.imageWidth = cubeImage[0].cols;
    cubeImageHyperstack.metaData.imageLength = cubeImage[0].rows;
    cubeImageHyperstack.metaData.slices = cubeImage.size();
    // cubeImageHyperstack.metaData.bitsPerSample = 32;

    Image3D image3d;
    image3d.slices = cubeImage;
    Channel channel;
    channel.image = image3d;
    cubeImageHyperstack.channels.push_back(channel);
    cubeImageHyperstack.saveAsTifFile("../result/"+std::string(name)+".tif");
}



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
            cv::Mat result(height, width, CV_32F); // Zero-initialize
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




void loadingBar(int i, int max){
    // Calculate progress
    int progress = (i * 100) / max;
    int barWidth = 50;
    int pos = (i * barWidth) / max;
    
    // Print progress bar
    std::cerr << "\rDeconvoluting Image [ ";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cerr << "=";
        else if (i == pos) std::cerr << ">";
        else std::cerr << " ";
    }
    std::cerr <<  "] " << std::setw(3) << progress << "% (" 
            << i << "/" << max << ")";
    std::cerr.flush();

}


//-----------------------------------------------------------

static ComplexData processDeconvolution(
    std::vector<ComplexData>& psfs_device,
    ComplexData& g_device,
    const RectangleShape& cubeShapePadded,
    const std::unique_ptr<DeconvolutionAlgorithm>& algorithm,
    std::shared_ptr<IDeconvolutionBackend> backend) {
    
    ComplexData f_device = backend->allocateMemoryOnDevice(cubeShapePadded);

    for (ComplexData& psf : psfs_device){
        algorithm->deconvolve(psf, g_device, f_device);
        backend->freeMemoryOnDevice(psf);

    }
    ComplexData f_host = backend->moveDataFromDevice(f_device);

    backend->freeMemoryOnDevice(g_device);
    backend->freeMemoryOnDevice(f_device);

    return f_host;
}




Hyperstack DeconvolutionProcessor::run(Hyperstack& input, const std::vector<PSF>& psfs){
    init(input, psfs);
    Image3D deconvolutedImage;

    for (auto& channel : input.channels){
        std::vector<std::vector<cv::Mat>> cubeImages = preprocessChannel(channel);
        parallelDeconvolution(cubeImages);
        std::vector<cv::Mat> deconvolutedChannel = postprocessChannel(input.metaData, cubeImages);

        std::cout << "[STATUS] Saving result of channel " << std::endl;
        deconvolutedImage.slices = deconvolutedChannel;
        input.channels[channel.id].image = deconvolutedImage;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return input;
}



void DeconvolutionProcessor::parallelDeconvolution(std::vector<std::vector<cv::Mat>>& cubeImages) {
    struct Task {
        int cubeIndex;
        std::future<ComplexData> future;
    };

    // Thread-safe queue for completed tasks
    std::queue<Task> completedTasks;
    std::mutex completedTasksMutex;
    std::condition_variable taskAvailable;
    std::atomic<int> tasksSubmitted(0);
    bool allTasksSubmitted = false;

    // Store running tasks
    std::vector<Task> runningTasks;
    std::mutex runningTasksMutex;
    std::atomic<int> processedCount;
    std::mutex progressMutex;
    // Create a thread pool for our tasks
   
        
    for (int cubeIndex = 0; cubeIndex < cubeImages.size(); ++cubeIndex) {
        // Launch deconvolution task (this already runs asynchronously)
        auto future = deconvolveSingleCube(cubeIndex, cubeImages[cubeIndex]);

        // Spawn a continuation task in the same thread pool
        threadPool->enqueue([&, cubeIndex, f = std::move(future)]() mutable {
            try {
                // Wait for the deconvolution result
                ComplexData deconvolvedImage = f.get();

                // Consume the result
                cubeImages[cubeIndex] = convertFFTWComplexToCVMatVector(deconvolvedImage);
                backend_->freeMemoryOnDevice(deconvolvedImage);

                // Optional: update progress
                {
                    std::lock_guard<std::mutex> lock(progressMutex);
                    loadingBar(++processedCount, cubeImages.size());
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(progressMutex);
                std::cerr << "Error in cube " << cubeIndex << ": " << e.what() << std::endl;
            }
        });
    }
    // Producer task
    // auto producerFuture = threadPool->enqueue([&]() {
    //     for (int cubeIndex = 0; cubeIndex < cubeImages.size(); cubeIndex++) {
    //         auto future = deconvolveSingleCube(cubeIndex, cubeImages[cubeIndex]);
            
    //         {
    //             std::lock_guard<std::mutex> lock(runningTasksMutex);
    //             runningTasks.emplace_back(Task{cubeIndex, std::move(future)});
    //         }
            
    //         tasksSubmitted++;
    //     }
    //     allTasksSubmitted = true;
    // });

    // // Monitor task
    // auto monitorFuture = threadPool->enqueue([&]() {
    //     while (!allTasksSubmitted || !runningTasks.empty()) {
    //         std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
    //         std::lock_guard<std::mutex> runLock(runningTasksMutex);
    //         auto it = runningTasks.begin();
            
    //         while (it != runningTasks.end()) {
    //             if (it->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    //                 // Task is completed, move to completed queue
    //                 {
    //                     std::lock_guard<std::mutex> queueLock(completedTasksMutex);
    //                     completedTasks.push(std::move(*it));
    //                 }
    //                 taskAvailable.notify_one();
    //                 it = runningTasks.erase(it);
    //             } else {
    //                 ++it;
    //             }
    //         }
    //     }
    // });

    // // Consumer task
    // auto consumerFuture = threadPool->enqueue([&]() {
    //     int processedCount = 0;
    //     while (processedCount < cubeImages.size()) {
    //         Task task;
    //         bool hasTask = false;

    //         // Wait for a task to be available
    //         {
    //             std::unique_lock<std::mutex> lock(completedTasksMutex);


    //             taskAvailable.wait(lock, [&] { 
    //                 return !completedTasks.empty() || 
    //                        (allTasksSubmitted && processedCount >= tasksSubmitted.load()); 
    //             });

    //             if (!completedTasks.empty()) {
    //                 task = std::move(completedTasks.front());
    //                 completedTasks.pop();
    //                 hasTask = true;
    //             }
    //         }

    //         if (hasTask) {
    //             loadingBar(processedCount++, cubeImages.size());
    //             ComplexData deconvolvedImage = task.future.get();
    //             cubeImages[task.cubeIndex] = convertFFTWComplexToCVMatVector(deconvolvedImage);
    //             backend_->freeMemoryOnDevice(deconvolvedImage);
                 
    //         }
    //     }
    // });

    // Wait for all tasks to complete
    // producerFuture.get();   // Wait for producer to finish
    // monitorFuture.get();    // Wait for monitor to finish  
    // consumerFuture.get();   // Wait for consumer to finish
}

std::future<ComplexData> DeconvolutionProcessor::deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage) {

    // copy operations
    ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage, cubeShapePadded);
    ComplexData g_device = backend_->moveDataToDevice(g_host);
    cpu_backend_->freeMemoryOnDevice(g_host);

    const std::vector<ComplexData> psfs = selectPSFsForCube(cubeIndex);
    std::vector<ComplexData> psfs_device;
    for ( const auto psf : psfs){
        psfs_device.emplace_back(backend_->moveDataToDevice(psf));
    }




    if (psfs.size() == 0) {
        std::cout << "[INFO] using no psf for cube " << cubeIndex << std::endl;
        backend_->freeMemoryOnDevice(g_host);
    }

    // auto deconvolutionFunc = [cubeShapePadded = this->cubeShapePadded]
    //     (std::vector<ComplexData>& psfs_device, 
    //     ComplexData& g_device,
    //     const std::unique_ptr<DeconvolutionAlgorithm>& algorithm,
    //     std::shared_ptr<IDeconvolutionBackend> backend) -> ComplexData {
    // };   
    std::future<ComplexData> future = threadPool->enqueue([psfs_device, g_device, workSize = cubeShapePadded, algorithm = std::move(algorithm_->clone()), backend = backend_]() mutable { 
        return processDeconvolution(psfs_device, g_device, workSize, algorithm, backend);
    });
        
    // std::future<ComplexData> future = deconvolutionBackendThreadManager->registerTask(
    //     psfs_device, g_device, deconvolutionFunc);

    return future;
 
}




// TODO refactor
std::vector<cv::Mat> DeconvolutionProcessor::postprocessChannel(ImageMetaData& metaData, const std::vector<std::vector<cv::Mat>>& cubeImages){

    if(cubeImages.empty()){
        std::cerr << "[ERROR] No subimages processed" << std::endl;
    }

    std::vector<cv::Mat> result;
    
    // legacy, dont know if it works
    ////////////////////////////////////////
    if(config.saveSubimages){

        std::vector<std::vector<cv::Mat>> tiles(cubeImages); // Kopie erstellen

        // Postprocessor::removePadding(tiles, config.psfSafetyBorder);

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
            cubeImageHyperstack.metaData.imageWidth = subimageShape.width;
            cubeImageHyperstack.metaData.imageLength = subimageShape.width;
            cubeImageHyperstack.metaData.slices = subimageShape.width;

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
        result = Postprocessor::mergeImage(
            cubeImages,
			subimageShape,
            imageOriginalShape,
            imageShapePadded,
            cubeShapePadded);
        std::cout << "[INFO] Image size: " << result[0].rows << "x" << result[0].cols << "x" << result.size()<< std::endl;
    }else{
        result = cubeImages[0];
    }
    Postprocessor::cropToOriginalSize(result, imageOriginalShape);
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

void DeconvolutionProcessor::init(const Hyperstack& input, const std::vector<PSF>& psfs){
    if (!configured){
        std::__throw_runtime_error("Processor not configured");
    }

    setImageOriginalShape(input.channels[0]);
    setPSFOriginalShape(psfs.front());
    setSubimageShape(imageOriginalShape, config.grid, config.subimageSize);
    addPaddingToShapes(psfOriginalShape);
    setupCubeArrangement();
   

    cpu_backend_->init(cubeShapePadded); // for psf preprocessing

    backend_->setWorkShape(cubeShapePadded); // not initialized, only used for copying, not fftw

    size_t maxThreads = 10; //TODO where should this be defined?
    threadPool = std::make_shared<ThreadPool>(maxThreads);
    size_t numberWorkerThreads = getNumberThreads(maxThreads);
    // deconvolutionBackendThreadManager = std::make_unique<DeconvolutionBackendThreadManager>(threadPool, numberWorkerThreads, algorithm_->clone(), backend_);
    preprocessPSF(psfs);

}



std::vector<std::vector<cv::Mat>> DeconvolutionProcessor::preprocessChannel(Channel& channel){
    Preprocessor::padToShape(channel.image.slices, imageShapePadded, config.borderType);

    if (config.grid){
        std::vector<std::vector<cv::Mat>> cubes = Preprocessor::splitImage(channel.image.slices,
            subimageShape,
            imageOriginalShape,
            imageShapePadded,
            cubeShapePadded);

        return cubes;
    }
    else{
        return std::vector<std::vector<cv::Mat>>{channel.image.slices};
    }
}

void DeconvolutionProcessor::configure(DeconvolutionConfig config) {
    this->config = config;
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = fact.create(config);
    this->backend_ = loadBackend(config.backenddeconv);
    this->cpu_backend_ = loadBackend("cpu");
    
    configured = true;
}


// because fftw3 and cufftw define the same api names they are loaded like this
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



void DeconvolutionProcessor::setPSFOriginalShape(const PSF& psf) {
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

void DeconvolutionProcessor::setSubimageShape(
    const RectangleShape& imageOriginalShape,
    bool configgrid,
    int subimageSize 
) {
    if (!configgrid) {
        // Non-grid processing - cube equals entire image
        subimageShape.width = imageOriginalShape.width;
        subimageShape.height = imageOriginalShape.height;
        subimageShape.depth = imageOriginalShape.depth;
        subimageShape.volume = subimageShape.width * subimageShape.height * subimageShape.depth;
        std::cout << "[INFO] Processing without grid" << std::endl;
    } else {
        subimageShape.width = std::min(subimageSize, imageOriginalShape.width);
        subimageShape.height = std::min(subimageSize, imageOriginalShape.height);
        subimageShape.depth = std::min(subimageSize, imageOriginalShape.depth);
        subimageShape.volume = subimageShape.width * subimageShape.height * subimageShape.depth;

    }

}

void DeconvolutionProcessor::addPaddingToShapes( const RectangleShape& padding){

    
    cubeShapePadded = subimageShape + padding + (-1);
    imageShapePadded = imageOriginalShape + padding + (-1);
}



void DeconvolutionProcessor::preprocessPSF(
    std::vector<PSF> inputPSFs
    ) {
        assert(cpu_backend_->isInitialized() + "backend not initialized");

        std::cout << "[STATUS] Preprocessing PSFs" << std::endl;
        for (int i = 0; i < inputPSFs.size(); i++) {
            
            Preprocessor::padToShape(inputPSFs[i].image.slices, cubeShapePadded, 0);

            ComplexData h = convertCVMatVectorToFFTWComplex(inputPSFs[i].image.slices, cubeShapePadded);          

            cpu_backend_->octantFourierShift(h);

            cpu_backend_->forwardFFT(h, h);

            preparedpsfs.push_back(h);

        }
        
        initPSFMaps(inputPSFs);

}

const std::vector<ComplexData> DeconvolutionProcessor::selectPSFsForCube(int cubeIndex) {
    int layerIndex = getLayerIndex(cubeIndex, cubes.cubesPerLayer);

    std::vector<ComplexData> psfs;
    
    std::vector<ComplexData> layerPSFs = layerPreparedPSFMap.get(layerIndex);
    if (!layerPSFs.empty()) {
        psfs.insert(psfs.end(), layerPSFs.begin(), layerPSFs.end());
    }
    
    std::vector<ComplexData> cubePSFs = cubePreparedPSFMap.get(cubeIndex);
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
        cubes.cubesPerX = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.width) / subimageShape.width));
        cubes.cubesPerY = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.height) / subimageShape.height));
        cubes.cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.depth) / subimageShape.depth));
        cubes.cubesPerLayer = cubes.cubesPerX * cubes.cubesPerY;
    }
    
    cubes.totalGridNum = cubes.cubesPerX * cubes.cubesPerY * cubes.cubesPerZ;
}



void DeconvolutionProcessor::cleanup() {
    // Clean up FFTW resources

    
    // Clean up PSFs
    for (auto& psf : preparedpsfs) {
        if (psf.data) {
            cpu_backend_->freeMemoryOnDevice(psf);
            psf.data = nullptr;
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
    if (config.layerPSFMap.empty() && config.cubePSFMap.empty()){
        layerPreparedPSFMap.addRange(0, -1, preparedpsfs[0]); // failsafe
        return;
    }
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

size_t DeconvolutionProcessor::getNumberThreads(size_t maxNumberThreads){
    assert(backend_->isInitialized() && "[ERROR] DeconvolutionBackendThreadManager: Backend must be initialized");


    size_t memoryPerCube = sizeof(complex) * cubeShapePadded.volume; 
    size_t backendMemoryMultiplier = backend_->getMemoryMultiplier();
    size_t algorithmMemoryMultiplier = algorithm_->getMemoryMultiplier();
    size_t memoryPerThread = memoryPerCube * (backendMemoryMultiplier + algorithmMemoryMultiplier); // * 2 for forward and backward fft plan, * 3 for inputimage, psf, outputimage (backend already includes 3 input copies)
    size_t availableMemory = backend_->getAvailableMemory();

    size_t possibleNumberThreads = availableMemory / memoryPerThread;
    size_t numberThreads = std::min(possibleNumberThreads, maxNumberThreads);

    if (numberThreads < 1){
        throw;
    }
    return numberThreads; 
}
// unused

// void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) {
//     try {
//         int width = input.size.width;
//         int height = input.size.height;
//         int depth = input.size.depth;
        
//         std::vector<cv::Mat> tempOutput;
//         for (int z = 0; z < depth; ++z) {
//             cv::Mat result(height, width, CV_32F);
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     int index = z * height * width + y * width + x;
//                     result.at<float>(y, x) = input.data[index][0];
//                 }
//             }
//             tempOutput.push_back(result);
//         }
//         output = tempOutput;
//     } catch (const std::exception& e) {
//         std::cerr << "[ERROR] Exception in convertFFTWComplexRealToCVMatVector: " << e.what() << std::endl;
//     }
// }

// void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) {
//     try {
//         int width = input.size.width;
//         int height = input.size.height;
//         int depth = input.size.depth;
        
//         std::vector<cv::Mat> tempOutput;
//         for (int z = 0; z < depth; ++z) {
//             cv::Mat result(height, width, CV_32F);
//             for (int y = 0; y < height; ++y) {
//                 for (int x = 0; x < width; ++x) {
//                     int index = z * height * width + y * width + x;
//                     result.at<float>(y, x) = input.data[index][1];
//                 }
//             }
//             tempOutput.push_back(result);
//         }
//         output = tempOutput;
//     } catch (const std::exception& e) {
//         std::cerr << "[ERROR] Exception in convertFFTWComplexImgToCVMatVector: " << e.what() << std::endl;
//     }
// }