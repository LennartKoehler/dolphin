#include "deconvolution/DeconvolutionProcessor.h"
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include "algorithms/TestAlgorithm.cpp"
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"

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






Hyperstack DeconvolutionProcessor::run(Hyperstack& input, const std::vector<PSF>& psfs){
    init(input, psfs);
    Image3D deconvolutedImage;
    std::cout << "[STATUS] Starting deconvolution" << std::endl;

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
    std::vector<std::future<void>> runningTasks;
    std::atomic<int> processedCount(0);
    std::mutex loadingBarMutex;

    std::mutex memoryMutex;
    std::condition_variable memoryFull;
    bool memoryAvailable = true;

    // std::mutex queueMutex;
    // std::condition_variable queueFull;
    // const int maxNumberWorkerThreads = numberThreads;

    std::atomic<int> numberCubes(cubeImages.size());

    for (int cubeIndex = 0; cubeIndex < numberCubes; ++cubeIndex) {
        // Wait if too many tasks are running LK dont need because there is basically no memory allocation per task, almost all passed by reference
        // {
        //     std::unique_lock<std::mutex> lock(queueMutex);
        //     queueFull.wait(lock, [&runningTasks, maxNumberWorkerThreads] {
        //         // Clean up any completed tasks
        //         runningTasks.erase(
        //             std::remove_if(
        //                 runningTasks.begin(), runningTasks.end(),
        //                 [](std::future<void>& f) {
        //                     return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        //                 }),
        //             runningTasks.end()
        //         );
        //         return static_cast<int>(runningTasks.size()) < maxNumberWorkerThreads;
        //     });
        // }

        // Launch deconvolution asynchronously, no real memory copies as all is passed by reference or pointer
        std::vector<const ComplexData*> psfs = selectPSFsForCube(cubeIndex);

        std::function<void()> singleDeconvHandled = [&, cubeIndex, psfs]() {
            try{
                deconvolveSingleCube(
                    backend_,
                    algorithm_->clone(),
                    cubeImages[cubeIndex],
                    cubeShapePadded,
                    psfs);
                {
                    std::unique_lock<std::mutex> loack(loadingBarMutex);
                    loadingBar(++processedCount, numberCubes);

                }
                {
                    std::unique_lock<std::mutex> lock(memoryMutex);
                    memoryAvailable = true;
                }
                // queueFull.notify_one();
                memoryFull.notify_all(); // Signal that one thread is done and new memory might be available
            }
            catch (const dolphin::backend::MemoryException& e) {
                std::cerr << "[ERROR] Memory exception in cube " << cubeIndex << ": " 
                         << e.getDetailedMessage() << " Reqeueing task" << std::endl;
                
                {
                    std::unique_lock<std::mutex> lock(memoryMutex);
                    memoryAvailable = false;
                    memoryFull.wait(lock, [&memoryAvailable]{
                        return memoryAvailable;
                    });
                }
                runningTasks.push_back(threadPool->enqueue(singleDeconvHandled));
            }
            catch (const dolphin::backend::BackendException& e) {
                std::cerr << "[ERROR] Backend exception in cube " << cubeIndex << ": " 
                         << e.getDetailedMessage() << std::endl;
                throw;
            }
            catch (const std::exception& e) {
                std::cerr << "[ERROR] General exception in cube " << cubeIndex << ": " 
                         << e.what() << std::endl;
                throw;
            }
        };

        runningTasks.push_back(threadPool->enqueue(singleDeconvHandled));
    }

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();
}

void DeconvolutionProcessor::deconvolveSingleCube(
    std::shared_ptr<IBackend> backend,
    std::unique_ptr<DeconvolutionAlgorithm> algorithm,
    std::vector<cv::Mat>& cubeImage,
    const RectangleShape& workShape,
    const std::vector<const ComplexData*>& psfs_device) {

    ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
    
    try{
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage, workShape);
        ComplexData g_device = backend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);
        ComplexData f_device = backend->getMemoryManager().allocateMemoryOnDevice(cubeShapePadded);

        for (const auto* psf_device : psfs_device){
            algorithm->deconvolve(*psf_device, g_device, f_device);

        }

        f_host = backend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
    }
    catch(...){
        throw; // dont overwrite image if exception
    }
    cubeImage = convertFFTWComplexToCVMatVector(f_host);
}




std::vector<cv::Mat> DeconvolutionProcessor::postprocessChannel(ImageMetaData& metaData, const std::vector<std::vector<cv::Mat>>& cubeImages){

    if(cubeImages.empty()){
        std::cerr << "[ERROR] No subimages processed" << std::endl;
    }

    std::vector<cv::Mat> result;

    if (imageShapePadded != cubeShapePadded){
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
    size_t memoryPerCube = getMemoryPerCube(numberThreads);

    setImageOriginalShape(input.channels[0]);
    setPSFOriginalShape(psfs.front()); // TODO for multiple psfs this would have to be the largest psf to get the proper padding
    setWorkShapes(imageOriginalShape, psfOriginalShape, memoryPerCube);
    setupCubeArrangement();
   
    backend_->mutableDeconvManager().init(cubeShapePadded);
    algorithm_->setBackend(backend_);

    preprocessPSF(psfs);

}



std::vector<std::vector<cv::Mat>> DeconvolutionProcessor::preprocessChannel(Channel& channel){
    
    Preprocessor::padToShape(channel.image.slices, imageShapePadded, config.borderType);

    if (imageShapePadded != cubeShapePadded){
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

void DeconvolutionProcessor::configure(const DeconvolutionConfig config) {
    this->config = config;
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = fact.create(config);

    BackendFactory& bf = BackendFactory::getInstance();

    this->backend_ = bf.create(config.backenddeconv);
    this->cpuMemoryManager= bf.createMemManager(config.backenddeconv);

    numberThreads = config.backenddeconv == "cuda" ? 1 : config.nThreads; // TODO change
    threadPool = std::make_shared<ThreadPool>(numberThreads);

    configured = true;
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

void DeconvolutionProcessor::setWorkShapes(
    const RectangleShape& imageOriginalShape,
    const RectangleShape& padding,
    size_t memoryPerCube
) {
    // this function determines the shape into which the input image is cut
    // current strategy is to only slice the largest dimension while leaving the smaller two dimensions the same shape
    // the constraints are that all threads should be used but it all needs to fit on the available memory
    // due to padding it is most optimal (smallest 3dshape) to have all dimensions the same size as this reduces the increase in volume caused by padding
    // but this is difficult as we want to have all threads have a similar workload aswell as reducing the overhead of each thread having to read/write more than once
    // ideally we have number of cubes (all dim same length) of equal size equal to number of threads
    // there are different strategies to split the original image but this is just what I went with
    // it is useful to keep all cubes the same dimensionality as the psfs then only need to be transformed once into that shape and the fftw plans can be reused

    imageShapePadded = imageOriginalShape + padding;

    size_t maxMemCubeVolume = memoryPerCube / sizeof(complex); // cut into pieces so that they still fit on memory

    subimageShape = imageOriginalShape;
    std::array<int*, 3> sortedDimensionsSubimage = subimageShape.getDimensionsAscending();
    size_t maxThreadcubeLargestDim = (*sortedDimensionsSubimage[2] + numberThreads -1) / numberThreads; // ceiling divide

    RectangleShape tempPadded = imageOriginalShape + padding;
    std::array<int*, 3> sortedDimensionsPadded = tempPadded.getDimensionsAscending();
    size_t maxMemCubeLargestDim = maxMemCubeVolume / (*sortedDimensionsPadded[0] * *sortedDimensionsPadded[1]);

    *sortedDimensionsSubimage[2] = std::min(maxMemCubeLargestDim, maxThreadcubeLargestDim);
    assert(*sortedDimensionsSubimage[2] != 0 && "[ERROR] setWorkShapes: not enough memory to fit a single slice of the image");


    subimageShape.updateVolume();
    cubeShapePadded = subimageShape + padding;
    // TODO could also start halfing the other dimension until it fits
    // idea: always half the largest dimension until:
    //      number of cubes = number of threads && size of cubes fits on memory

    // size_t cubeVolume = std::min(memCubeVolume, threadCubeVolume);
    // double scaleFactor = std::cbrt( static_cast<double>(cubeVolume) / imageShapePadded.volume);
    // subimageShape = imageOriginalShape * scaleFactor;
    // subimageShape.clamp(imageOriginalShape);

    // cubeShapePadded = subimageShape + padding;

}




void DeconvolutionProcessor::preprocessPSF(
    std::vector<PSF> inputPSFs
    ) {
        assert(backend_->getDeconvManager().plansInitialized() + "backend not initialized");

        std::cout << "[STATUS] Preprocessing PSFs" << std::endl;
        for (int i = 0; i < inputPSFs.size(); i++) {
            
            Preprocessor::padToShape(inputPSFs[i].image.slices, cubeShapePadded, 0);
            ComplexData h = convertCVMatVectorToFFTWComplex(inputPSFs[i].image.slices, cubeShapePadded);
            ComplexData h_device = backend_->getMemoryManager().copyDataToDevice(h);
            backend_->getDeconvManager().octantFourierShift(h_device);
            backend_->getDeconvManager().forwardFFT(h_device, h_device);
            preparedpsfs.push_back(std::move(h_device));

        }
        
        initPSFMaps(inputPSFs);

}

const std::vector<const ComplexData*> DeconvolutionProcessor::selectPSFsForCube(int cubeIndex) {
    int layerIndex = getLayerIndex(cubeIndex, cubes.cubesPerLayer);

    std::vector<const ComplexData*> psfs;
    
    std::vector<const ComplexData*> layerPSFs = layerPreparedPSFMap.getPointers(layerIndex);
    psfs.insert(psfs.end(), layerPSFs.begin(), layerPSFs.end());
    
    std::vector<const ComplexData*> cubePSFs = cubePreparedPSFMap.getPointers(cubeIndex);
    psfs.insert(psfs.end(), cubePSFs.begin(), cubePSFs.end());

    return psfs;

}



void DeconvolutionProcessor::setupCubeArrangement() {

    cubes.cubesPerX = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.width) / subimageShape.width));
    cubes.cubesPerY = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.height) / subimageShape.height));
    cubes.cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.depth) / subimageShape.depth));
    cubes.cubesPerLayer = cubes.cubesPerX * cubes.cubesPerY;

    cubes.totalGridNum = cubes.cubesPerX * cubes.cubesPerY * cubes.cubesPerZ;
}



void DeconvolutionProcessor::cleanup() {
    // Clean up FFTW resources

    
    // Clean up PSFs
    for (auto& psf : preparedpsfs) {
        if (psf.data) {
            cpuMemoryManager->freeMemoryOnDevice(psf);
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
                    ComplexData& psfs_host = preparedpsfs[psfindex];
                    layerPreparedPSFMap.addRange(range.start, range.end, std::move(psfs_host));
                }
            }
        }
    }
}


int DeconvolutionProcessor::getLayerIndex(int cubeIndex, int cubesPerLayer){
    return static_cast<int>(std::ceil(static_cast<double>((cubeIndex)) / cubesPerLayer));
}

size_t DeconvolutionProcessor::getMemoryPerCube(size_t maxNumberThreads){

    size_t algorithmMemoryMultiplier = algorithm_->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
    size_t memoryBuffer = 1e9; // TESTVALUE
    size_t availableMemory = backend_->mutableMemoryManager().getAvailableMemory() - memoryBuffer;
    size_t memoryPerThread = availableMemory / maxNumberThreads;
    size_t memoryPerCube = memoryPerThread / algorithmMemoryMultiplier;
    return memoryPerCube;

    
}



// Conversion Functions
ComplexData DeconvolutionProcessor::convertCVMatVectorToFFTWComplex(
    const std::vector<cv::Mat>& input, 
    const RectangleShape& shape) {

    ComplexData result = cpuMemoryManager->allocateMemoryOnDevice(shape);

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

    return result;
}

std::vector<cv::Mat> DeconvolutionProcessor::convertFFTWComplexToCVMatVector(const ComplexData& input) {
    
    std::vector<cv::Mat> output;
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

    return output;
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