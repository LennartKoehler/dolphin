/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
#include "deconvolution/ImageMap.h"







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



DeconvolutionProcessor::DeconvolutionProcessor(){
    
    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
    const RectangleShape shape,
    std::shared_ptr<PSF> inputPSF,
    std::shared_ptr<IBackend> backend
    ) -> ComplexData* {
       
        Preprocessor::padToShape(inputPSF->image.slices, shape, 0);
        ComplexData h = convertCVMatVectorToFFTWComplex(inputPSF->image.slices, shape);
        ComplexData h_device = backend->getMemoryManager().copyDataToDevice(h);
        backend->getDeconvManager().octantFourierShift(h_device);
        backend->getDeconvManager().forwardFFT(h_device, h_device);
        return new ComplexData(std::move(h_device));
    };
    psfPreprocessor.setPreprocessingFunction(psfPreprocessFunction);
}


Hyperstack DeconvolutionProcessor::run(const Hyperstack& image, const std::vector<PSF>& inputPSFS, DeconvolutionStrategy& strategy){
    setImageOriginalShape(image.channels[0]);

    Hyperstack inputCopy(image); // copy to not edit in place, the inputcopy is used to be padded and then used for reading
    Hyperstack result(image); // to get the correct shape so i can later paste the result in the corresponding position
    // i need both because cubes might overlap, so i cant write in place


    ImageMap<std::vector<std::shared_ptr<PSF>>> psfStrategy = strategy.getStrategy(
        inputPSFS,
		imageOriginalShape,
		image.channels.size(),
		config,
		backend_,
		algorithm_);
    
    psfStrategy.sortByDimensions(); // this sorts by rectangleShape, so that fft plans can be destroyed when new dimension is needed, because the old dimension wont be needed as its sorted
    init(image, psfStrategy);


    std::cout << "[STATUS] Starting deconvolution" << std::endl;

    for (int i = 0; i < image.channels.size(); i++){
        
        RectangleShape paddingShift = Preprocessor::padToShape(inputCopy.channels[i].image.slices, imageShapePadded, config.borderType); // pad to largest psf, should be the easiest
        parallelDeconvolution(inputCopy.channels[i].image.slices, result.channels[i].image.slices, psfStrategy, paddingShift);
        postprocessChannel(result.channels[i].image.slices);

        std::cout << "[STATUS] Saving result of channel " << std::endl;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return result;
}



// TODO this function is a mess
void DeconvolutionProcessor::parallelDeconvolution(
        const std::vector<cv::Mat>& inputImagePadded,
        std::vector<cv::Mat>& outputImage,
        const ImageMap<std::vector<std::shared_ptr<PSF>>>& psfMap,
        const RectangleShape& imagePaddingShift) {

    std::vector<std::future<void>> runningTasks;
    std::atomic<int> processedCount(0);
    std::mutex loadingBarMutex;

    std::mutex writerMutex;

    std::mutex memoryMutex;
    std::condition_variable memoryFull;
    bool memoryAvailable = true;

    std::atomic<int> numberCubes(psfMap.size());

    for (int cubeIndex = 0; cubeIndex < numberCubes; ++cubeIndex) {
        // if (cubeIndex == 50){
        //     break;//TESTVALUE
        // }

        
        // this task is run by the readwriterPool, which will then enqueue the work itself into the workerpool
        // since there are more readwriters than workers, the workers should always be occupied with work, 
        auto task = [this, cubeIndex, 
                     psfMap = std::move(psfMap),
                     inputImagePadded = std::move(inputImagePadded),
                     imagePaddingShift = std::move(imagePaddingShift),
                     &loadingBarMutex, &writerMutex, &memoryMutex, &memoryFull, &memoryAvailable,
                     &processedCount, &numberCubes, &outputImage, &runningTasks]() mutable {
            try{

                const BoxEntryPair<std::vector<std::shared_ptr<PSF>>> psfs = psfMap.get(cubeIndex);

                BoxCoord srcBox = psfs.box;
                RectangleShape padding = getCubePadding(psfs.entry);
                RectangleShape workShape = srcBox.dimensions + padding * 2;
                std::vector<cv::Mat> cubeImage = getCubeImage(inputImagePadded, srcBox, padding, imagePaddingShift);


                deconvolveSingleCube(
                    backend_,
                    algorithm_,
                    cubeImage,
                    workShape,
                    psfs);


                // Wait for processing to complete, then insert the result
                
                {
                    std::unique_lock<std::mutex> lock(writerMutex);
                    Postprocessor::insertCubeInImage(cubeImage, outputImage, srcBox, padding);
                }

                {
                    std::unique_lock<std::mutex> lock(memoryMutex);
                    memoryAvailable = true;
                }
                {
                    std::unique_lock<std::mutex> lock(loadingBarMutex);
                    loadingBar(++processedCount, numberCubes);
                    
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
                // runningTasks.push_back(workerPool->enqueue(task));
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

        runningTasks.push_back(readwriterPool->enqueue(task));
    }

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();
}

void DeconvolutionProcessor::deconvolveSingleCube(
    std::shared_ptr<IBackend> prototypebackend,
    std::shared_ptr<DeconvolutionAlgorithm> prototypealgorithm,
    std::vector<cv::Mat>& cubeImage,
    const RectangleShape& workShape,
    const BoxEntryPair<std::vector<std::shared_ptr<PSF>>>& psfs_host) {

    ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};

    try{

        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage, workShape);
        ComplexData g_device = prototypebackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);
        ComplexData f_device = prototypebackend->getMemoryManager().allocateMemoryOnDevice(workShape);


        // on workerThread
        std::future<void> resultDone = workerPool->enqueue([
            this,
            &prototypebackend,
            &prototypealgorithm,
            &psfs_host,
            &workShape,
            &g_device,
            &f_device
        ](){
            std::shared_ptr<IBackend> threadbackend = prototypebackend->onNewThread();
            
            std::unique_ptr<DeconvolutionAlgorithm> algorithm = prototypealgorithm->clone();
            algorithm->setBackend(threadbackend);

            // Add debug logging to validate backend object
            if (!threadbackend) {
                std::cerr << "[CRITICAL ERROR] Thread backend is null!" << std::endl;
                throw std::runtime_error("Thread backend is null");
            }
            std::vector<const ComplexData*> preprocessedPSFs;
            for (auto& psf : psfs_host.entry){
                preprocessedPSFs.emplace_back(psfPreprocessor.getPreprocessedPSF(workShape, psf, threadbackend));
            }
            
    


            for (const auto* psf_device : preprocessedPSFs){
                algorithm->deconvolve(*psf_device, g_device, f_device);
            }
            threadbackend->sync();
            threadbackend->releaseBackend(); // TODO do i run this here
        });
        resultDone.get();

        // on host
        f_host = prototypebackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
    }
    catch(...){
        throw; // dont overwrite image if exception
    }
    cubeImage = convertFFTWComplexToCVMatVector(f_host);
}


// TODOmove to preprocessor
std::vector<cv::Mat> DeconvolutionProcessor::getCubeImage(const std::vector<cv::Mat>& paddedImage, const BoxCoord& coords, const RectangleShape& cubePadding, const RectangleShape& imagePaddingShift){
    // remember that coords is for the original image, not the padded Image, so it has to be shifted, so the paddingShift member variable can be used
    assert(coords.x >= 0 && coords.y >= 0 && coords.z >= 0 && "getCubeImage source Box coordinates must be non-negative");
    assert(coords.x + coords.dimensions.width <= paddedImage[0].cols && "getCubeImage cube extends beyond image width");
    assert(coords.y + coords.dimensions.height <= paddedImage[0].rows && "getCubeImage cube extends beyond image height");
    assert(coords.z + coords.dimensions.depth <= static_cast<int>(paddedImage.size()) && "getCubeImage cube extends beyond image depth");
    assert(imagePaddingShift >= cubePadding && "getCubeImage too little image padding");

    std::vector<cv::Mat> cube;
    cube.reserve(coords.dimensions.depth + 2 * cubePadding.depth);
    
    // Triple nested loop to iterate through all cube positions with padding
    int xImage = coords.x + imagePaddingShift.width - cubePadding.width;
    int yImage = coords.y + imagePaddingShift.height - cubePadding.height;


    for (int zCube = 0; zCube < coords.dimensions.depth + 2 * cubePadding.depth; ++zCube) {
        

        // Define the ROI in the source image
        cv::Rect imageROI(xImage, yImage, coords.dimensions.width + 2 * cubePadding.width, coords.dimensions.height + 2 * cubePadding.height);
        int zImage = zCube + coords.z + imagePaddingShift.depth - cubePadding.depth;
        cv::Mat slice(coords.dimensions.height + 2 * cubePadding.height, coords.dimensions.width + 2 * cubePadding.width, CV_32F, cv::Scalar(0));

        // Copy from the source image ROI to the entire slice
        paddedImage[zImage](imageROI).copyTo(slice);

        cube.push_back(std::move(slice));
    }
    
    return cube;
}

RectangleShape DeconvolutionProcessor::getCubePadding(const std::vector<std::shared_ptr<PSF>> psfs){
    RectangleShape maxPsfShape{0, 0, 0};
    
    // Find the largest PSF dimensions
    for (const auto& psf : psfs) {
        int psfWidth = psf->image.slices[0].cols;
        int psfHeight = psf->image.slices[0].rows;
        int psfDepth = static_cast<int>(psf->image.slices.size());
        
        maxPsfShape.width = std::max(maxPsfShape.width, psfWidth);
        maxPsfShape.height = std::max(maxPsfShape.height, psfHeight);
        maxPsfShape.depth = std::max(maxPsfShape.depth, psfDepth);
    }
    
    // Return half of the largest PSF dimensions
    return RectangleShape(
        static_cast<int>(maxPsfShape.width / 2),
        static_cast<int>(maxPsfShape.height / 2),
        static_cast<int>(maxPsfShape.depth / 2)
    );
}


void DeconvolutionProcessor::postprocessChannel(std::vector<cv::Mat>& image){


    Postprocessor::cropToOriginalSize(image, imageOriginalShape);
    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    for (const auto& slice : image) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }

    for (auto& slice : image) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));  // Add epsilon to avoid division by zero
        cv::threshold(slice, slice, config.epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    }
}

void DeconvolutionProcessor::init(const Hyperstack& input, const ImageMap<std::vector<std::shared_ptr<PSF>>>& psfs){
    if (!configured){
        std::__throw_runtime_error("Processor not configured");
    }

    setImageShapePadded(psfs);
   
    algorithm_->setBackend(backend_);


}





void DeconvolutionProcessor::configure(const DeconvolutionConfig config) {
    this->config = config;
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = fact.create(config);

    BackendFactory& bf = BackendFactory::getInstance();

    this->backend_ = bf.create(config.backenddeconv);
    this->backend_->mutableMemoryManager().setMemoryLimit(config.maxMem_GB * 1e9);
    this->cpuMemoryManager= bf.createMemManager("cpu");

    numberThreads = config.nThreads;
    // numberThreads = config.backenddeconv == "cuda" ? 1 : config.nThreads; // TODO change
    workerPool = std::make_shared<ThreadPool>(numberThreads);
    readwriterPool = std::make_shared<ThreadPool>(numberThreads + 2);

    configured = true;
}







void DeconvolutionProcessor::setImageOriginalShape(const Channel& channel) {
    imageOriginalShape.width = channel.image.slices[0].cols;
    imageOriginalShape.height = channel.image.slices[0].rows;
    imageOriginalShape.depth = channel.image.slices.size();
    imageOriginalShape.volume = imageOriginalShape.width * imageOriginalShape.height * imageOriginalShape.depth;
}


void DeconvolutionProcessor::setImageShapePadded(const ImageMap<std::vector<std::shared_ptr<PSF>>>& psfs){
    RectangleShape padding = getImagePadding(psfs);
    imageShapePadded = imageOriginalShape + padding;
}

RectangleShape DeconvolutionProcessor::getImagePadding(const ImageMap<std::vector<std::shared_ptr<PSF>>>& psfs){ // pad to largest psf
    RectangleShape padding{0,0,0};
    RectangleShape currentEntry;
    for (auto& it : psfs){
        for (auto& psf : it.entry){
            currentEntry.depth = psf->image.slices.size();
            currentEntry.height = psf->image.slices[0].rows;
            currentEntry.width = psf->image.slices[0].cols;
            if (currentEntry.depth > padding.depth){
                padding.depth = currentEntry.depth;
            }
            if (currentEntry.width > padding.width){
                padding.width = currentEntry.width;
            }
            if (currentEntry.height > padding.height){
                padding.height = currentEntry.height;
            }
        }
    }
    return padding;
}


size_t DeconvolutionProcessor::memoryForShape(const RectangleShape& shape){
    size_t algorithmMemoryMultiplier = algorithm_->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
    size_t memory = sizeof(complex) * shape.volume * algorithmMemoryMultiplier;
    return memory;
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







// void DeconvolutionProcessor::setupCubeArrangement() {

//     cubes.cubesPerX = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.width) / subimageShape.width));
//     cubes.cubesPerY = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.height) / subimageShape.height));
//     cubes.cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(imageOriginalShape.depth) / subimageShape.depth));
//     cubes.cubesPerLayer = cubes.cubesPerX * cubes.cubesPerY;

//     cubes.totalGridNum = cubes.cubesPerX * cubes.cubesPerY * cubes.cubesPerZ;
// }


// size_t DeconvolutionProcessor::getMemoryPerCube(size_t maxNumberThreads){

//     size_t algorithmMemoryMultiplier = algorithm_->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
//     size_t memoryBuffer = 1e9; // TESTVALUE
//     size_t availableMemory = backend_->mutableMemoryManager().getAvailableMemory() - memoryBuffer;
//     size_t memoryPerThread = availableMemory / maxNumberThreads;
//     size_t memoryPerCube = memoryPerThread / algorithmMemoryMultiplier;
//     return memoryPerCube;

    
// }





// void DeconvolutionProcessor::preprocessPSFs(
//     const std::vector<PSF>& psfs,
//     const ImageMap<std::string>& psfIDMap
// ){
//     std::unordered_map<std::string, ComplexData> tempMap;
//     for (auto psf : psfs){
//         tempMap.emplace(psf.ID, preprocessPSF(psf));
//     }
//     for (auto& coordPsfID: psfIDMap){
//         std::vector<ComplexData> data;
//         for (auto& psf : coordPsfID.psfs){
//            auto it = tempMap.find(psf);
//            if (it != tempMap.end()){
//             data.push_back(it->second);
//            }
//         }
//         psfMap.add(coordPsfID.box, data);
//     }
// }



// // find the psf of which the id corresopnds to the one in the config.layerPSFMap
// // place the corresponding prepared psf into the layerPreparedPSFMap
// // basically using the raw psf and raw ranges to map to the prepared psfs
// // refactor
// void DeconvolutionProcessor::initPSFMaps(const std::vector<PSF>& psfs){
//     if (config.layerPSFMap.empty() && config.cubePSFMap.empty()){
//         layerPreparedPSFMap.addRange(0, -1, preparedpsfs[0]); // failsafe
//         return;
//     }
//     for (const auto& range : config.layerPSFMap) {
//         for (int psfindex = 0; psfindex < psfs.size(); psfindex++) {
//             std::string ID = psfs[psfindex].ID;
//             for (const std::string& mapID : range.get()){
//                 if (ID == mapID){
//                     ComplexData& psfs_host = preparedpsfs[psfindex];
//                     layerPreparedPSFMap.addRange(range.start, range.end, std::move(psfs_host));
//                 }
//             }
//         }
//     }
// }


// int DeconvolutionProcessor::getLayerIndex(int cubeIndex, int cubesPerLayer){
//     return static_cast<int>(std::ceil(static_cast<double>((cubeIndex)) / cubesPerLayer));
// }





// const std::vector<const ComplexData*> DeconvolutionProcessor::selectPSFsForCube(int cubeIndex) {
//     int layerIndex = getLayerIndex(cubeIndex, cubes.cubesPerLayer);

//     std::vector<const ComplexData*> psfs;
    
//     std::vector<const ComplexData*> layerPSFs = layerPreparedPSFMap.getPointers(layerIndex);
//     psfs.insert(psfs.end(), layerPSFs.begin(), layerPSFs.end());
    
//     std::vector<const ComplexData*> cubePSFs = cubePreparedPSFMap.getPointers(cubeIndex);
//     psfs.insert(psfs.end(), cubePSFs.begin(), cubePSFs.end());

//     return psfs;

// }





// std::vector<cv::Mat> DeconvolutionProcessor::getPaddedCube(std::vector<cv::Mat>& image, BoxCoord box){

//     cv::Rect cubeSlice(widthStart, heightStart, widthEnd - widthStart, heightEnd - heightStart);
//     cv::Mat paddedSlice = image[z](cubeSlice).clone();

//     return 
// }


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

// std::vector<BoxCoord> DeconvolutionProcessor::preprocessChannel(Channel& channel){
    
//     Preprocessor::padToShape(channel.image.slices, imageShapePadded, config.borderType);

//     if (imageShapePadded != cubeShapePadded){
//         std::vector<std::vector<cv::Mat>> cubes = Preprocessor::splitImageHomogeneous(channel.image.slices,
//             subimageShape,
//             imageOriginalShape,
//             imageShapePadded,
//             cubeShapePadded);

//         return cubes;
//     }
//     else{
//         return std::vector<std::vector<cv::Mat>>{channel.image.slices};
//     }
// }

// void DeconvolutionProcessor::setPSFOriginalShape(const PSF& psf) {
//     psfOriginalShape.width = psf.image.slices[0].cols;
//     psfOriginalShape.height = psf.image.slices[0].rows;
//     psfOriginalShape.depth = psf.image.slices.size();
//     psfOriginalShape.volume = psfOriginalShape.width * psfOriginalShape.height * psfOriginalShape.depth;

// }

// void savecubeDebug(const std::vector<cv::Mat> cubeImage, const char* name, ImageMetaData metaData = globalmetadata){
//     // double global_max_val_tile= 0.0;
//     // double global_min_val_tile = MAXFLOAT;
//     // // Global normalization of the merged volume
//     // for (const auto& slice : cubeImage) {
//     //     cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
//     //     double min_val, max_val;
//     //     cv::minMaxLoc(slice, &min_val, &max_val);
//     //     global_max_val_tile = std::max(global_max_val_tile, max_val);
//     //     global_min_val_tile = std::min(global_min_val_tile, min_val);
//     //     }
    
//     // // Global normalization of the merged volume

//     // for (auto& slice : cubeImage) {
//     //     slice.convertTo(slice, CV_32F, 1.0 / (global_max_val_tile - global_min_val_tile), -global_min_val_tile * (1 / (global_max_val_tile - global_min_val_tile)));  // Add epsilon to avoid division by zero
//     //     cv::threshold(slice, slice, 0.0005, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
//     // }
//     Hyperstack cubeImageHyperstack;
//     cubeImageHyperstack.metaData = metaData;
//     cubeImageHyperstack.metaData.imageWidth = cubeImage[0].cols;
//     cubeImageHyperstack.metaData.imageLength = cubeImage[0].rows;
//     cubeImageHyperstack.metaData.slices = cubeImage.size();
//     // cubeImageHyperstack.metaData.bitsPerSample = 32;

//     Image3D image3d;
//     image3d.slices = cubeImage;
//     Channel channel;
//     channel.image = image3d;
//     cubeImageHyperstack.channels.push_back(channel);
//     cubeImageHyperstack.saveAsTifFile("../result/"+std::string(name)+".tif");
// }