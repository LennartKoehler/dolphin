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
    
    std::function<ComplexData*(RectangleShape, std::shared_ptr<PSF>&)> psfPreprocessFunction = [&](
    RectangleShape shape,
    std::shared_ptr<PSF>& inputPSF
    ) -> ComplexData* {

        std::cout << "[STATUS] Preprocessing PSFs" << std::endl;
            
        Preprocessor::padToShape(inputPSF->image.slices, shape, 0);
        ComplexData h = convertCVMatVectorToFFTWComplex(inputPSF->image.slices, shape);
        ComplexData h_device = backend_->getMemoryManager().copyDataToDevice(h);
        backend_->getDeconvManager().octantFourierShift(h_device);
        backend_->getDeconvManager().forwardFFT(h_device, h_device);
        return new ComplexData(std::move(h_device));
    };
    psfPreprocessor.setPreprocessingFunction(psfPreprocessFunction);
}


Hyperstack DeconvolutionProcessor::run(Hyperstack& input, ImageMap<std::shared_ptr<PSF>>& psfs){
    init(input, psfs);
    Image3D deconvolutedImage;
    std::cout << "[STATUS] Starting deconvolution" << std::endl;

    for (auto& channel : input.channels){
        Preprocessor::padToShape(channel.image.slices, imageShapePadded, config.borderType); // pad to largest psf, should be the easiest

        parallelDeconvolution(channel.image.slices, psfs);
        postprocessChannel(input.metaData, channel.image.slices);

        std::cout << "[STATUS] Saving result of channel " << std::endl;
        deconvolutedImage.slices = channel.image.slices;
        input.channels[channel.id].image = deconvolutedImage;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return input;
}

std::vector<cv::Mat> DeconvolutionProcessor::getCubeImage(const std::vector<cv::Mat>& image, BoxCoord coords, RectangleShape padding){
    std::vector<cv::Mat> cube;
    cube.reserve(coords.depth + 2 * padding.depth);
    
    // Triple nested loop to iterate through all cube positions with padding
    for (int z = coords.z - padding.depth; z < coords.z + coords.depth + padding.depth; ++z) {
        int actual_z = std::max(0, std::min(z, static_cast<int>(image.size()) - 1));
        
        cv::Mat slice(coords.height + 2 * padding.height, coords.width + 2 * padding.width, CV_32F, cv::Scalar(0));
        
        for (int y = coords.y - padding.height; y < coords.y + coords.height + padding.height; ++y) {
            for (int x = coords.x - padding.width; x < coords.x + coords.width + padding.width; ++x) {
                int actual_x = std::max(0, std::min(x, image[actual_z].cols - 1));
                int actual_y = std::max(0, std::min(y, image[actual_z].rows - 1));
                
                int cube_x = x - (coords.x - padding.width);
                int cube_y = y - (coords.y - padding.height);
                
                slice.at<float>(cube_y, cube_x) = image[actual_z].at<float>(actual_y, actual_x);
            }
        }
        cube.push_back(slice);
    }
    
    return cube;
}

RectangleShape DeconvolutionProcessor::getCubePadding(BoxCoord box){
    return RectangleShape(static_cast<int>(box.width/2), static_cast<int>(box.height/2), static_cast<int>(box.depth/2));
}


void DeconvolutionProcessor::parallelDeconvolution(
        std::vector<cv::Mat>& image,
        ImageMap<std::shared_ptr<PSF>>& psfMap) {

    std::vector<std::future<void>> runningTasks;
    std::atomic<int> processedCount(0);
    std::mutex loadingBarMutex;

    std::mutex writerMutex;

    std::mutex memoryMutex;
    std::condition_variable memoryFull;
    bool memoryAvailable = true;

    // std::mutex queueMutex;
    // std::condition_variable queueFull;
    // const int maxNumberWorkerThreads = numberThreads;

    std::atomic<int> numberCubes(psfMap.size());

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

        // std::vector<const ComplexData*> psfs = selectPSFsForCube(cubeIndex);
        BoxEntryPair psfs = psfMap.get(cubeIndex);
        RectangleShape padding = getCubePadding(psfs.box);

        std::vector<cv::Mat> cubeImage = getCubeImage(image, psfs.box, padding);

        RectangleShape workShape = RectangleShape(psfs.box.width, psfs.box.height, psfs.box.depth) + padding * 2;
        std::vector<const ComplexData*> preprocessedPSFs;
        for (auto& psf : psfs.entry){
            preprocessedPSFs.emplace_back(psfPreprocessor.getPreprocessedPSF(workShape, psf));
        }

        // Capture values by copy/move to avoid unnecessary copies while maintaining mutability
        BoxCoord srcBox = psfs.box;

        auto task = [this, cubeIndex, 
                     cubeImage = std::move(cubeImage), 
                     preprocessedPSFs = std::move(preprocessedPSFs), 
                     srcBox = std::move(srcBox), 
                     workShape = std::move(workShape),
                     &loadingBarMutex, &writerMutex, &memoryMutex, &memoryFull, &memoryAvailable,
                     &processedCount, &numberCubes, &image, &runningTasks]() mutable {
            try{
                deconvolveSingleCube(
                    backend_,
                    algorithm_->clone(),
                    cubeImage,
                    workShape,
                    preprocessedPSFs);
                {
                    std::unique_lock<std::mutex> lock(loadingBarMutex);
                    loadingBar(++processedCount, numberCubes);

                }
                {
                    std::unique_lock<std::mutex> lock(writerMutex);
                    Postprocessor::insertCubeInImage(cubeImage, image, srcBox, workShape);
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
                // runningTasks.push_back(threadPool->enqueue(task));
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

        runningTasks.push_back(threadPool->enqueue(task));
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
    const std::vector<const ComplexData*> psfs_device) {

    ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
    
    try{
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage, workShape);
        ComplexData g_device = backend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);
        ComplexData f_device = backend->getMemoryManager().allocateMemoryOnDevice(workShape);

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




void DeconvolutionProcessor::postprocessChannel(ImageMetaData& metaData, std::vector<cv::Mat>& image){


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

void DeconvolutionProcessor::init(const Hyperstack& input, ImageMap<std::shared_ptr<PSF>>& psfs){
    if (!configured){
        std::__throw_runtime_error("Processor not configured");
    }
    // size_t memoryPerCube = getMemoryPerCube(numberThreads);

    setImageOriginalShape(input.channels[0]);
    // setPSFOriginalShape(psfs.front()); // TODO for multiple psfs this would have to be the largest psf to get the proper padding
    setImageShapePadded(psfs);
    // setWorkShapes(imageOriginalShape, psfOriginalShape, memoryPerCube);
    // setupCubeArrangement();
   
    algorithm_->setBackend(backend_);

    // preprocessPSFs(psfs, psfIDMap);

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







void DeconvolutionProcessor::setImageOriginalShape(const Channel& channel) {
    imageOriginalShape.width = channel.image.slices[0].cols;
    imageOriginalShape.height = channel.image.slices[0].rows;
    imageOriginalShape.depth = channel.image.slices.size();
    imageOriginalShape.volume = imageOriginalShape.width * imageOriginalShape.height * imageOriginalShape.depth;
}


void DeconvolutionProcessor::setImageShapePadded(const ImageMap<std::shared_ptr<PSF>>& psfs){
    RectangleShape padding = getPadding(psfs);
    imageShapePadded = imageOriginalShape + padding;
}

RectangleShape DeconvolutionProcessor::getPadding(const ImageMap<std::shared_ptr<PSF>>& psfs){
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


// void DeconvolutionProcessor::setWorkShapes(
//     const RectangleShape& imageOriginalShape,
//     const RectangleShape& padding,
//     size_t memoryPerCube
// ) {
//     // this function determines the shape into which the input image is cut
//     // current strategy is to only slice the largest dimension while leaving the smaller two dimensions the same shape
//     // the constraints are that all threads should be used but it all needs to fit on the available memory
//     // due to padding it is most optimal (smallest 3dshape) to have all dimensions the same size as this reduces the increase in volume caused by padding
//     // but this is difficult as we want to have all threads have a similar workload aswell as reducing the overhead of each thread having to read/write more than once
//     // ideally we have number of cubes (all dim same length) of equal size equal to number of threads
//     // there are different strategies to split the original image but this is just what I went with
//     // it is useful to keep all cubes the same dimensionality as the psfs then only need to be transformed once into that shape and the fftw plans can be reused

//     imageShapePadded = imageOriginalShape + padding;

//     size_t maxMemCubeVolume = memoryPerCube / sizeof(complex); // cut into pieces so that they still fit on memory

//     subimageShape = imageOriginalShape;
//     std::array<int*, 3> sortedDimensionsSubimage = subimageShape.getDimensionsAscending();
//     size_t maxThreadcubeLargestDim = (*sortedDimensionsSubimage[2] + numberThreads -1) / numberThreads; // ceiling divide

//     RectangleShape tempPadded = imageOriginalShape + padding;
//     std::array<int*, 3> sortedDimensionsPadded = tempPadded.getDimensionsAscending();
//     size_t maxMemCubeLargestDim = maxMemCubeVolume / (*sortedDimensionsPadded[0] * *sortedDimensionsPadded[1]);

//     *sortedDimensionsSubimage[2] = std::min(maxMemCubeLargestDim, maxThreadcubeLargestDim);
//     assert(*sortedDimensionsSubimage[2] != 0 && "[ERROR] setWorkShapes: not enough memory to fit a single slice of the image");


//     subimageShape.updateVolume();
//     cubeShapePadded = subimageShape + padding;
//     // TODO could also start halfing the other dimension until it fits
//     // idea: always half the largest dimension until:
//     //      number of cubes = number of threads && size of cubes fits on memory

//     // size_t cubeVolume = std::min(memCubeVolume, threadCubeVolume);
//     // double scaleFactor = std::cbrt( static_cast<double>(cubeVolume) / imageShapePadded.volume);
//     // subimageShape = imageOriginalShape * scaleFactor;
//     // subimageShape.clamp(imageOriginalShape);

//     // cubeShapePadded = subimageShape + padding;

// }




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