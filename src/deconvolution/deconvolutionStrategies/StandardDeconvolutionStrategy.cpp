#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"
#include "deconvolution/ImageMap.h"
#include "HelperClasses.h"




StandardDeconvolutionStrategy::StandardDeconvolutionStrategy(){
    
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


Hyperstack StandardDeconvolutionStrategy::run(const Hyperstack& image, const std::vector<PSF>& inputPSFS){

    Hyperstack inputCopy(image); // copy to not edit in place, the inputcopy is used to be padded and then used for reading
    Hyperstack result(image); // to get the correct shape so i can later paste the result in the corresponding position
    // i need both because cubes might overlap, so i cant write in place


    for (int i = 0; i < image.channels.size(); i++){
        ComputationalPlan channelPlan = createComputationalPlan(i, image.channels[i].image, inputPSFS, *config, backend_, algorithm_);

        PaddedImage channel = preprocessChannel(inputCopy.channels[i].image.slices, channelPlan);
        parallelDeconvolution(channel, result.channels[i].image.slices, channelPlan);
        postprocessChannel(channel);

        std::cout << "[STATUS] Saving result of channel " << std::endl;
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return result;
}

PaddedImage StandardDeconvolutionStrategy::preprocessChannel(std::vector<cv::Mat>& input, const ComputationalPlan& channelPlan) {
    Preprocessor::padImage(input, channelPlan.imagePadding, config->borderType); // pad to largest psf, should be the easiest
    return PaddedImage{std::move(input), channelPlan.imagePadding};
}



void StandardDeconvolutionStrategy::parallelDeconvolution(
        const PaddedImage& inputImagePadded,
        std::vector<cv::Mat>& outputImage,
        const ComputationalPlan& channelPlan) {

    std::vector<std::future<void>> runningTasks;

    loadingBar.setMax(channelPlan.totalTasks);

    std::atomic<int> numberTasks(channelPlan.totalTasks);
    std::mutex writerMutex;

    for (const std::unique_ptr<CubeTaskDescriptor>& task : channelPlan.tasks) {

        // this task is run by the readwriterPool, which will then enqueue the work itself into the workerpool
        // since there are more readwriters than workers, the workers should always be occupied with work,
        std::function<void()> threadtask = createTask(task, inputImagePadded, outputImage, writerMutex);

        runningTasks.push_back(readwriterPool->enqueue(threadtask));
    }

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();
}

std::function<void()> StandardDeconvolutionStrategy::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const PaddedImage& inputImagePadded,
    std::vector<cv::Mat>& outputImage,
    std::mutex& writerMutex
) {
    StandardCubeTaskDescriptor* standardTask = dynamic_cast<StandardCubeTaskDescriptor*>(taskDesc.get());
    if (!standardTask) {
        throw std::runtime_error("Expected StandardCubeTaskDescriptor but got different type");
    }

    return [this, task = *standardTask, &inputImagePadded, &writerMutex, &outputImage]() {
        
        RectangleShape workShape = task.srcBox.dimensions + task.requiredPadding.before + task.requiredPadding.after;
        // PaddedImage cubeImage = getCubeImage(inputImagePadded, taskDesc.srcBox, taskDesc.requiredPadding);

        PaddedImage cubeImage = getCubeImage(inputImagePadded, task.srcBox, task.requiredPadding);



        std::shared_ptr<IBackend> iobackend = backend_->onNewThread();
        ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};

        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);
        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);
        ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

        try {
            std::future<void> resultDone = processor.deconvolveSingleCube(
                iobackend,
                algorithm_,
                workShape,
                task.psfs,
                g_device,
                f_device,
                psfPreprocessor);

            resultDone.get(); //wait for result
            f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
            iobackend->releaseBackend();
        }
        catch (...) {
            throw; // dont overwrite image if exception
        }
        cubeImage.image = convertFFTWComplexToCVMatVector(f_host);
        {
            std::unique_lock<std::mutex> lock(writerMutex);
            // deconovlutionStrategy.insertImage(cubeImage, outputImage);
            // Postprocessor::insertLabeledCubeInImage(cubeImage, outputImage, taskDesc);
            Postprocessor::insertCubeInImage(cubeImage, outputImage, task.srcBox);
        }
        loadingBar.addOne();
    };
}




// TODOmove to preprocessor
PaddedImage StandardDeconvolutionStrategy::getCubeImage(const PaddedImage& paddedImage, const BoxCoord& coords, const Padding& cubePadding){
    // remember that coords is for the original image, not the padded Image, so it has to be shifted, so the paddingShift member variable can be used
    assert(coords.x >= 0 && coords.y >= 0 && coords.z >= 0 && "getCubeImage source Box coordinates must be non-negative");
    assert(coords.x + coords.dimensions.width <= paddedImage.image[0].cols && "getCubeImage cube extends beyond image width");
    assert(coords.y + coords.dimensions.height <= paddedImage.image[0].rows && "getCubeImage cube extends beyond image height");
    assert(coords.z + coords.dimensions.depth <= static_cast<int>(paddedImage.image.size()) && "getCubeImage cube extends beyond image depth");
    assert(paddedImage.padding.before >= cubePadding.before && "getCubeImage too little image padding");
    assert(paddedImage.padding.after >= cubePadding.after && "getCubeImage too little image padding");

    std::vector<cv::Mat> cube;
    cube.reserve(coords.dimensions.depth + cubePadding.before.depth + cubePadding.after.depth);
    
    // Triple nested loop to iterate through all cube positions with padding
    int xImage = coords.x + paddedImage.padding.before.width - cubePadding.before.width;
    int yImage = coords.y + paddedImage.padding.before.height - cubePadding.before.height;

    RectangleShape totalPadding = cubePadding.before + cubePadding.after;

    for (int zCube = 0; zCube < coords.dimensions.depth + totalPadding.depth; ++zCube) {
        

        // Define the ROI in the source image
        cv::Rect imageROI(xImage, yImage, coords.dimensions.width + totalPadding.width, coords.dimensions.height + totalPadding.height);
        int zImage = zCube + coords.z + paddedImage.padding.before.depth - cubePadding.before.depth;
        cv::Mat slice(coords.dimensions.height + totalPadding.height, coords.dimensions.width + totalPadding.width, CV_32F, cv::Scalar(0));

        // Copy from the source image ROI to the entire slice
        paddedImage.image[zImage](imageROI).copyTo(slice);

        cube.push_back(std::move(slice));
    }
    
    return PaddedImage{std::move(cube), cubePadding};
}

Padding StandardDeconvolutionStrategy::getCubePadding(const std::vector<std::shared_ptr<PSF>> psfs){
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
    
    RectangleShape paddingbefore = RectangleShape(
        static_cast<int>(maxPsfShape.width / 2),
        static_cast<int>(maxPsfShape.height / 2),
        static_cast<int>(maxPsfShape.depth / 2)
    );
    return Padding{paddingbefore, paddingbefore};
}


void StandardDeconvolutionStrategy::postprocessChannel(PaddedImage& image){


    Postprocessor::removePadding(image.image, image.padding);
    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    for (const auto& slice : image.image) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }

    for (auto& slice : image.image) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));  // Add epsilon to avoid division by zero
        cv::threshold(slice, slice, config->epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    }
}





void StandardDeconvolutionStrategy::configure(std::unique_ptr<DeconvolutionConfig> configuration) {
    this->config = std::move(configuration);
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    this->algorithm_ = fact.create(*config);

    BackendFactory& bf = BackendFactory::getInstance();

    this->backend_ = bf.create(config->backenddeconv);
    this->backend_->mutableMemoryManager().setMemoryLimit(config->maxMem_GB * 1e9);
    this->cpuMemoryManager= bf.createMemManager("cpu");

    numberThreads = config->nThreads;
    int workerThreads;
    int ioThreads;

    if (config->backenddeconv == "cpu"){
        workerThreads = static_cast<int>(numberThreads * 0.75);
        ioThreads = workerThreads + 2 ;
    }
    else{
        workerThreads = numberThreads;
        ioThreads = workerThreads + 2;
    }

    readwriterPool = std::make_shared<ThreadPool>(ioThreads);
    processor.init(workerThreads);
    configured = true;
}




size_t StandardDeconvolutionStrategy::memoryForShape(const RectangleShape& shape){
    size_t algorithmMemoryMultiplier = algorithm_->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
    size_t memory = sizeof(complex) * shape.volume * algorithmMemoryMultiplier;
    return memory;
}

// Conversion Functions
ComplexData StandardDeconvolutionStrategy::convertCVMatVectorToFFTWComplex(
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

std::vector<cv::Mat> StandardDeconvolutionStrategy::convertFFTWComplexToCVMatVector(
        const ComplexData& input)
{
    const int width  = input.size.width;
    const int height = input.size.height;
    const int depth  = input.size.depth;

    std::vector<cv::Mat> output;
    output.reserve(depth);

    const auto* in = input.data;   // pointer to FFTW complex data

    for (int z = 0; z < depth; ++z) {

        cv::Mat result(height, width, CV_32F);
        float* dst = reinterpret_cast<float*>(result.data);

        const int sliceSize = width * height;
        int baseIndex = z * sliceSize;

        for (int i = 0; i < sliceSize; ++i) {
            double real = in[baseIndex + i][0];
            double imag = in[baseIndex + i][1];
            dst[i] = static_cast<float>(std::sqrt(real * real + imag * imag));
        }

        output.push_back(result);
    }

    return output;
}



ComputationalPlan StandardDeconvolutionStrategy::createComputationalPlan(
        int channelNumber,
        const Image3D& image,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm
){
    
    // validateConfiguration(psfs, imageShape, channelNumber, config, backend, algorithm);
    ComputationalPlan plan;
    plan.imagePadding = Padding{RectangleShape{30,30,30}, RectangleShape{30,30,30}}; //TESTVALUE //TODO
    // plan.imagePadding = getChannelPadding();
    plan.executionStrategy = ExecutionStrategy::PARALLEL;



    std::vector<std::shared_ptr<PSF>> psfPointers;
    for (const auto& psf: psfs){
        psfPointers.push_back(std::make_shared<PSF>(psf));
    }

    
    // Use the imageShape parameter
    RectangleShape imageSize = RectangleShape{image.slices[0].cols, image.slices[0].rows, static_cast<int>(image.slices.size())};

    size_t t = config.nThreads;
    // t = static_cast<size_t>(2);
    size_t memoryPerCube = maxMemoryPerCube(t, config.maxMem_GB * 1e9, algorithm);
    RectangleShape idealCubeSize = getCubeShape(memoryPerCube, config.nThreads, imageSize, psfs, paddingStrat);

    Padding cubePadding = getCubePadding(psfPointers);

    // idealCubeSize = RectangleShape(393, 313, 46);
    std::vector<BoxCoord> cubeCoordinates = splitImageHomogeneous(idealCubeSize, imageSize);
    // Create task descriptors for each cube

    plan.tasks.reserve(cubeCoordinates.size());
    for (size_t i = 0; i < cubeCoordinates.size(); ++i) {
        StandardCubeTaskDescriptor descriptor;
        descriptor.taskId = static_cast<int>(i);
        descriptor.srcBox = cubeCoordinates[i];
        descriptor.psfs = psfPointers;
        descriptor.estimatedMemoryUsage = estimateMemoryUsage(idealCubeSize, algorithm);
        descriptor.requiredPadding = cubePadding;
        // setPSFsAndLabel(descriptor, psfPointers, labelImage, psfLabelMap);

        plan.tasks.push_back(std::make_unique<StandardCubeTaskDescriptor>(descriptor));
    }
    
    plan.totalTasks = plan.tasks.size();
    return plan;
}


// void StandardDeconvolutionStrategy::setPSFsAndLabel(
//     StandardCubeTaskDescriptor& task,
//     const std::vector<std::shared_ptr<PSF>>& psfs,
//     const std::shared_ptr<Image3D> labelImage,
//     const std::unordered_map<int, PSFID> psfLabelMap
// ){

//     for (auto [label, psfid] : psfLabelMap){
//         std::vector<std::shared_ptr<PSF>> assignedPSFs;
//         LabelGroup labelGroup(label, labelImage);

//         for (const std::shared_ptr<PSF>& psf : psfs){
//             if ( psfid == psf->ID){
//                 assignedPSFs.push_back(psf);
//             }
//         }
//         labelGroup.setPSFs(assignedPSFs);
//     }
// }


// ImageMap<std::vector<std::shared_ptr<PSF>>> StandardDeconvolutionStrategy::addPSFS(std::vector<BoxCoord>& coords, const std::vector<PSF>& psfs){
//     ImageMap<std::vector<std::shared_ptr<PSF>>> result;

//     for (auto& coordinate : coords){
//         std::vector<std::shared_ptr<PSF>> psfPointers;
//         for (const auto& psf : psfs){
//             psfPointers.push_back(std::make_shared<PSF>(psf));
//         }
//         result.add(coordinate, std::move(psfPointers));
//     }
//     return result;
// }

size_t StandardDeconvolutionStrategy::maxMemoryPerCube(
    size_t maxNumberThreads, 
    size_t maxMemory,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm){
    
    size_t algorithmMemoryMultiplier = algorithm->getMemoryMultiplier(); // how many copies of a cube does each algorithm have
    size_t memoryBuffer = 1e9; // TESTVALUE
    size_t memoryPerThread = maxMemory / maxNumberThreads;
    size_t memoryPerCube = memoryPerThread / algorithmMemoryMultiplier;
    return memoryPerCube; 
}

size_t StandardDeconvolutionStrategy::estimateMemoryUsage(
    const RectangleShape& cubeSize,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm
){
    return cubeSize.volume * algorithm->getMemoryMultiplier() * sizeof(complex);
}

//test function //TESTVALUE
RectangleShape StandardDeconvolutionStrategy::getCubeShape(
    size_t memoryPerCube,
    size_t numberThreads,
    const RectangleShape& imageOriginalShape,
    const std::vector<PSF>& psfs,
    const PaddingStrategy& paddingStrategy
){
    size_t width = 128;
    size_t height = 128;
    size_t depth = 128;
    std::vector<RectangleShape> psfSizes;
    for (const auto& psf : psfs){
        psfSizes.push_back(psf.image.getShape());
    }
    Padding padding = paddingStrategy.getPadding(imageOriginalShape, psfSizes);
    return RectangleShape(128, 128, 64) - padding.before - padding.after;
}

// RectangleShape StandardDeconvolutionStrategy::getCubeShape(
//     size_t memoryPerCube,
//     size_t numberThreads,
//     RectangleShape imageOriginalShape,
//     RectangleShape imagePadding
//     const PaddingStrategy& paddingStrat

// ) {
//     // this function determines the shape into which the input image is cut
//     // current strategy is to only slice the largest dimension while leaving the smaller two dimensions the same shape
//     // the constraints are that all threads should be used but it all needs to fit on the available memory
//     // due to padding it is most optimal (smallest 3dshape) to have all dimensions the same size as this reduces the increase in volume caused by padding
//     // but this is difficult as we want to have all threads have a similar workload aswell as reducing the overhead of each thread having to read/write more than once
//     // ideally we have number of cubes (all dim same length) of equal size equal to number of threads
//     // there are different strategies to split the original image but this is just what I went with
//     // it is useful to keep all cubes the same dimensionality as the psfs then only need to be transformed once into that shape and the fftw plans can be reused


//     size_t maxMemCubeVolume = memoryPerCube / sizeof(complex); // cut into pieces so that they still fit on memory


//     RectangleShape subimageShape = imageOriginalShape;
//     std::array<int*, 3> sortedDimensionsSubimage = subimageShape.getDimensionsAscending();
//     size_t maxThreadcubeLargestDim = (*sortedDimensionsSubimage[2] + numberThreads -1) / numberThreads; // ceiling divide

//     RectangleShape tempMemory = imageOriginalShape;
//     std::array<int*, 3> sortedDimensionsMemory = tempMemory.getDimensionsAscending();
//     size_t maxMemCubeLargestDim = maxMemCubeVolume / (*sortedDimensionsMemory[0] * *sortedDimensionsMemory[1]);

//     *sortedDimensionsSubimage[2] = std::min(maxMemCubeLargestDim, maxThreadcubeLargestDim);
//     assert(*sortedDimensionsSubimage[2] != 0 && "[ERROR] getCubeShape: not enough memory to fit a single slice of the image");


//     subimageShape.updateVolume();
//     return subimageShape;

//     // TODO could also start halfing the other dimension until it fits
//     // idea: always half the largest dimension until:
//     //      number of cubes = number of threads && size of cubes fits on memory

//     // size_t cubeVolume = std::min(memCubeVolume, threadCubeVolume);
//     // double scaleFactor = std::cbrt( static_cast<double>(cubeVolume) / imageShapePadded.volume);
//     // subimageShape = imageOriginalShape * scaleFactor;
//     // subimageShape.clamp(imageOriginalShape);

//     // cubeShapePadded = subimageShape + padding;

// }