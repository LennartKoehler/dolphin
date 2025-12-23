#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "frontend/SetupConfig.h"
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
#include "HelperClasses.h"

StandardDeconvolutionExecutor::StandardDeconvolutionExecutor(){
    // Initialize thread pool and processor will be done in configure


    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
    const RectangleShape shape,
    std::shared_ptr<PSF> inputPSF,
    std::shared_ptr<IBackend> backend
    ) -> ComplexData* {
        Preprocessor::padToShape(inputPSF->image, shape, 0);
        ComplexData h = convertCVMatVectorToFFTWComplex(inputPSF->image, shape);
        ComplexData h_device = backend->getMemoryManager().copyDataToDevice(h);
        backend->getDeconvManager().octantFourierShift(h_device);
        backend->getDeconvManager().forwardFFT(h_device, h_device);
        backend->sync();
        return new ComplexData(std::move(h_device));
    };
    psfPreprocessor.setPreprocessingFunction(psfPreprocessFunction);


}
StandardDeconvolutionExecutor::~StandardDeconvolutionExecutor(){
    psfPreprocessor.cleanup();
}

void StandardDeconvolutionExecutor::execute(const ChannelPlan& plan, const ImageReader& reader, const ImageWriter& writer) {
    parallelDeconvolution(plan, reader, writer);
}

void StandardDeconvolutionExecutor::configure(std::unique_ptr<DeconvolutionConfig> config) {
    
    BackendFactory& bf = BackendFactory::getInstance();

    this->cpuMemoryManager = bf.createMemManager("cpu");

    numberThreads = config->nThreads;
    int workerThreads;
    int ioThreads;

    if (config->backenddeconv == "cpu"){
        workerThreads = static_cast<int>(numberThreads * 0.75);
        ioThreads = workerThreads + 2 ;
    }
    else if (config->backenddeconv == "openmp"){

        workerThreads = 1; //TESTVALUE for openmp 
        ioThreads = workerThreads + 2;
        
    }
    else{
        workerThreads = 2;
        ioThreads = workerThreads + 2;
    }
    
    readwriterPool = std::make_shared<ThreadPool>(ioThreads);
    processor.init(workerThreads);
    configured = true;
}

void StandardDeconvolutionExecutor::configure(const SetupConfig& setupConfig) {
    // Base implementation - does not need specific setup config handling
    // Derived classes can override this for specialized configuration
}

std::function<void()> StandardDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const ImageReader& reader,
    const ImageWriter& writer) {
    
    return [this, task = *taskDesc, &reader, &writer]() {

        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

        PaddedImage cubeImage = reader.getSubimage(task.paddedBox);

        std::shared_ptr<IBackend> iobackend = task.backend->onNewThread(task.backend);
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);
        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);
        ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);
        iobackend->sync();


        ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();

        try {
            std::future<void> resultDone = processor.deconvolveSingleCube(
                iobackend,
                std::move(algorithm),
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
        // std::cout << iobackend->getMemoryManager().getAllocatedMemory() << std::endl; 
        cubeImage.image = convertFFTWComplexToCVMatVector(f_host);

        writer.setSubimage(cubeImage.image, task.paddedBox);


        loadingBar.addOne();
    };
}

void StandardDeconvolutionExecutor::parallelDeconvolution(
    const ChannelPlan& channelPlan,
    const ImageReader& reader,
    const ImageWriter& writer) {

    std::vector<std::future<void>> runningTasks;
    loadingBar.setMax(channelPlan.totalTasks);
    std::mutex writerMutex;

    for (const std::unique_ptr<CubeTaskDescriptor>& task : channelPlan.tasks) {
        std::function<void()> threadtask = createTask(task, reader, writer);
        runningTasks.push_back(readwriterPool->enqueue(threadtask));
    }

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();
}

// PaddedImage StandardDeconvolutionExecutor::preprocessChannel(Channel& image, const ChannelPlan& channelPlan) {
//     Preprocessor::padImage(image.image, channelPlan.imagePadding, config->borderType);
//     return PaddedImage{std::move(image.image.slices), channelPlan.imagePadding};
// }

void StandardDeconvolutionExecutor::postprocessChannel(Image3D& image){
    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    for (const auto& slice : image.slices) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO);
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }
    float epsilon = 1e-6; //TESTVALUE
    for (auto& slice : image.slices) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));
        cv::threshold(slice, slice, epsilon, 0.0, cv::THRESH_TOZERO);
    }
}



ComplexData StandardDeconvolutionExecutor::convertCVMatVectorToFFTWComplex(
    const Image3D& input, 
    const RectangleShape& shape) {

    ComplexData result = cpuMemoryManager->allocateMemoryOnDevice(shape);

    int width = shape.width;
    int height = shape.height;
    int depth = shape.depth;
    
    for (int z = 0; z < depth; ++z) {
        CV_Assert(input.slices[z].type() == CV_32F);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                result.data[z * height * width + y * width + x][0] = static_cast<double>(input.slices[z].at<float>(y, x));
                result.data[z * height * width + y * width + x][1] = 0.0;
            }
        }
    }

    return result;
}

Image3D StandardDeconvolutionExecutor::convertFFTWComplexToCVMatVector(
        const ComplexData& input)
{
    const int width  = input.size.width;
    const int height = input.size.height;
    const int depth  = input.size.depth;

    Image3D output;
    output.slices.reserve(depth);

    const auto* in = input.data;

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

        output.slices.push_back(result);
    }

    return output;
}