#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "frontend/SetupConfig.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include <iostream>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "deconvolution/Postprocessor.h"
#include "backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "HelperClasses.h"
#include "backend/DefaultBackendMemoryManager.h"
#include <thread>

StandardDeconvolutionExecutor::StandardDeconvolutionExecutor(){
    // Initialize thread pool and processor will be done in configure

    this->cpuMemoryManager = std::make_shared<DefaultBackendMemoryManager>();

    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
        const RectangleShape shape,
        std::shared_ptr<PSF> inputPSF,
        std::shared_ptr<IBackend> backend
            ) -> ComplexData* {
                Preprocessor::padToShape(inputPSF->image, shape, PaddingType::ZERO);
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

void StandardDeconvolutionExecutor::execute(const ChannelPlan& plan) {
    parallelDeconvolution(plan);
}

void StandardDeconvolutionExecutor::configure(std::unique_ptr<DeconvolutionConfig> config) {

}

void StandardDeconvolutionExecutor::configure(const SetupConfig& setupConfig) {

}

std::function<void()> StandardDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc) {
    
    return [this, task = *taskDesc]() {

        TaskContext* context = task.context.get();

        thread_local std::shared_ptr<IBackend> iobackend = context->prototypebackend->onNewThreadSharedMemory(context->prototypebackend);

        std::shared_ptr<ImageReader> reader = task.reader;
        std::shared_ptr<ImageWriter> writer = task.writer;


        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

        PaddedImage cubeImage = reader->getSubimage(task.paddedBox);

        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);

        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);

        cpuMemoryManager->freeMemoryOnDevice(g_host);

        ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

        ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();


        try {
            std::future<void> resultDone = context->processor.deconvolveSingleCube(
                iobackend,
                std::move(algorithm),
                workShape,
                task.psfs,
                g_device,
                f_device,
                psfPreprocessor);

            resultDone.get(); //wait for result
            f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
        }
        catch (...) {
            throw; // dont overwrite image if exception
        }

        cubeImage.image = convertFFTWComplexToCVMatVector(f_host);

        writer->setSubimage(cubeImage.image, task.paddedBox);


        loadingBar.addOne();
    };
}



void StandardDeconvolutionExecutor::parallelDeconvolution(
    const ChannelPlan& channelPlan) {

    std::vector<std::future<void>> runningTasks;
    loadingBar.setMax(channelPlan.totalTasks);
    std::mutex writerMutex;

    for (const std::unique_ptr<CubeTaskDescriptor>& task : channelPlan.tasks) {


        std::function<void()> threadtask = createTask(task);
        runningTasks.push_back(task->context->ioPool.enqueue(threadtask));
    }

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();
}




ComplexData StandardDeconvolutionExecutor::convertCVMatVectorToFFTWComplex(
    const Image3D& input, 
    const RectangleShape& shape) {

    ComplexData result = cpuMemoryManager->allocateMemoryOnDevice(shape);

    int width = shape.width;
    int height = shape.height;
    int depth = shape.depth;
    
    int index = 0;
    for (const auto& it : input) {
        
        result.data[index][0] = static_cast<double>(it);
        result.data[index][1] = 0.0;
        index ++;
    }

    return result;
}

Image3D StandardDeconvolutionExecutor::convertFFTWComplexToCVMatVector(
        const ComplexData& input)
{
    const int width  = input.size.width;
    const int height = input.size.height;
    const int depth  = input.size.depth;

    Image3D output(RectangleShape(width, height, depth));

    const auto* in = input.data;
    int index = 0;
    for (auto& it : output) {
        double real = in[index][0];
        double imag = in[index][1];
        it = static_cast<float>(std::sqrt(real * real + imag * imag));
        index ++;

    }

    return output;
}