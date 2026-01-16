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

void StandardDeconvolutionExecutor::execute(const ChannelPlan& plan, const ImageReader& reader, const ImageWriter& writer) {
    parallelDeconvolution(plan, reader, writer);
}

void StandardDeconvolutionExecutor::configure(std::unique_ptr<DeconvolutionConfig> config) {
    
    BackendFactory& bf = BackendFactory::getInstance();

    this->cpuMemoryManager = std::make_shared<DefaultBackendMemoryManager>();

    numberThreads = config->nThreads;
    int workerThreads;
    int ioThreads;

    if (config->backenddeconv == "../backends/cpu/libcpu_backend.so"){ // TODO MOVE TO CONFIG?
        workerThreads = static_cast<int>(numberThreads * 0.75);
        ioThreads = workerThreads + 2 ; // test" << std::this_thread::get_id() << "VALUE
    }
    else if (config->backenddeconv == "../backends/cpu/libopenmp_backend.so"){

        workerThreads = 1; //test" << std::this_thread::get_id() << "VALUE for openmp 
        ioThreads = workerThreads + 2;
        
    }
    else{
        workerThreads = numberThreads;
        ioThreads = workerThreads + 2;
    }
    
    workerThreads = std::max(1, workerThreads);
    ioThreads = std::max(1, ioThreads);
    readwriterPool = std::make_shared<ThreadPool>(ioThreads);

    processor.init(workerThreads, [](){});
    configured = true;
}

void StandardDeconvolutionExecutor::configure(const SetupConfig& setupConfig) {

}

// std::function<void()> StandardDeconvolutionExecutor::createTask(
//     const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
//     const ImageReader& reader,
//     const ImageWriter& writer) {
    
//     return [this, task = *taskDesc, &reader, &writer]() {
//         RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

//         PaddedImage cubeImage = reader.getSubimage(task.paddedBox);

//         std::shared_ptr<IBackend> iobackend = task.backend->onNewThread(task.backend);
//         ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);

//         ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);

//         cpuMemoryManager->freeMemoryOnDevice(g_host);

//         ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

//         ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
//         std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();


//         try {
//             std::future<void> resultDone = processor.deconvolveSingleCube(
//                 iobackend,
//                 std::move(algorithm),
//                 workShape,
//                 task.psfs,
//                 g_device,
//                 f_device,
//                 psfPreprocessor);

//             resultDone.get(); //wait for result
//             f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
//             iobackend->releaseBackend();
//         }
//         catch (...) {
//             throw; // dont overwrite image if exception
//         }

//         cubeImage.image = convertFFTWComplexToCVMatVector(f_host);

//         writer.setSubimage(cubeImage.image, task.paddedBox);


//         loadingBar.addOne();
//     };
// }

std::function<void()> StandardDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const ImageReader& reader,
    const ImageWriter& writer) {
    
    return [this, task = *taskDesc, &reader, &writer]() {

        thread_local std::shared_ptr<IBackend> iobackend = task.backend->onNewThread(task.backend); //TODO with thread local the task.backend is irrelevant (except the first few)
        // thread_local DeconvolutionProcessor tl_processor{};
        // thread_local struct ThreadInitializer {
        //     ThreadInitializer() { 
        //         tl_processor.init(1);
        //     }
        // } thread_init;


        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

        PaddedImage cubeImage = reader.getSubimage(task.paddedBox);


        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);

        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);

        cpuMemoryManager->freeMemoryOnDevice(g_host);

        ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

        ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();


        try {
            f_device = DeconvolutionProcessor::staticDeconvolveSingleCube(
                iobackend,
                std::move(algorithm),
                workShape,
                task.psfs,
                g_device,
                f_device,
                psfPreprocessor);

            // resultDone.get(); //wait for result
            f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
            // iobackend->releaseBackend();
        }
        catch (...) {
            throw; // dont overwrite image if exception
        }

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