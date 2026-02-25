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

#include "dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>
#include <iostream>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphin/HelperClasses.h"
#include <thread>






StandardDeconvolutionExecutor::StandardDeconvolutionExecutor(){

}
StandardDeconvolutionExecutor::~StandardDeconvolutionExecutor(){
 }

void StandardDeconvolutionExecutor::execute(const DeconvolutionPlan& plan) {
    parallelDeconvolution(plan);
}

void StandardDeconvolutionExecutor::configure(std::unique_ptr<DeconvolutionConfig> config) {

}

void StandardDeconvolutionExecutor::configure(const SetupConfig& setupConfig) {

}


void StandardDeconvolutionExecutor::runTask(const CubeTaskDescriptor& task){

    using progressFunction = std::function<void(int)>;
    progressFunction tracker = [this](int max){
        float iteration = 1.0 / max;
        this->loadingBar.add(iteration);
    };

    TaskContext* context = task.context.get();
    // thread_local IBackend& iobackend = context->iobackend.cloneSharedMemory();
    thread_local IBackend& iobackend = context->manager.getBackend(context->ioconfig); 
    thread_local IBackend& workerbackend = context->manager.cloneSharedMemory(iobackend, context->workerconfig); // copied in deconvolutionprocessor

    std::shared_ptr<ImageReader> reader = task.reader;
    std::shared_ptr<ImageWriter> writer = task.writer;

    CuboidShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

    std::optional<PaddedImage> cubeImage_o = reader->getSubimage(task.paddedBox); 
    if (!cubeImage_o.has_value()){
        throw std::runtime_error("StandardDeconvolutionExecutor: No input image recieved from reader");
    }
    PaddedImage& cubeImage = *cubeImage_o;

    ComplexData g_host = Preprocessor::convertImageToComplexData(cubeImage.image);
    ComplexData g_device = iobackend.getMemoryManager().copyDataToDevice(g_host);
    BackendFactory::getInstance().getDefaultBackendMemoryManager().freeMemoryOnDevice(g_host);
    ComplexData f_device = iobackend.getMemoryManager().allocateMemoryOnDevice(workShape);
    std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();
    algorithm->setProgressTracker(tracker);

    std::future<void> resultDone = context->processor.deconvolveSingleCube(
        workerbackend,
        std::move(algorithm),
        workShape,
        task.psfs,
        g_device,
        f_device,
        *context->psfpreprocessor.get());

    resultDone.get(); //wait for result
    iobackend.sync();

    ComplexData f_host = iobackend.getMemoryManager().moveDataFromDevice(f_device, BackendFactory::getInstance().getDefaultBackendMemoryManager());

    cubeImage.image = Preprocessor::convertComplexDataToImage(f_host);

    writer->setSubimage(cubeImage.image, task.paddedBox);
}


std::function<void()> StandardDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc) {
    
    return [this, task = *taskDesc]() {

        TaskContext* context = task.context.get();
        try {
            runTask(task);
        }
        catch (const dolphin::backend::MemoryException& e){
            // log the exception,  then enqueue the task in another thread, while this thread simply waits for the result
            // This effectively removes this thread from the pool until the other thread is done. Then just reduce NumberThreads(1)
            // will remove the first thread that finishes a task (probably this one as its basically ) 
            spdlog::get("deconvolution")->warn("{} reducing number of threads and copies of subimages", e.getDetailedMessage());            
            //TODO reduce number of workerthreads aswell
            bool noMoreWorkers = context->ioPool.reduceActiveWorkers(1); // marks self as waiting 

            if (noMoreWorkers) throw std::runtime_error("Can't fit a single cube for deconvolution onto the device");
            context->ioPool.enqueue(createTask(std::make_unique<CubeTaskDescriptor>(task))).get();
            bool maxReached = context->ioPool.reduceNumberThreads(1);
            if (maxReached) throw std::runtime_error("Can't fit a single cube for deconvolution onto the device");
        }
        catch (const dolphin::backend::BackendException& e) {
            spdlog::get("deconvolution")->error(e.getDetailedMessage());
            throw std::runtime_error(e.what()); // dont overwrite image if exception
        }
    };
}



void StandardDeconvolutionExecutor::parallelDeconvolution(
    const DeconvolutionPlan& channelPlan) {

    std::vector<std::future<void>> runningTasks;
    loadingBar.setMax(channelPlan.totalTasks);

    for (const std::unique_ptr<CubeTaskDescriptor>& task : channelPlan.tasks) {


        std::function<void()> threadtask = createTask(task);
        runningTasks.push_back(task->context->ioPool.enqueue(threadtask));
    }

    // Wait for all remaining tasks to finish

    for (auto& f : runningTasks)
        f.get();
}




