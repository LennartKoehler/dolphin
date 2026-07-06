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
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphin_image/HelperClasses.h"






StandardDeconvolutionExecutor::StandardDeconvolutionExecutor(){
}

StandardDeconvolutionExecutor::~StandardDeconvolutionExecutor(){
}

void StandardDeconvolutionExecutor::execute(DeconvolutionPlan plan) {
    parallelDeconvolution(std::move(plan));
}

void StandardDeconvolutionExecutor::configure(const SetupConfig& setupConfig, const DeconvolutionConfig& deconvConfig, progressCallbackFn fn) {
    spdlog::get("deconvolution")->debug("Configuring StandardDeconvolutionExecutor");
    loadingBar.setCallback(fn);
}


void StandardDeconvolutionExecutor::runTask(const CubeTaskDescriptor& task){

    spdlog::get("deconvolution")->debug("[Task {}] Starting deconvolution of cube ({}). Padding before: {}; padding after: {}",
        task.taskId, task.paddedBox.box.print(), task.paddedBox.padding.before.print(), task.paddedBox.padding.after.print());

    std::shared_ptr<TaskContext> context = task.context;
    // thread_local IBackend& iodevice = context->iodevice.cloneSharedMemory();
    thread_local IBackend& iobackend = context->manager.createBackendForCurrentThread(context->ioconfig);
    thread_local IBackend& workerbackend = context->manager.createBackendSharedMemoryForCurrentThread(iobackend, context->workerconfig); // copied in deconvolutionprocessor

    std::shared_ptr<ImageReader> reader = task.sharedDescriptor->reader;
    std::shared_ptr<ImageWriter> writer = task.sharedDescriptor->writer;

    CuboidShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;
    spdlog::get("deconvolution")->debug("[Task {}] Work shape (with padding): {}", task.taskId, workShape.print());


    RealData g_host;

    {
        spdlog::get("deconvolution")->debug("[Task {}] Reading subimage from input", task.taskId);
        std::optional<PaddedImage> cubeImage_o = reader->getSubimage(task.paddedBox);
        if (!cubeImage_o.has_value()){
            throw std::runtime_error("StandardDeconvolutionExecutor: No input image recieved from reader");
        }
        PaddedImage& cubeImage = *cubeImage_o;
        spdlog::get("deconvolution")->debug("[Task {}] Converting input image to real data", task.taskId);
        g_host = Preprocessor::convertImageToRealData(cubeImage.image);
    }

    spdlog::get("deconvolution")->debug("[Task {}] Copying input data to device", task.taskId);
    RealData g_device = iobackend.getMemoryManager().copyDataToDevice(g_host);
    BackendFactory::getInstance().getDefaultBackendMemoryManager().freeMemoryOnDevice(g_host);
    spdlog::get("deconvolution")->debug("[Task {}] Allocating output buffer on device", task.taskId);
    RealData f_device = iobackend.getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);

    using progressFunction = std::function<void(int)>;
    progressFunction tracker = [this, numPsfs = task.sharedDescriptor->psfs.size()](int max){
        float iteration = 1.0 / (max * numPsfs);
        this->loadingBar.add(iteration);
    };

    spdlog::get("deconvolution")->info("[Task {}] Starting deconvolution with {} PSF(s)", task.taskId, task.sharedDescriptor->psfs.size());
    std::future<void> resultDone = context->processor.deconvolveSingleCube(
        workerbackend,
        task.sharedDescriptor->prototypeAlgorithm,
        workShape,
        task.sharedDescriptor->psfs,
        g_device,
        f_device,
        *context->psfpreprocessor.get(),
        tracker);

    resultDone.get(); //wait for result
    spdlog::get("deconvolution")->debug("[Task {}] Deconvolution finished, syncing backend", task.taskId);
    iobackend.sync();

    spdlog::get("deconvolution")->debug("[Task {}] Moving result data from device to host", task.taskId);
    RealData f_host = iobackend.getMemoryManager().moveDataFromDevice(f_device, BackendFactory::getInstance().getDefaultBackendMemoryManager());

    spdlog::get("deconvolution")->debug("[Task {}] Converting result to image", task.taskId);
    Image3D resultImage = Preprocessor::convertRealDataToImage(f_host);

    spdlog::get("deconvolution")->debug("[Task {}] Writing result subimage", task.taskId);
    writer->setSubimage(resultImage, task.paddedBox);
    spdlog::get("deconvolution")->debug("[Task {}] Done", task.taskId);
}


std::function<void()> StandardDeconvolutionExecutor::createTask(
    CubeTaskDescriptor& taskDesc) {

    spdlog::get("deconvolution")->debug("[Task {}] Creating task for cube ({})", taskDesc.taskId, taskDesc.paddedBox.box.print());

    return [this, &taskDesc]() {

        std::shared_ptr<TaskContext> context = taskDesc.context;
        spdlog::get("deconvolution")->debug("[Task {}] Task dispatched to worker thread", taskDesc.taskId);
        try {
            runTask(taskDesc);
        }
        catch (const dolphin::backend::MemoryException& e){
            throw std::runtime_error("Not enough free memory on the backend");
            // // log the exception,  then enqueue the task in another thread, while this thread simply waits for the result
            // // This effectively removes this thread from the pool until the other thread is done. Then just reduce NumberThreads(1)
            // // will remove the first thread that finishes a task (probably this one as its basically )
            // spdlog::get("deconvolution")->warn("{} reducing number of threads and copies of subimages", e.getDetailedMessage());
            // //TODO reduce number of workerthreads aswell
            // bool noMoreWorkers = context->ioPool.reduceActiveWorkers(1); // marks self as waiting
            //
            // if (noMoreWorkers) throw std::runtime_error("Can't fit a single cube for deconvolution onto the device");
            // context->ioPool.enqueue(createTask(taskDesc));
            // bool maxReached = context->ioPool.reduceNumberThreads(1);
            // if (maxReached) throw std::runtime_error("Can't fit a single cube for deconvolution onto the device");
        }
        catch (const dolphin::backend::BackendException& e) {
            spdlog::get("deconvolution")->error(e.getDetailedMessage());
            throw std::runtime_error(e.what()); // dont overwrite image if exception
        }
    };
}



void StandardDeconvolutionExecutor::parallelDeconvolution(
    DeconvolutionPlan channelPlan) {

    spdlog::get("deconvolution")->info("Starting parallel deconvolution with {} task(s)", channelPlan.totalTasks);

    std::vector<std::future<void>> runningTasks;
    loadingBar.setMax(channelPlan.totalTasks);

    for (auto& task : channelPlan.tasks){
        std::shared_ptr<TaskContext> context = task->context;
        std::function<void()> threadtask = createTask(*task);
        runningTasks.push_back(context->ioPool.enqueue(threadtask));
    }
    spdlog::get("deconvolution")->debug("All {} task(s) enqueued, waiting for completion", channelPlan.totalTasks);

    // Wait for all remaining tasks to finish
    for (auto& f : runningTasks)
        f.get();

    spdlog::get("deconvolution")->info("Parallel deconvolution finished");
}




