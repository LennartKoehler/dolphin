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

    // create both backends here so that they both use the same backenddevice
    // then pass the workerbackend to the workerthread to use. If it were created threadlocal on the workerthread
    // a mismatch could happen where the workerthreads backend is on a different device than the iobackend
    thread_local IBackend& iobackend = context->manager.createBackendForCurrentThread(context->ioconfig);
    thread_local IBackend& workerbackend = context->manager.createBackendSharedMemoryForCurrentThread(iobackend, context->workerconfig);

    std::shared_ptr<ReaderHandler> reader = task.sharedDescriptor->reader;
    std::shared_ptr<WriterHandler> writer = task.sharedDescriptor->writer;

    CuboidShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

    // Per-io-thread scratch buffers: allocated on first use, reused for every
    // following cube of the same (run-constant) padded shape, and freed when
    // the thread exits. This removes the per-cube allocation/free churn and
    // the overlap of fresh input and output buffers.
    thread_local RealData g_device;
    thread_local RealData f_device;

    // allocate the first time its ever used (per thread)
    // in case the workshape changes then free memory and allocate new size
    if (!g_device.isValid() || g_device.getSize() != workShape) {
        iobackend.getMemoryManager().freeMemoryOnDevice(g_device);
        iobackend.getMemoryManager().freeMemoryOnDevice(f_device);
        g_device = iobackend.getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);
        f_device = iobackend.getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);

    }

    {
        // allocate and deallocate host_scratch buffer so that it doesnt overlap with the deconvolution buffers (fft plan buffers)
        RealData host_scratch_buffer = BackendFactory::getInstance().getHostBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);
        // Read the unpadded sub-cube and pad it straight into the FFT-layout
        // host buffer - no intermediate padded Image3D, no allocation.
        UnpaddedCube cube = reader->getSubimageForPadding(task.paddedBox);
        Preprocessor::convertImageToRealDataPadded(std::move(cube.image), cube.padding,
                                                   reader->getPaddingFillType(), host_scratch_buffer);

        // host -> device into the reused input buffer (direction inferred)
        iobackend.getMemoryManager().memCopy(host_scratch_buffer, g_device);
    }
    using progressFunction = std::function<void(int)>;
    progressFunction tracker = [this, numPsfs = task.sharedDescriptor->psfs.size()](int max){
        float iteration = 1.0 / (max * numPsfs);
        this->loadingBar.add(iteration);
    };

    std::future<void> resultDone = context->processor.deconvolveSingleCube(
        workerbackend,
        task.sharedDescriptor->prototypeAlgorithm,
        workShape,
        task.sharedDescriptor->psfs,
        g_device,
        f_device,
        *context->psfpreprocessor.get(),
        tracker);

    resultDone.get();
    iobackend.sync();

    {
        RealData host_scratch_buffer = BackendFactory::getInstance().getHostBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);
        // device -> host into the reused output buffer (direction inferred)
        iobackend.getMemoryManager().memCopy(f_device, host_scratch_buffer);

        // Write straight from the host buffer into the writer - no resultImage.
        writer->setSubimageReal(host_scratch_buffer, task.paddedBox);
    }
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
        // TODO make these exceptions recoverable?
        catch (const dolphin::backend::MemoryException& e){
            throw std::runtime_error("Not enough free memory on the backend" + e.getDetailedMessage());
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

