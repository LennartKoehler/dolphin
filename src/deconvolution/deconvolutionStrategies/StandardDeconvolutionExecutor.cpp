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

std::function<void()> StandardDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc) {
    
    return [this, task = *taskDesc]() {

        TaskContext* context = task.context.get();

        thread_local std::shared_ptr<IBackend> iobackend = context->prototypebackend->onNewThreadSharedMemory(context->prototypebackend);

        std::shared_ptr<ImageReader> reader = task.reader;
        std::shared_ptr<ImageWriter> writer = task.writer;


        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

        PaddedImage cubeImage = reader->getSubimage(task.paddedBox);

        ComplexData g_host = Preprocessor::convertImageToComplexData(cubeImage.image);

        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);

        defaultBackendMemoryManager.freeMemoryOnDevice(g_host);

        ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

        ComplexData f_host{&defaultBackendMemoryManager, nullptr, RectangleShape()};
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();


        try {
            std::future<void> resultDone = context->processor.deconvolveSingleCube(
                iobackend,
                std::move(algorithm),
                workShape,
                task.psfs,
                g_device,
                f_device,
                *context->psfpreprocessor.get());

            resultDone.get(); //wait for result
            f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, defaultBackendMemoryManager);
        }
        catch (...) {
            throw; // dont overwrite image if exception
        }

        cubeImage.image = Preprocessor::convertComplexDataToImage(f_host);

        writer->setSubimage(cubeImage.image, task.paddedBox);


        loadingBar.addOne();
    };
}



void StandardDeconvolutionExecutor::parallelDeconvolution(
    const DeconvolutionPlan& channelPlan) {

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




