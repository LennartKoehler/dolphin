#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "deconvolution/Postprocessor.h"
#include <set>
#include <stdexcept>
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"
#include "deconvolution/ImageMap.h"
#include "HelperClasses.h"

LabeledDeconvolutionExecutor::LabeledDeconvolutionExecutor(){
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



std::function<void()> LabeledDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const ImageReader& reader,
    const ImageWriter& writer) {
    
    LabeledCubeTaskDescriptor* labeledTask = dynamic_cast<LabeledCubeTaskDescriptor*>(taskDesc.get());
    if (!labeledTask) {
        throw std::runtime_error("Expected LabeledCubeTaskDescriptor but got different type");
    }

    return [this, task = *labeledTask, &reader, &writer]() {
        // TODO: Implement task logic using reader and writer
        // The previous implementation used inputImagePadded and outputImage directly
        // This will need to be updated to use the reader and writer interfaces
        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;
        PaddedImage cubeImage = reader.getSubimage(task.paddedBox);
        std::vector<Label> labelgroups = getLabelGroups(task.channelNumber, task.paddedBox.box, psfs);
        std::shared_ptr<IBackend> iobackend = backend_->onNewThread();
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);
        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);

        for (const Label& labelgroup : labelgroups){
            ComplexData local_g_device = iobackend->getMemoryManager().copyData(g_device);
            ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};

            ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);
            std::vector<std::shared_ptr<PSF>> psfs = labelgroup.getPSFs();
            std::unique_ptr<DeconvolutionAlgorithm> algorithm = plan.algorithm->clone();
            if (psfs.size() != 0){
                try {
                    std::future<void> resultDone = processor.deconvolveSingleCube(
                        iobackend,
                        std::move(algorithm),
                        workShape,
                        psfs,
                        local_g_device,
                        f_device,
                        psfPreprocessor);

                    resultDone.get(); //wait for result
                    f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
                }
                catch (...) {
                    throw; // dont overwrite image if exception
                }
                PaddedImage resultCube;
                resultCube.padding = cubeImage.padding;
                resultCube.image = convertFFTWComplexToCVMatVector(f_host);
                // Postprocessor::insertLabeledCubeInImage(resultCube, outputImage, task.srcBox, labelgroup);
            }
         }

        iobackend->releaseBackend();
        loadingBar.addOne();
    };
}




