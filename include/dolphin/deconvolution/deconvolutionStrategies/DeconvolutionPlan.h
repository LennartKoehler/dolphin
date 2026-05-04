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

#pragma once
#include <memory>
#include <vector>
#include <spdlog/spdlog.h>
#include <atomic>

#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/Image3D.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/ServiceAbstractions.h"


/*
The context for a specific CubeTaskDescription. These can be shared between tasks
Currently a TaskContext is created for each Device. So each device has its own workers, psfpreprocessor and ioThreads
*/
struct TaskContext{
    TaskContext(
        IBackendManager& manager,
        BackendConfig ioconfig,
        BackendConfig workerconfig,
        int nWorkerThreads,
        int nIOThreads
    ) :     manager(manager),
            ioconfig(ioconfig),
            workerconfig(workerconfig),
            processor(),
          ioPool(nIOThreads)
    {
        processor.init(nWorkerThreads);
    }
    ~TaskContext(){
        psfpreprocessor->cleanup();
    }
    void setPreprocessor(std::unique_ptr<PSFPreprocessor> preprocessor){
        psfpreprocessor = std::move(preprocessor);
    }
    BackendConfig ioconfig;
    BackendConfig workerconfig;
    IBackendManager& manager;
    DeconvolutionProcessor processor;
    ThreadPool ioPool;
    std::unique_ptr<PSFPreprocessor> psfpreprocessor;
};



/*
This Description is the blueprint of how the specified part of the image is to be deconvolved.
*/
struct CubeTaskDescriptor {
        CubeTaskDescriptor(int taskId,
                        const BoxCoordWithPadding& paddedBox,
                        const std::shared_ptr<DeconvolutionAlgorithm>& algorithm,
                        size_t estimatedMemoryUsage,
                        const std::vector<std::shared_ptr<PSF>>& psfs,
                        const std::shared_ptr<ImageReader> reader,
                        const std::shared_ptr<ImageWriter> writer,
                        std::shared_ptr<TaskContext> context)
        : taskId(taskId),
          paddedBox(paddedBox),
          algorithm(algorithm),
          estimatedMemoryUsage(estimatedMemoryUsage),
          psfs(psfs),
          reader(reader),
          writer(writer),
          context(context)
    {}


    const int taskId;
    const BoxCoordWithPadding paddedBox;
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm;
    const size_t estimatedMemoryUsage;
    const std::vector<std::shared_ptr<PSF>> psfs;
    const std::shared_ptr<ImageReader> reader;
    const std::shared_ptr<ImageWriter> writer;
    std::shared_ptr<TaskContext> context;
};




struct DeconvolutionPlan {
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    size_t totalTasks;
};

template<typename T>
class Label{
public:
    Label(T&& mask, std::vector<std::shared_ptr<PSF>> psfs) : weightedMask(std::make_unique<T>(std::move(mask))), psfs(psfs){}

    const T* getMask() const { return weightedMask.get();}

    T* getMask() { return weightedMask.get();}

    std::vector<std::shared_ptr<PSF>> getPSFs() const {
        return psfs;
    }

private:
    // Range<std::shared_ptr<PSF>> psfs;
    std::vector<std::shared_ptr<PSF>> psfs;
    // Image3D* labelImage;
    std::unique_ptr<T> weightedMask;


};


//
// class LabelBuilder{
// public:
//
//     LabelBuilder(std::vector<std::string> psfIds, Image3D&& mask) : psfIDs(psfIds), unprocessedMask(mask){}
//     Image3D& accessImage() {return unprocessedMask;}
//
//     void setMaskConverterFunction(std::function<RealData&&(Image3D&)> maskConverter){
//         maskConverterFn = maskConverter;
//     }
//
//     void setPSFConverterFunction(std::function<std::vector<std::shared_ptr<PSF>>(std::vector<std::string>)> psfConverter){
//         psfConverterFn = psfConverter;
//     }
//
//     Label build(){
//         assert(psfConverterFn && maskConverterFn);
//         return Label{
//             maskConverterFn(unprocessedMask),
//             psfConverterFn(psfIDs)};
//     }
//
// private:
//     std::vector<std::string> psfIDs;
//     Image3D unprocessedMask;
//
//     std::function<std::vector<std::shared_ptr<PSF>>(std::vector<std::string>)> psfConverterFn;
//     std::function<RealData&&(Image3D&)> maskConverterFn;
//
// };
//

class PaddingStrategy{
public:
    virtual Padding getPadding(const std::vector<PSF>& psfs, const CuboidShape& imageShape, const DeconvolutionConfig& config) const = 0;
};
class ParentPaddingStrategy : public PaddingStrategy{
public:
    Padding getPadding(const std::vector<PSF>& psfs, const CuboidShape& imageShape, const DeconvolutionConfig& config) const override {

        std::vector<CuboidShape> psfSizes = getShapes<PSF>(psfs);
        CuboidShape largestPSF = getLargestShape(psfSizes);

        PSF psf = psfs[0]; // TODO for multiple
        float threshold = config.paddingRelativeMax * psf.getMax(); // pad up until values drop below 0.01% of max value (their influence is negligable)
        CuboidShape paddingRegion = psf.getRegionLargerThreshold(threshold);
        paddingRegion.setMin(largestPSF - imageShape); // always pad atleast to the size of the psf. This is only relevant if psf is loaded as file

        CuboidShape paddingbefore = paddingRegion / 2;
        return Padding{paddingbefore, paddingRegion - paddingbefore};
    }
};


class FullPSFPaddingStrategy : public PaddingStrategy{
public:
    Padding getPadding(const std::vector<PSF>& psfs, const CuboidShape& imageShape, const DeconvolutionConfig& config) const override {

        std::vector<CuboidShape> psfSizes = getShapes<PSF>(psfs);
        CuboidShape largestPSF = getLargestShape(psfSizes);

        CuboidShape paddingRegion = largestPSF;

        CuboidShape paddingbefore = paddingRegion / 2;
        return Padding{paddingbefore, paddingRegion - paddingbefore};
    }
};

class ManualPaddingStrategy : public PaddingStrategy{
public:
    Padding getPadding(const std::vector<PSF>& psfs, const CuboidShape& imageShape, const DeconvolutionConfig& config) const override{
        CuboidShape paddingHalf = imageShape / 2;
        return Padding(paddingHalf, imageShape - paddingHalf);

    }
};

Result<std::vector<BoxCoordWithPadding>> splitImageHomogeneous(
    const Padding& cubePadding,
    const CuboidShape& imageOriginalShape,
    const size_t& maxVolumePerCube,
    const size_t& minNumberCubes,
    const PaddingStrategyType& imagePadding,
    const CuboidShape& minShape);


