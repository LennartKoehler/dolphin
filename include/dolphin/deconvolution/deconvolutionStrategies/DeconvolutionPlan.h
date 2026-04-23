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
class Label{
public:
    Label() = default;

    void setRange(Range<std::shared_ptr<PSF>> psfs) {this->psfs = psfs;}
    void setMask(RealData&& mask) { this->weightedMask = std::make_unique<RealData>(std::move(mask));}

    const RealData* getMask() const { return weightedMask.get();}

    RealData* getMask() { return weightedMask.get();}


    Image3D getMask(const Image3D& labelImage) const {
        return labelImage.getInRange(psfs.start, psfs.end);
    }

    std::vector<std::shared_ptr<PSF>> getPSFs() const {
        return psfs.values;
    }

private:
    Range<std::shared_ptr<PSF>> psfs;
    // Image3D* labelImage;
    std::unique_ptr<RealData> weightedMask;


};



class LoadingBar{
public:
    LoadingBar() = default;
    LoadingBar(float max) : max(max){}
    void setMax(float max) {this->max = max;}
    void reset() {counter.store(0);}
    void update(){
        // Calculate progress

        float barWidth = 50;
        int pos = static_cast<int>((counter * barWidth) / max);
        int progress = static_cast<int>((counter * 100) / max);
        // Print progress bar
        std::cerr << "\rDeconvoluting Image [ ";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cerr << "=";
            else if (i == pos) std::cerr << ">";
            else std::cerr << " ";
        }
        std::cerr << "] "
          << std::setw(3)
          << progress << "%";
        std::cerr.flush();

    }


    void add(float value){
        counter += value;
        if(mutex.try_lock()) {update(); mutex.unlock();}
    }
private:
    float max;
    std::atomic<float> counter{0};
    std::mutex mutex;
};





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
