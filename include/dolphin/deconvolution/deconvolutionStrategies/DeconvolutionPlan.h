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
#include "dolphin_image/Image3D.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin_image/IO/TiffWriter.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin/ServiceAbstractions.h"
#include "dolphinbackend/MemoryComponent.h"

class MemoryFootprint; // composite of all MemoryComponents, see dolphinbackend/MemoryFootprint.h

/**
 * @brief Resolved memory budgets for one deconvolution run, in bytes.
 *
 * All totals are hard upper bounds, enforced at runtime where possible
 * (reader prefetch backpressure, backend allocation guards).
 */
struct MemoryBudget{
    /// Cap for all host-RAM consumers of the run (reader cache + io staging,
    /// and for the CPU backend also all "device" allocations).
    size_t hostTotal_byte = 0;
    /// Cap for accelerator consumers per device (device buffers, worker
    /// working sets, FFT workspace, PSF cache). For the CPU backend device ==
    /// host, so this equals hostTotal.
    size_t deviceTotal_byte = 0;
    /// Whether the compute "device" is host RAM (CPU backend): one shared pool.
    bool deviceIsHost = true;

    // Derived caps, set once the actual cube size is known:
    /// Reader in-flight + cache cap (own tracker with backpressure).
    size_t readerCache_byte = 0;
    /// Writer peak reserve (tiles buffered + write path, no cap mechanism -
    /// the writer reports its true maximum, the rest of the system must stay
    /// below the remainder).
    size_t writerReserve_byte = 0;
    /// Host cap for backend allocations (io staging etc.)
    /// = hostTotal - readerCache - writerReserve.
    size_t backendHost_byte = 0;

    /// Whether a demand fits this budget. For backends where device == host
    /// both demands draw from the single host pool.
    bool fits(const MemoryRequirement& need) const {
        const size_t hostNeed = need.hostBytes + (deviceIsHost ? need.deviceBytes : 0);
        if (hostNeed > hostTotal_byte) return false;
        if (!deviceIsHost && need.deviceBytes > deviceTotal_byte) return false;
        return true;
    }
};


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

struct CubeTaskSharedDescriptor{

    CubeTaskSharedDescriptor(
                    const std::shared_ptr<DeconvolutionAlgorithm>& algorithm,
                    const MemoryRequirement& estimatedMemoryUsage,
                    const std::vector<std::shared_ptr<PSF>>& psfs,
                    const std::shared_ptr<ReaderHandler> reader,
                    const std::shared_ptr<WriterHandler> writer)
    : prototypeAlgorithm(algorithm),
      estimatedMemoryUsage(estimatedMemoryUsage),
      psfs(psfs),
      reader(reader),
      writer(writer)
    {}
    const std::shared_ptr<DeconvolutionAlgorithm> prototypeAlgorithm;
    const MemoryRequirement estimatedMemoryUsage;
    const std::vector<std::shared_ptr<PSF>> psfs;
    const std::shared_ptr<ReaderHandler> reader;
    const std::shared_ptr<WriterHandler> writer;
};

struct CubeTaskDescriptor {
    CubeTaskDescriptor(int taskId,
                    const BoxCoordWithPadding& paddedBox,
                    const std::shared_ptr<CubeTaskSharedDescriptor> sharedDescriptor,
                    const std::shared_ptr<TaskContext> context)
    : taskId(taskId),
      paddedBox(paddedBox),
      sharedDescriptor(sharedDescriptor),
      context(context)
    {}


    const std::shared_ptr<CubeTaskSharedDescriptor> sharedDescriptor;
    const int taskId;
    const BoxCoordWithPadding paddedBox;
    const std::shared_ptr<TaskContext> context;
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



// --- Optimize for FFTW (Smooth Numbers) ---
// A "smooth" number has prime factors of only 2, 3, and 5.
// This ensures FFTW can use its fastest algorithms.
static bool isSmooth(size_t n){
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
};

static size_t nextSmooth(size_t dim){
    if (dim == 0) return dim;
    while (!isSmooth(dim)) {
        dim++;
    }
    return dim;
};
static size_t previousSmooth(size_t dim){
    if (dim <= 1) return dim;
    dim--;
    while (!isSmooth(dim)) {
        dim--;
    }
    return dim;
};

class DolphinImageSplitException : public std::runtime_error{
    using std::runtime_error::runtime_error;
};

Result<std::vector<BoxCoordWithPadding>> splitImageHomogeneous(
    const Padding& cubePadding,
    const Padding& imagePadding,
    const CuboidShape& imageOriginalShape,
    const MemoryBudget& budget,
    const MemoryFootprint& footprint,
    const size_t& minNumberCubes,
    const CuboidShape& minSize,
    bool sliceAllDimensions);

Result<std::vector<BoxCoordWithPadding>> splitImageHomogeneousFixedSize(
    const Padding& cubePadding,
    const Padding& imagePadding,
    const CuboidShape& imageOriginalShape,
    const CuboidShape& fixedCubeSize);

