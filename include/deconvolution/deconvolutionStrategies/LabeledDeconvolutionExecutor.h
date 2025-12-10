#pragma once
#include "StandardDeconvolutionExecutor.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "../../HyperstackImage.h"
#include "../../psf/PSF.h"
#include "../DeconvolutionAlgorithmFactory.h"
#include "../algorithms/DeconvolutionAlgorithm.h"
#include "../../ThreadPool.h"
#include "../Preprocessor.h"
#include "ComputationalPlan.h"
#include "../DeconvolutionProcessor.h"
#include "../../IO/TiffReader.h"
#include "../../IO/TiffWriter.h"
#include "../../backend/BackendFactory.h"
#include "../../backend/IBackend.h"
#include "../../backend/IBackendMemoryManager.h"

class LabeledDeconvolutionExecutor : public StandardDeconvolutionExecutor {
public:
    LabeledDeconvolutionExecutor();
    virtual ~LabeledDeconvolutionExecutor() = default;





protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
        const ImageReader& reader,
        const ImageWriter& writer);



    virtual std::vector<Label> getLabelGroups(int channelNumber, const BoxCoord& roi, std::vector<std::shared_ptr<PSF>>& psfs);

protected:
    std::unique_ptr<DeconvolutionConfig> config;
    std::shared_ptr<IBackendMemoryManager> cpuMemoryManager;
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;
    PSFPreprocessor psfPreprocessor;
    DeconvolutionProcessor processor;
    std::shared_ptr<ThreadPool> readwriterPool;
    size_t numberThreads;
    LoadingBar loadingBar;
    std::shared_ptr<Hyperstack> labelImage;
    RangeMap<std::string> psfLabelMap;
    std::vector<std::shared_ptr<PSF>> psfs;
    bool configured = false;
};