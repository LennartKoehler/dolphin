#pragma once
#include "IDeconvolutionExecutor.h"
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

class StandardDeconvolutionExecutor : public IDeconvolutionExecutor {
public:
    StandardDeconvolutionExecutor();
    virtual ~StandardDeconvolutionExecutor() = default;

    // IDeconvolutionExecutor interface
    virtual void execute(const ChannelPlan& plan, const ImageReader& reader, const ImageWriter& writer) override;
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) override;

protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
        const ImageReader& reader,
        const ImageWriter& writer);

    virtual PaddedImage preprocessChannel(Channel& image, const ChannelPlan& channelPlan);
    virtual void postprocessChannel(Channel& image);
    virtual PaddedImage getCubeImage(const PaddedImage& paddedImage, const BoxCoord& coords, const Padding& cubePadding);
    virtual ComplexData convertCVMatVectorToFFTWComplex(const Image3D& input, const RectangleShape& shape);
    virtual Image3D convertFFTWComplexToCVMatVector(const ComplexData& input);

    // Parallel execution
    virtual void parallelDeconvolution(
        const ChannelPlan& channelPlan,
        const ImageReader& reader,
        const ImageWriter& writer);

protected:
    std::shared_ptr<IBackendMemoryManager> cpuMemoryManager;
    ChannelPlan plan;
    PSFPreprocessor psfPreprocessor;
    DeconvolutionProcessor processor;
    std::shared_ptr<ThreadPool> readwriterPool;
    size_t numberThreads;
    LoadingBar loadingBar;
    bool configured = false;
};