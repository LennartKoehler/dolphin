#pragma once
#include "IDeconvolutionExecutor.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "deconvolution/Preprocessor.h"
#include "ComputationalPlan.h"
#include "deconvolution/DeconvolutionProcessor.h"
#include "IO/TiffReader.h"
#include "IO/TiffWriter.h"
#include "backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"

class SetupConfig;

class StandardDeconvolutionExecutor : public IDeconvolutionExecutor {
public:
    StandardDeconvolutionExecutor();
    virtual ~StandardDeconvolutionExecutor();

    // IDeconvolutionExecutor interface
    virtual void execute(const ChannelPlan& plan) override;
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) override;
    virtual void configure(const SetupConfig& setupConfig);

protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc);
    
    // virtual PaddedImage preprocessChannel(Channel& image, const ChannelPlan& channelPlan);
    virtual ComplexData convertCVMatVectorToFFTWComplex(const Image3D& input, const RectangleShape& shape);
    virtual Image3D convertFFTWComplexToCVMatVector(const ComplexData& input);

    // Parallel execution
    virtual void parallelDeconvolution(
        const ChannelPlan& channelPlan);

protected:
    std::shared_ptr<IBackendMemoryManager> cpuMemoryManager;
    PSFPreprocessor psfPreprocessor;

    int numberIOThreads;
    int numberWorkerThreads;
    int numberDevices; // for devices with no shared memory
    LoadingBar loadingBar;
    bool configured = false;
};