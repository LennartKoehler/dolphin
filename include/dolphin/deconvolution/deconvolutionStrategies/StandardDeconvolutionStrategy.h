#pragma once
#include "IDeconvolutionStrategy.h"
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

// Forward declaration
class SetupConfig;

class StandardDeconvolutionStrategy : public IDeconvolutionStrategy {
public:
    StandardDeconvolutionStrategy() = default;
    virtual ~StandardDeconvolutionStrategy() = default;

    // IDeconvolutionStrategy interface
    virtual void configure(const SetupConfig& setupConfig) override;
    
    virtual ChannelPlan createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) override;


protected:
    // Helper methods for plan creation
    virtual size_t maxMemoryPerCube(
        size_t maxNumberThreads, 
        size_t maxMemory,
        const DeconvolutionAlgorithm* algorithm);


    std::shared_ptr<DeconvolutionAlgorithm> getAlgorithm(const DeconvolutionConfig& config);
    std::shared_ptr<IBackend> getBackend(const SetupConfig& config);
    
    virtual size_t estimateMemoryUsage(
        const RectangleShape& cubeSize,
        const DeconvolutionAlgorithm* algorithm);
    
    virtual RectangleShape getCubeShape(
        size_t memoryPerCube,
        size_t numberThreads,
        const RectangleShape& imageOriginalShape,
        const Padding& cubePadding);

    virtual Padding getImagePadding(
        const RectangleShape& imageSize,
        const RectangleShape& cubeSizeUnpadded,
        const Padding& cubePadding
    );

    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(std::shared_ptr<IBackend> backend, const SetupConfig& config) const ;

    virtual Padding getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs);


};