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
#include "dolphin/deconvolution/deconvolutionStrategies/IDeconvolutionStrategy.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"


class SetupConfig;

class StandardDeconvolutionStrategy : public IDeconvolutionStrategy {
public:
    StandardDeconvolutionStrategy() = default;
    virtual ~StandardDeconvolutionStrategy() = default;

    // IDeconvolutionStrategy interface
    virtual void configure(const SetupConfig& setupConfig) override;
    
    virtual Result<DeconvolutionPlan> createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) override;


protected:
    // Helper methods for plan creation
    virtual size_t getMaxMemoryPerCube(
        size_t ioThreads,
        size_t workerThreads,
        std::shared_ptr<IBackend> backend,
        std::shared_ptr<DeconvolutionAlgorithm> algorithm);


    std::shared_ptr<DeconvolutionAlgorithm> getAlgorithm(const DeconvolutionConfig& config);
    std::shared_ptr<IBackend> getBackend(const SetupConfig& config);
    
    virtual size_t estimateMemoryUsage(
        const CuboidShape& cubeSize,
        const DeconvolutionAlgorithm* algorithm,
        const SetupConfig& config
    );

    std::vector<CuboidShape> getPSFSizes(const std::vector<PSF>& psfs);
    
    virtual Result<CuboidShape> getCubeShape(
        size_t memoryPerCube,
        const CuboidShape& configCubeSize,
        const CuboidShape& imageOriginalShape,
        const Padding& cubePadding,
        size_t nWorkerThreads
    );

    virtual Padding getImagePadding(
        const CuboidShape& imageSize,
        const CuboidShape& cubeSizeUnpadded,
        const Padding& cubePadding
    );

    virtual std::unique_ptr<PSFPreprocessor> createPSFPreprocessor() const ;

    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(
        std::shared_ptr<IBackend> backend,
        const int nDevices,
        const size_t nWorkerThreads,
        const size_t nIOThreads) const ;

    virtual Result<Padding> getCubePadding(const std::vector<PSF> psfs, const CuboidShape& configPadding);

    void configureThreads(
        size_t& totalThreads,
        size_t& ioThreads,
        size_t& workerThreads,
        std::shared_ptr<IBackend> backend,
        const SetupConfig& config
    );

    virtual int getNextPowerOfTwo(int v) const ;


};