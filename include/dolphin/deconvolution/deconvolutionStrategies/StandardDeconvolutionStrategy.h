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
#include "dolphinbackend/IBackendManager.h"



class StandardDeconvolutionStrategy : public IDeconvolutionStrategy {
public:
    StandardDeconvolutionStrategy() = default;
    virtual ~StandardDeconvolutionStrategy() = default;


    virtual Result<DeconvolutionPlan> createPlan(
        std::shared_ptr<ImageReader> reader,
        std::shared_ptr<ImageWriter> writer,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) override;


protected:
    virtual size_t getMaxMemoryPerCube(
        size_t ioThreads,
        size_t workerThreads,
        IBackendManager& manager,
        std::shared_ptr<DeconvolutionAlgorithm> algorithm);


    std::shared_ptr<DeconvolutionAlgorithm> getAlgorithm(const DeconvolutionConfig& config);
    IBackendManager& getBackendManager(const SetupConfig& config);

    virtual size_t estimateMemoryUsage(
        const CuboidShape& cubeSize,
        const DeconvolutionAlgorithm* algorithm,
        const SetupConfig& config
    );


    virtual std::unique_ptr<PSFPreprocessor> createPSFPreprocessor() const ;

    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(
        IBackendManager& manager,
        int nDevices,
        size_t& nWorkerThreads,
        size_t& nIOThreads,
        size_t& totalThreads) const ;

    virtual Result<std::pair<Padding, CuboidShape>> getCubePadding(
        const std::vector<PSF>& psfs,
        const CuboidShape& configPadding,
        const CuboidShape& imageSize,
        const DeconvolutionConfig& config);

};

