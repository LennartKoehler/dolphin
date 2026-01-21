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
#include "IDeconvolutionStrategy.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "deconvolution/Preprocessor.h"
#include "DeconvolutionPlan.h"
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
    
    virtual DeconvolutionPlan createPlan(
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

    virtual std::unique_ptr<PSFPreprocessor> createPSFPreprocessor() const ;

    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(
        std::shared_ptr<IBackend> backend,
		const SetupConfig& config) const ;

    virtual Padding getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs);


};