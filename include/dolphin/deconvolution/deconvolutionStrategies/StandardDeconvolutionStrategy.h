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
#include "dolphin/ProgressTracking.h"
#include "dolphin/deconvolution/deconvolutionStrategies/IDeconvolutionStrategy.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphinbackend/IBackendManager.h"
#include <functional>



/// Type of the FFT-workspace estimator supplied by a backend.
/// Takes a candidate cube shape and returns the estimated workspace in bytes.
using FFTWorkspaceCopiesEstimator= std::function<float(const CuboidShape&)>;


class StandardDeconvolutionStrategy : public IDeconvolutionStrategy {
public:
    StandardDeconvolutionStrategy() = default;
    virtual ~StandardDeconvolutionStrategy() = default;


    virtual Result<DeconvolutionPlan> createPlan(
        std::shared_ptr<ReaderHandler> reader,
        std::shared_ptr<ImageWriter> writer,
        PSFHandler& psfHandler,
        const DeconvolutionConfig& config,
        const SetupConfig& setupConfig) override;


protected:
    virtual size_t getMaxMemoryPerCube(
        size_t ioThreads,
        size_t workerThreads,
        const IBackendMemoryManager& backend,
        std::shared_ptr<DeconvolutionAlgorithm> algorithm);

    /// Pure memory-budget calculation with no backend dependency.
    /// Given the device's available memory (already minus any safety buffer the
    /// caller wishes to apply) and a function to estimate the FFT workspace for
    /// a candidate cube, returns the maximum bytes usable for a single cube.
    static size_t computeMaxMemoryPerCube(
        size_t availableMemory,
        size_t ioThreads,
        size_t workerThreads,
        size_t algorithmMemoryMultiplier,
        const FFTWorkspaceCopiesEstimator& estimateFFTWorkspace);

    std::shared_ptr<DeconvolutionAlgorithm> getAlgorithm(const DeconvolutionConfig& config);
    IBackendManager& getBackendManager(const SetupConfig& config);

    virtual size_t estimateMemoryUsage(
        const CuboidShape& cubeSize,
        const DeconvolutionAlgorithm* algorithm,
        const SetupConfig& config
    );



    virtual std::vector<std::shared_ptr<TaskContext>> createContexts(
        IBackendManager& manager,
        PSFHandler& psfHandler,
        int nDevices,
        size_t& nWorkerThreads,
        size_t& nIOThreads,
        size_t& totalThreads) const ;

};
