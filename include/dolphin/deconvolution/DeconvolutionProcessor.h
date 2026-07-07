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
#include <future>
#include <vector>
#include <memory>
#include "dolphinbackend/ComplexData.h"
#include "dolphin/ThreadPool.h"

class TaskContext;
class CuboidShape;
class IBackend;
class IBackendMemoryManager;
class DeconvolutionAlgorithm;
class PSF;
class PSFPreprocessor;

class DeconvolutionProcessor{
public:
    DeconvolutionProcessor() = default;

    void init(size_t numberThreads, std::function<void()> threadInitFunc = [](){}){
        workerPool = std::make_unique<ThreadPool>(numberThreads, threadInitFunc);
    }

    std::future<void> deconvolveSingleCube(
        std::shared_ptr<TaskContext>& context,
        IBackend& iobackend,
        std::shared_ptr<DeconvolutionAlgorithm> prototypealgorithm,
        const CuboidShape& workShape,
        const std::vector<std::shared_ptr<PSF>>& psfs_host, // dont pass psfs as ComplexData, because the workerbackend might be needed to preprocess psfs
        RealData& g_device,
        RealData& f_device,
        std::function<void(int)> progressFunction);

    // static ComplexData staticDeconvolveSingleCube(
    //     IBackend& backend,
    //     std::unique_ptr<DeconvolutionAlgorithm> algorithm,
    //     const CuboidShape& workShape,
    //     const std::vector<std::shared_ptr<PSF>>& psfs_host,
    //     ComplexData& g_device,
    //     ComplexData& f_device,
    //     PSFPreprocessor& psfpreprocessor);
    //
    // static ComplexData staticDeconvolveSingleCubeWithCopying(
    //     IBackend& backend,
    //     std::shared_ptr<IBackendMemoryManager> hostbackend,
    //     std::unique_ptr<DeconvolutionAlgorithm> algorithm,
    //     const CuboidShape& workShape,
    //     const std::vector<std::shared_ptr<PSF>>& psfs_host,
    //     ComplexData& g_host,
    //     PSFPreprocessor& psfpreprocessor);



private:

    std::unique_ptr<ThreadPool> workerPool;

};



