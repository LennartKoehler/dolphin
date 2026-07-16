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

#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"



std::future<void> DeconvolutionProcessor::deconvolveSingleCube(
    IBackend& threadbackend,
    std::shared_ptr<DeconvolutionAlgorithm> prototypealgorithm,
    const CuboidShape& workShape,
    const std::vector<std::shared_ptr<PSF>>& psfs_host, // dont pass psfs as ComplexData, because the workerbackend might be needed to preprocess psfs
    RealData& g_device,
    RealData& f_device,
    PSFPreprocessor& psfPreprocessor,
    std::function<void(int)> progressFunction){

    // on workerThread
    std::future<void> resultDone = workerPool->enqueue([
        this,
        &threadbackend,
        prototypealgorithm,
        &psfs_host,
        &workShape,
        &g_device,
        &f_device,
        &psfPreprocessor,
        progressFunction
    ]() mutable {

        //dont allocate and deallocate the helpers of the algorithm every time, just overwrite buffer every time its used
        // the worker owns the algorithm, and doesnt always need to re initialize
        thread_local std::unique_ptr<DeconvolutionAlgorithm> workeralgorithm = prototypealgorithm->clone();
        thread_local bool initialized = [&threadbackend, progressFunction, workShape](){
            workeralgorithm->setBackend(threadbackend);
            workeralgorithm->init(workShape);
            workeralgorithm->setProgressTracker(progressFunction);
            return true;}();

        std::vector<const ComplexData*> preprocessedPSFs;

        for (auto& psf : psfs_host){
            preprocessedPSFs.push_back(psfPreprocessor.getPreprocessedPSF(workShape, psf, threadbackend));
            threadbackend.sync();
        }

        for (const auto* psf_device : preprocessedPSFs){
            workeralgorithm->deconvolve(*psf_device, g_device, f_device);
            threadbackend.sync();
        }


    });
    return resultDone;
}
