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

#include "deconvolution/DeconvolutionProcessor.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "deconvolution/Preprocessor.h"




std::future<void> DeconvolutionProcessor::deconvolveSingleCube(
    std::shared_ptr<IBackend> prototypebackend,
    std::unique_ptr<DeconvolutionAlgorithm> algorithm, 
    const RectangleShape& workShape,
    const std::vector<std::shared_ptr<PSF>>& psfs_host, // dont pass psfs as ComplexData, because the workerbackend might be needed to preprocess psfs
    ComplexData& g_device,
    ComplexData& f_device,
    PSFPreprocessor& psfPreprocessor){

    // on workerThread
    std::future<void> resultDone = workerPool->enqueue([
        this,
        &prototypebackend,
        algorithm = std::move(algorithm),
        &psfs_host,
        &workShape,
        &g_device,
        &f_device,
        &psfPreprocessor
    ]() mutable {

        std::shared_ptr<IBackend> threadbackend = prototypebackend->onNewThread(prototypebackend);
        
        algorithm->setBackend(threadbackend);
        algorithm->init(workShape);
        std::vector<const ComplexData*> preprocessedPSFs;

        for (auto& psf : psfs_host){

            preprocessedPSFs.push_back(psfPreprocessor.getPreprocessedPSF(workShape, psf, threadbackend));
            threadbackend->sync();
        
        }
        for (const auto* psf_device : preprocessedPSFs){
            algorithm->deconvolve(*psf_device, g_device, f_device);
            // threadbackend->getDeconvManager().scalarMultiplication(f_device, 1.0 / f_device.size.volume, f_device); // Add normalization
            threadbackend->sync();
        }
        
        threadbackend->releaseBackend();


    });
    return resultDone;
}




