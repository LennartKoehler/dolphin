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


DeconvolutionAlgorithm& DeconvolutionProcessor::ensureWorker(
    IBackend& backend,
    const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
    const CuboidShape& workShape){

    // Resolve the per-backend entry. The map lock is held for the lookup/insert
    // only; the returned reference stays valid afterwards because unordered_map
    // is node-based (rehash relinks nodes, it does not move values).
    WorkerState& state = [&]() -> WorkerState& {
        std::lock_guard<std::mutex> lock(statesMutex);
        return states[&backend];
    }();

    // Serialize only against THIS backend: one init per backend, but different
    // backends (devices) may initialize concurrently.
    std::lock_guard<std::mutex> entryLock(state.mutex);

    if (!state.algorithm || state.shape != workShape) {
        // init(): FFT plans (transient work area, caller wants this before its
        // own image buffers exist), then preAllocate(): persistent scratch.
        state.algorithm = prototype->clone();
        state.algorithm->setBackend(backend);
        state.algorithm->init(workShape);
        state.algorithm->preAllocate(workShape);
        state.shape = workShape;
    }
    return *state.algorithm;
}

void DeconvolutionProcessor::initWorker(
    IBackend& backend,
    const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
    const CuboidShape& workShape){
    // synchronous on purpose: guarantees the plan-creation transient is done
    // before the caller allocates its buffers, and that any later
    // deconvolveSingleCube() on any pool thread finds a ready algorithm.
    ensureWorker(backend, prototype, workShape);
}

std::future<void> DeconvolutionProcessor::deconvolveSingleCube(
    IBackend& backend,
    const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
    const CuboidShape& workShape,
    const std::vector<std::shared_ptr<PSF>>& psfs_host,
    RealData& g_device,
    RealData& f_device,
    PSFPreprocessor& psfPreprocessor,
    std::function<void(int)> progressFunction,
    const DeconvolutionRunContext& runContext){

    // on workerThread
    std::future<void> resultDone = workerPool->enqueue([
        this,
        &backend,
        prototype,
        &workShape,
        &psfs_host,
        &g_device,
        &f_device,
        &psfPreprocessor,
        progressFunction,
        runContext
    ]() mutable {

        // Hit path: initWorker() already ran on the io thread for this backend +
        // shape, so this only looks the algorithm up. (If a caller skipped
        // initWorker, this still initializes - just with the transient inside
        // the task again.)
        DeconvolutionAlgorithm& workeralgorithm =
            ensureWorker(backend, prototype, workShape);
        // The tracker is per call (it differs e.g. between label groups), so
        // it is refreshed here, not only at creation in ensureWorker().
        workeralgorithm.setProgressTracker(progressFunction);

        std::vector<const ComplexData*> preprocessedPSFs;

        for (auto& psf : psfs_host){
            preprocessedPSFs.push_back(psfPreprocessor.getPreprocessedPSF(workShape, psf, backend));
            backend.sync();
        }

        for (const auto* psf_device : preprocessedPSFs){
            workeralgorithm.deconvolve(*psf_device, g_device, f_device, runContext);
            backend.sync();
        }


    });
    return resultDone;
}
