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
#include <mutex>
#include <unordered_map>
#include "dolphinbackend/ComplexData.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"

struct CuboidShape;
class IBackend;
class DeconvolutionAlgorithm;
class PSF;
class PSFPreprocessor;

class DeconvolutionProcessor{
public:
    DeconvolutionProcessor() = default;

    void init(size_t numberThreads, std::function<void()> threadInitFunc = [](){}){
        workerPool = std::make_unique<ThreadPool>(numberThreads, threadInitFunc);
    }

    /**
     * @brief Create (or reuse) the worker algorithm for `backend` and run its
     *        init()/preAllocate() phases. SYNCHRONOUS ON THE CALLING THREAD.
     *
     * Call this from the io thread BEFORE allocating the io image buffers. The
     * FFT plans are created during init(), and plan creation needs a transient
     * backend-side work area; running it here keeps that transient from
     * overlapping the image buffers, which lowers the peak footprint.
     *
     * The state is cached per backend (not per thread): it doesn't matter which
     * pool thread executes the following deconvolveSingleCube(), and a backend
     * seen again with the same work shape reuses its algorithm without re-init.
     *
     * @param backend     the backend the algorithm runs on (pass the WORKER
     *                    backend, not the io backend: on CPU the two differ in
     *                    their per-op threading, and FFT plans are keyed by it).
     * @param prototype   algorithm to clone; only used on first use / shape change.
     * @param workShape   padded working shape the algorithm is sized for.
     */
    void initWorker(
        IBackend& backend,
        const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
        const CuboidShape& workShape);

    std::future<void> deconvolveSingleCube(
        IBackend& backend,
        const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
        const CuboidShape& workShape,
        const std::vector<std::shared_ptr<PSF>>& psfs_host,
        RealData& g_device,
        RealData& f_device,
        PSFPreprocessor& psfpreprocessor,
        std::function<void(int)> progressFunction,
        const DeconvolutionRunContext& runContext = {});

private:
    // One algorithm instance per backend, sized for one work shape.
    //
    // Lives in a node-based map (unordered_map), so a WorkerState's address is
    // stable across rehash and its mutex may be held while the map changes.
    // WorkerState is intentionally non-copyable/non-movable by the mutex member
    // (unordered_map never moves values, only relinks nodes).
    //
    // The entry mutex serializes init/re-init of that backend only. Using the
    // returned algorithm for deconvolve() happens WITHOUT the entry lock; that
    // is safe because deconvolve() calls for one backend are serialized by the
    // worker pool (nWorkerThreads == 1 per TaskContext) and re-init cannot run
    // concurrently with a deconvolve of the same shape (initWorker runs first,
    // synchronously, on the io thread).
    struct WorkerState {
        std::mutex mutex;                  // guards THIS entry's init/re-init
        std::unique_ptr<DeconvolutionAlgorithm> algorithm;
        CuboidShape shape{};
    };

    // Returns the ready-to-use algorithm for `backend`, creating it when absent
    // or when `workShape` changed.
    //
    // Locking: `statesMutex` is held only to look up/insert the entry, never
    // across init()/preAllocate(). The expensive phase runs under the ENTRY's
    // mutex instead, so different backends initialize fully in parallel while
    // the same backend is still initialized exactly once (double-checked).
    // Note: on CPU, FFTWManager still serializes plan creation system-wide
    // (it owns a global lock around planning); the parallelism won is on CUDA.
    DeconvolutionAlgorithm& ensureWorker(
        IBackend& backend,
        const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
        const CuboidShape& workShape);

    std::unique_ptr<ThreadPool> workerPool;

    std::mutex statesMutex;                             // guards `states` only
    std::unordered_map<IBackend*, WorkerState> states;

};