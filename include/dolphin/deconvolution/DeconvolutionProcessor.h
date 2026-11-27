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
        expectedWorkerBackends = numberThreads == 0 ? 1 : numberThreads;
        workerPool = std::make_unique<ThreadPool>(numberThreads, threadInitFunc);
    }

    /**
     * @brief Register the calling worker pool thread's own backend.
     *
     * Called from the pool's threadInitFunc, so it runs ON the worker thread:
     * each worker thread owns one backend for its whole life (created with the
     * context's worker config, whose deviceId pins the context's device). All
     * io backends of a context share that memory domain anyway, so a backend
     * per io thread would only duplicate plans, work areas and scratch.
     */
    void registerWorkerBackend(IBackend& backend);

    /// The calling worker thread's own registered backend. Worker threads only.
    IBackend& workerBackend();

    /**
     * @brief Initialize the worker algorithm on EVERY worker backend.
     *        SYNCHRONOUS ON THE CALLING (IO) THREAD.
     *
     * Call this from the io thread BEFORE allocating the io image buffers.
     * First waits until all worker pool threads have registered their backends
     * (they do so first thing at pool start, so this returns immediately in
     * practice), then runs init()/preAllocate() per backend: the FFT plans are
     * created during init(), and plan creation needs a transient backend-side
     * work area; doing all of it here keeps every transient from overlapping
     * any image buffer, which lowers the peak footprint.
     *
     * Idempotent per backend+shape: repeated calls (other io threads, later
     * cubes) skip backends already initialized for the shape.
     *
     * @param prototype   algorithm to clone; only used on first use / shape change.
     * @param workShape   padded working shape the algorithms are sized for.
     */
    void initWorker(
        const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
        const CuboidShape& workShape);

    std::future<void> deconvolveSingleCube(
        const std::shared_ptr<DeconvolutionAlgorithm>& prototype,
        const CuboidShape& workShape,
        const std::vector<std::shared_ptr<PSF>>& psfs_host,
        RealData& g_device,
        RealData& f_device,
        PSFPreprocessor& psfpreprocessor,
        std::function<void(int)> progressFunction,
        const DeconvolutionRunContext& runContext = {});

private:
    // One algorithm instance per worker backend, sized for one work shape.
    //
    // Lives in a node-based map (unordered_map), so a WorkerState's address is
    // stable across rehash and its mutex may be held while the map changes.
    // WorkerState is intentionally non-copyable/non-movable by the mutex member
    // (unordered_map never moves values, only relinks nodes).
    //
    // Each worker pool thread exclusively uses its own registered backend, so
    // one algorithm instance is never shared across threads and deconvolve()
    // needs no per-entry lock during execution. The entry mutex only
    // serializes init/re-init of that backend.
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

    // Worker backend registration: each pool thread registers exactly one
    // backend from its threadInitFunc. initWorker() waits for the full count
    // so the per-backend plan transients are guaranteed to precede the io
    // buffers. Backends are manager-owned (stable references); worker threads
    // never unregister (the deconvolution flow never shrinks the pool).
    std::mutex registrationMutex;
    std::condition_variable registrationCv;
    size_t expectedWorkerBackends = 0;
    size_t registeredWorkerBackends = 0;
    std::vector<IBackend*> workerBackends;

    std::mutex statesMutex;                             // guards `states` only
    std::unordered_map<IBackend*, WorkerState> states;

};