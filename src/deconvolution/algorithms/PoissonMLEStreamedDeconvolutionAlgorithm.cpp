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

#include "dolphin/deconvolution/algorithms/PoissonMLEStreamedDeconvolutionAlgorithm.h"
#include "dolphin/backend/BackendFactory.h"
#include <cassert>
#include <algorithm>
#include <stdexcept>

void PoissonMLEStreamedDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    core_.configure(asConfig<PoissonMLEConfig>(config, DeconvolutionAlgorithms::kPoissonMLEStreamed));
}

void PoissonMLEStreamedDeconvolutionAlgorithm::init(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE (streamed) algorithm initialization");
    core_.initPlans(backend->mutableComputeManager(), dataSize);
    initialized = true;
}

void PoissonMLEStreamedDeconvolutionAlgorithm::preAllocate(const CuboidShape& dataSize) {
    assert(backend && "No backend available for PoissonMLE (streamed) algorithm allocation");
    assert(initialized && "PoissonMLE (streamed) algorithm not initialized. Call init() first.");

    // Deliberately NO full work buffer (unlike the classic variant): the data
    // step runs out-of-core with the caller's g buffer as work. Only the
    // staging slice is kept.
    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const size_t realStride = 2 * (dataSize.width / 2 + 1);
    ratioSlab = memory.allocateMemoryOnDeviceReal(CuboidShape(realStride, dataSize.height, 1));
}

bool PoissonMLEStreamedDeconvolutionAlgorithm::isInitialized() const {
    return initialized;
}

void PoissonMLEStreamedDeconvolutionAlgorithm::deconvolve(const ComplexData& H, RealData& g, RealData& f,
                                                          const DeconvolutionRunContext& runContext) {
    assert(backend && "No backend available for PoissonMLE (streamed) algorithm");
    assert(initialized && "PoissonMLE (streamed) algorithm not initialized. Call init() first.");
    if (core_.iterations <= 0) return;

    const IBackendMemoryManager& memory = backend->getMemoryManager();
    const IComputeBackend& compute = backend->getComputeManager();
    const CuboidShape shape = f.getSize();

    assert(memory.isOnDevice(f) && "f is not on device");
    assert(memory.isOnDevice(g) && "g is not on device");
    if (!backend->supportsStreamedPoissonDataStep()) {
        throw std::runtime_error(
            "PoissonMLEStreamed requires a backend with the streamed data step; "
            "use the plain PoissonMLE algorithm on this backend");
    }

    // The observed image is captured on the host once per cube; the device g
    // buffer then serves as the data-step work buffer and is destroyed by it.
    const IBackendMemoryManager& hostMem = BackendFactory::getInstance().getHostBackendMemoryManager();
    RealData gHost = memory.isHostBackend() ? memory.createCopy(g)
                                            : memory.moveDataFromDevice(g, hostMem);
    RealData& work = g;

    // ---- Initial estimate: the data, optionally prefiltered. ----
    memory.memCopy(g, f);

    // ---- Physical parameters: cached PSF sigma widths (run-constant). ----
    ComplexView workView = work.reinterpret();
    core_.ensurePsfSigmas(compute, memory, H, shape, workView, work);

    // ---- Prefilter: a single mild anisotropic Gaussian low-pass of the raw data. ----
    core_.applyPrefilter(compute, f, workView, core_.psfSigX, core_.psfSigY, core_.psfSigZ);

    const real_t bg = core_.resolveBackground(runContext);
    const real_t lambda = core_.currentLambda();

    // f is the object ABOVE the background (m = Hf + bg), so the physical
    // constraint is f >= 0 and the background of f converges to 0.
    compute.realClampMin(f, 0.0f, f);

    const int maxIter = core_.iterations;
    const real_t kLast = std::max(1.0f, 0.8f * static_cast<real_t>(maxIter));

    for (int k = 0; k < maxIter; ++k) {
        // ---- DATA STEP: fused, out-of-core. work = R = H^T(g/(Hf+bg)), then
        // f = max(f*R, 0). Destroys work (= g's device buffer). ----
        backend->poissonDataStepStreamed(H, gHost, f, work, ratioSlab, bg);

        // ---- PRIOR + SIEVE: regularization, then the physical clamp. The work
        // buffer's complex view doubles as the sieve's Gaussian-OTF buffer (the
        // work volume is dead between data steps). ----
        core_.applyRegularization(compute, f, workView, k, kLast, lambda);

        backend->sync();
        if (progressFunction) progressFunction(maxIter);
    }
}

size_t PoissonMLEStreamedDeconvolutionAlgorithm::getWorkingSetBytes(const CuboidShape& dataSize) const {
    // Only the staging slice: the observed image lives in host RAM, the work
    // volume is the caller's g buffer.
    const size_t realStride = 2 * (dataSize.width / 2 + 1);
    return dataSize.height * realStride * sizeof(real_t);
}

std::unique_ptr<DeconvolutionAlgorithm> PoissonMLEStreamedDeconvolutionAlgorithm::cloneSpecific() const {
    auto copy = std::make_unique<PoissonMLEStreamedDeconvolutionAlgorithm>();
    copy->core_ = core_;
    copy->core_.resetForClone();
    copy->initialized = false;
    return copy;
}