#pragma once

#include "CPUBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include <fftw3.h>
#include <mutex>


extern LogCallback g_;
//manage all cpu backends, currently should be used as a singleton
class CPUBackendManager : public IBackendManager{
public:

    CPUBackendManager() = default;
    ~CPUBackendManager() override = default;
    void init(LogCallback fn) override;

    IDeconvolutionBackend& getDeconvolutionBackend(const BackendConfig& config) override;
    IBackendMemoryManager& getBackendMemoryManager(const BackendConfig& config) override;
    IBackend& getBackend(const BackendConfig& config) override;

    IBackend& clone(IBackend& backend, const BackendConfig& config) override ;
    IBackend& cloneSharedMemory(IBackend& backend, const BackendConfig& config) override;

    void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    int getNumberDevices() const override;
private:

    CPUBackendConfig configToConfig(const BackendConfig& config) const;

    std::vector<std::unique_ptr<CPUBackend>> backends;
    std::vector<std::unique_ptr<CPUDeconvolutionBackend>> deconvBackends;
    std::vector<std::unique_ptr<CPUBackendMemoryManager>> memoryManagers;

    std::mutex mutex_;
};


struct FFTWPlan{
    fftwf_plan plan;
    PlanDescription description;
};
class FFTWManager{
public:
    FFTWManager();
    ~FFTWManager();


    void executeForwardFFT(const PlanDescription& description, fftwf_complex* indata, fftwf_complex* outdata);
    void executeBackwardFFT(const PlanDescription& description, fftwf_complex* indata, fftwf_complex* outdata);
    void executeForwardFFTReal(const PlanDescription& description, real_t* in, fftwf_complex* out);
    void executeBackwardFFTReal(const PlanDescription& description, fftwf_complex* in, real_t* out);
    void destroyFFTPlans();
private:

    fftwf_plan initializePlan(const PlanDescription& description);
    fftwf_plan initializePlanComplexToReal(const PlanDescription& description);
    fftwf_plan initializePlanRealToComplex(const PlanDescription& description);


    const fftwf_plan* findPlan(const PlanDescription& description);
    std::vector<FFTWPlan> fftwPlans;
    // std::vector<FFTWPlan> forwardPlansReal;
    // std::vector<FFTWPlan> backwardPlans;
    // std::vector<FFTWPlan> backwardPlansReal;

    std::mutex mutex_;
};
