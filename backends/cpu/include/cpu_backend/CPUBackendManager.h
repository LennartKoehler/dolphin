#pragma once

#include "CPUBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include <fftw3.h>
#include <mutex>
#include <map>


extern LogCallback g_logger;
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

    LogCallback logger_;
    std::mutex mutex_;
};

struct FFTWPlan{
    fftwf_plan plan;
    int ompThreads;
    CuboidShape shape;
};
class FFTWManager{
public:
    FFTWManager();
    ~FFTWManager();
    

    void executeForwardFFT(int ompThreads, const CuboidShape& size, fftwf_complex* indata, fftwf_complex* outdata);
    void executeBackwardFFT(int ompThreads, const CuboidShape& size, fftwf_complex* indata, fftwf_complex* outdata);
    void destroyFFTPlans();
private:

    fftwf_plan initializePlan(const CuboidShape& shape, int direction, int ompThreads);
    const fftwf_plan* getForwardPlan(const CuboidShape& shape, int ompThreads);
    const fftwf_plan* getBackwardPlan(const CuboidShape& shape, int ompThreads);


    const fftwf_plan* findPlan( std::vector<FFTWPlan>& plans, int direction, const CuboidShape& shape, int ompThreads);
    std::vector<FFTWPlan> forwardPlans;
    std::vector<FFTWPlan> backwardPlans;
    
    std::mutex mutex_;
    LogCallback logger_;
};