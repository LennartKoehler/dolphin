#pragma once

#include "CPUBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include <fftw3.h>
#include <atomic>
#include <mutex>
#include <shared_mutex>


// Logger functions - use log() for logging. The underlying LogCallback is
// heap-allocated and never destroyed, preventing use-after-free during static
// shutdown (e.g. in FFTWManager destructor).
void log(const std::string& message, LogLevel level);
LogCallback& getGlobalLogger();



struct FFTWPlan{
    fftwf_plan plan;
    FFTWPlanDescription description;
};

class FFTWWisdomManager{
public:
    // Default wisdom file location: ~/.fftw/wisdom in user home directory
    FFTWWisdomManager() = default;
    FFTWWisdomManager(const std::string& wisdomFilename);
    ~FFTWWisdomManager();

    bool importWisdom();
    bool exportWisdom();
    bool wisdomFileExists() const;
private:
    std::string wisdomFilename_;
    std::string getFullPath() const;
};


class FFTWManager{
public:
    FFTWManager() = default;
    FFTWManager(FFTWWisdomManager wisdomManager);
    ~FFTWManager();

    void init();

    void executeForwardFFT(const FFTWPlanDescription& description, fftwf_complex* indata, fftwf_complex* outdata);
    void executeBackwardFFT(const FFTWPlanDescription& description, fftwf_complex* indata, fftwf_complex* outdata);
    void executeForwardFFTReal(const FFTWPlanDescription& description, real_t* in, fftwf_complex* out);
    void executeBackwardFFTReal(const FFTWPlanDescription& description, fftwf_complex* in, real_t* out);
    void destroyFFTPlans();
private:

    fftwf_plan initializePlan(const FFTWPlanDescription& description);
    fftwf_plan initializePlanComplexToReal(const FFTWPlanDescription& description);
    fftwf_plan initializePlanRealToComplex(const FFTWPlanDescription& description);


    const fftwf_plan* findPlan(const FFTWPlanDescription& description);
    std::vector<FFTWPlan> fftwPlans;

    static std::once_flag initFlag_;
    std::atomic<bool> didInit_{false};
    std::shared_mutex mutex_;
    FFTWWisdomManager wisdomManager_;
};



//manage all cpu backends, currently should be used as a singleton
class CPUBackendManager : public IBackendManager{
public:

    CPUBackendManager() = default;
    virtual ~CPUBackendManager() override = default;
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

    MemoryTracking memory;
    std::unique_ptr<FFTWManager> fftwManager;

    std::mutex mutex_;
};

