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
    ~FFTWWisdomManager() = default;

    bool importWisdom();
    bool exportWisdom();
    bool wisdomFileExists() const;
private:
    std::string wisdomFilename_;
    std::string getFullPath() const;
    std::string resolveWritablePath() const;
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

    void addPlan(fftwf_plan& handle, const FFTWPlanDescription& description);
    void initializePlan(const FFTWPlanDescription& description);
    fftwf_plan initializePlanComplex(const FFTWPlanDescription& description);
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
    ~CPUBackendManager() override = default;
    void init(LogCallback fn) override;

    IBackend& createBackendForCurrentThread(const BackendConfig& config) override;

    // IBackend& clone(IBackend& backend, const BackendConfig& config) override ;
    IBackend& createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config) override;

    void setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) override;

    int getNumberDevices() const override;
protected:

    CPUBackendConfig configToConfig(const BackendConfig& config) const;

    virtual std::unique_ptr<CPUComputeBackend> createComputeBackend(CPUBackendConfig config);
    virtual std::unique_ptr<CPUBackendMemoryManager> createMemoryManager(CPUBackendConfig config);

    std::vector<std::unique_ptr<CPUBackend>> backends;
    std::vector<std::unique_ptr<CPUComputeBackend>> computeBackends;
    std::vector<std::unique_ptr<CPUBackendMemoryManager>> memoryManagers;

    MemoryTracking memory;
    std::unique_ptr<FFTWManager> fftwManager;

    std::mutex mutex_;
};
