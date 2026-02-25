#include "CPUBackendManager.h"
#include "CPUBackend.h"
#include <mutex>
#include <stdexcept>
#include <format>

extern LogCallback g_logger;
void CPUBackendManager::init(LogCallback fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    g_logger = std::move(fn);
}

IDeconvolutionBackend& CPUBackendManager::getDeconvolutionBackend(const BackendConfig& config) {
    auto deconv = std::make_unique<CPUDeconvolutionBackend>(configToConfig(config));
    std::unique_lock<std::mutex> lock(mutex_);
    deconvBackends.push_back(std::move(deconv));
    return *deconvBackends.back();
}

IBackendMemoryManager& CPUBackendManager::getBackendMemoryManager(const BackendConfig& config) {
    auto manager = std::make_unique<CPUBackendMemoryManager>(configToConfig(config));
    std::unique_lock<std::mutex> lock(mutex_);
    memoryManagers.push_back(std::move(manager));
    return *memoryManagers.back();
}

IBackend& CPUBackendManager::getBackend(const BackendConfig& config) {

    auto backend = std::unique_ptr<CPUBackend>(
        CPUBackend::create(configToConfig(config))
    );
    std::unique_lock<std::mutex> lock(mutex_);
    IBackend& ref = *backend;
    backends.push_back(std::move(backend));
    return ref;
}

CPUBackendConfig CPUBackendManager::configToConfig(const BackendConfig& config) const {
    CPUBackendConfig cpuconfig{true, config.nThreads};
    return cpuconfig;
}

IBackend& CPUBackendManager::clone(IBackend& backend, const BackendConfig& config){
    return backend;
}
IBackend& CPUBackendManager::cloneSharedMemory(IBackend& backend, const BackendConfig& config){
    return getBackend(config);
}

int CPUBackendManager::getNumberDevices() const {
    return 1;
}

void CPUBackendManager::setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) {
    // workerThreads = static_cast<size_t>(2*totalThreads/3);
    // config.ompThreads = static_cast<int>(workerThreads);
    ioconfig.nThreads = 1;
    workerConfig.nThreads = workerThreads == 0 ? static_cast<size_t>(2*totalThreads/3) : workerThreads;
    workerConfig.nThreads = workerConfig.nThreads == 0 ? 1 : workerConfig.nThreads;
    workerConfig.nThreads = 1; //TESTVALUE
    
    ioThreads = ioThreads == 0 ? totalThreads : ioThreads;
    // workerThreads = 1; //TESTVALUE
}



//------------------------------------------
FFTWManager::FFTWManager() {
    fftwf_init_threads();
}

FFTWManager::~FFTWManager() {
    fftwf_cleanup_threads();
}


void FFTWManager::executeForwardFFT(int ompThreads, const CuboidShape& size, fftwf_complex* in, fftwf_complex* out){

    auto* forwardPlan = getForwardPlan(size, ompThreads);
    BACKEND_CHECK(forwardPlan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(forwardPlan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft(*forwardPlan, in, out);
}

void FFTWManager::executeBackwardFFT(int ompThreads, const CuboidShape& size, fftwf_complex* in, fftwf_complex* out){

    auto* backwardPlan = getBackwardPlan(size, ompThreads);
    BACKEND_CHECK(backwardPlan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(backwardPlan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft(*backwardPlan, in, out);
}

fftwf_plan FFTWManager::initializePlan(const CuboidShape& shape, int direction, int ompThreads) {
    //has to be holding lock


    fftwf_plan_with_nthreads(ompThreads); // each thread that calls the fftw_execute should run the fftw singlethreaded, but its called in parallel
    
    // Allocate temporary memory for plan creation
    complex_t* temp = nullptr;
    try {
        temp = (complex_t*)fftwf_malloc(sizeof(complex_t) * shape.getVolume());
        FFTW_MALLOC_UNIFIED_CHECK(temp, sizeof(complex_t) * shape.getVolume(), "initializePlan");

        // Create FFT plan
        fftwf_plan plan = fftwf_plan_dft_3d(shape.depth, shape.height, shape.width,
            temp, temp, direction, FFTW_MEASURE);
        
        
        FFTW_UNIFIED_CHECK(plan, "initializePlan - forward plan");
        
        if (logger_) {
            std::string planInfo = std::string("FFTWF3 plan:\n") + fftwf_sprint_plan(plan);
            logger_(planInfo, LogLevel::DEBUG);
        }

        std::string msg = std::format(
            "Successfully created FFTW plan for shape: {}x{}x{}",
            shape.width, shape.height, shape.depth
        );

        if (logger_) {
            logger_(msg, LogLevel::INFO);
        }
        
        fftwf_free(temp);
        return plan;
    }
    catch (...) {
        if (temp != nullptr) {
            fftwf_free(temp);
        }
        throw;
    }
}

const fftwf_plan* FFTWManager::getForwardPlan(const CuboidShape& shape, int ompThreads) {
    return findPlan(forwardPlans, FFTW_FORWARD, shape, ompThreads);
}

const fftwf_plan* FFTWManager::getBackwardPlan(const CuboidShape& shape, int ompThreads) {
    return findPlan(backwardPlans, FFTW_BACKWARD, shape, ompThreads);
}

void FFTWManager::destroyFFTPlans() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& plan : forwardPlans) {
        fftwf_destroy_plan(plan.plan);
    }
    for (auto& plan : backwardPlans) {
        fftwf_destroy_plan(plan.plan);
    }
}
//TODO this findPlan needs to be somewhat fast
const fftwf_plan* FFTWManager::findPlan(std::vector<FFTWPlan>& plans, int direction, const CuboidShape& shape, int ompThreads) {
    
    std::unique_lock<std::mutex> lock(mutex_); // the lookup is thread safe, one could do double search and give lock to init if not found
    for (FFTWPlan& plan : plans){
        if (shape == plan.shape && ompThreads == plan.ompThreads){
            return &plan.plan;
        }
    }
    
    // Create new plan and store it in the map
    fftwf_plan newPlan = initializePlan(shape, direction, ompThreads);
    FFTWPlan plan{ std::move(newPlan), ompThreads, shape };
    plans.push_back(std::move(plan));
    return &plans.back().plan;  // Return reference to the stored plan
}
    