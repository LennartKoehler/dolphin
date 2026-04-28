#include "CPUBackendManager.h"
#include "CPUBackend.h"
#include <dolphinbackend/Exceptions.h>
#include <fftw3.h>
#include <format>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

extern void log(const std::string& message, LogLevel level);

extern LogCallback& getGlobalLogger();

void CPUBackendManager::init(LogCallback fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    getGlobalLogger() = std::move(fn);

    FFTWWisdomManager wisdomManager(FFTW_WISDOM_PATH);
    fftwManager = std::make_unique<FFTWManager>(std::move(wisdomManager));
}

IDeconvolutionBackend& CPUBackendManager::getDeconvolutionBackend(const BackendConfig& config) {
    auto deconv = std::make_unique<CPUDeconvolutionBackend>(configToConfig(config), *fftwManager);
    std::unique_lock<std::mutex> lock(mutex_);
    deconvBackends.push_back(std::move(deconv));
    return *deconvBackends.back();
}

IBackendMemoryManager& CPUBackendManager::getBackendMemoryManager(const BackendConfig& config) {
    auto manager = std::make_unique<CPUBackendMemoryManager>(configToConfig(config), memory);
    std::unique_lock<std::mutex> lock(mutex_);
    memoryManagers.push_back(std::move(manager));
    return *memoryManagers.back();
}

IBackend& CPUBackendManager::getBackend(const BackendConfig& config) {

    auto backend = std::unique_ptr<CPUBackend>(
        CPUBackend::create(configToConfig(config), *fftwManager, memory)
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
    // workerConfig.nThreads = 1; //TESTVALUE

    ioThreads = ioThreads == 0 ? totalThreads : ioThreads;
    workerThreads = 1; //TESTVALUE
}



//------------------------------------------
FFTWManager::FFTWManager(FFTWWisdomManager wisdomManager) : wisdomManager_(wisdomManager) {
}

FFTWManager::~FFTWManager() {
    if (!didInit_.load(std::memory_order_acquire)) return;

    std::unique_lock<std::shared_mutex> lock(mutex_);
    destroyFFTPlans();
    wisdomManager_.exportWisdom();
    fftwf_cleanup_threads();
}

std::once_flag FFTWManager::initFlag_;

void FFTWManager::init(){
    std::call_once(initFlag_, [this]{
        fftwf_init_threads();
        wisdomManager_.importWisdom();
        didInit_.store(true, std::memory_order_release);
    });
}


void FFTWManager::executeForwardFFT(const FFTWPlanDescription& description, fftwf_complex* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(plan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft(*plan, in, out);
}

void FFTWManager::executeBackwardFFT(const FFTWPlanDescription& description, fftwf_complex* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(plan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft(*plan, in, out);
}

void FFTWManager::executeForwardFFTReal(const FFTWPlanDescription& description, real_t* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(plan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft_r2c(*plan, in, out);
}

void FFTWManager::executeBackwardFFTReal(const FFTWPlanDescription& description, fftwf_complex* in, real_t* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation");
    BACKEND_CHECK(plan != nullptr, "Forward FFT plan is null", "CPU", "forwardFFT - FFT plan");

    fftwf_execute_dft_c2r(*plan, in, out);
}



// WARNING this expects the input (realdata) to be allocated as if it will be used for inplacefft
// even if it wont be in place
fftwf_plan FFTWManager::initializePlanRealToComplex(const FFTWPlanDescription& description) {
    //has to be holding lock

    assert(getGlobalLogger() && "logger not yet set");

    fftwf_plan_with_nthreads(description.ompThreads); // each thread that calls the fftw_execute should run the fftw singlethreaded, but its called in parallel

    // Allocate temporary memory for plan creation
    real_t* in = nullptr;
    complex_t* out = nullptr;


    int rank = 3;
    int Nx = description.shape.width;
    int Ny = description.shape.height;
    int Nz = description.shape.depth;

    int n[3] = {Nz, Ny, Nx};

    // For out-of-place: real input is unpadded, inembed matches logical dimensions.
    // For in-place: real input must be padded on the last dimension to 2*(Nx/2+1)
    //              to accommodate the complex output, so inembed reflects the padded size.
    // onembed is always {Nz, Ny, Nx/2+1} (complex output has halved last dimension).
    int inembed[3];
    int onembed[3] = {Nz, Ny, Nx/2+1};

    int istride = 1;
    int ostride = 1;

    int idist;
    int odist = Nz * Ny * (Nx/2+1);

    // if (description.inPlace) {
    inembed[0] = Nz;
    inembed[1] = Ny;
    inembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    idist = Nz * Ny * 2*(Nx/2+1);
    // } else {
    //     inembed[0] = Nz;
    //     inembed[1] = Ny;
    //     inembed[2] = Nx;           // unpadded last dimension
    //     idist = Nz * Ny * Nx;
    // }

    try {
        out = (complex_t*)fftwf_malloc(sizeof(complex_t) * Nz * Ny * (Nx/2+1));
        FFTW_MALLOC_UNIFIED_CHECK(out, sizeof(complex_t) * Nz * Ny * (Nx/2+1), "initializePlanRealToComplex");

        if (description.inPlace) {
            in = (real_t*)out;  // shared buffer: out provides the padded storage
        } else {
            in = (real_t*)fftwf_malloc(sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1));
            FFTW_MALLOC_UNIFIED_CHECK(in, sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1), "initializePlanRealToComplex");
        }

        // Create FFT plan using advanced r2c interface
        fftwf_plan plan = fftwf_plan_many_dft_r2c(
            rank, n, 1,
            in, inembed,
            istride, idist,
            out, onembed,
            ostride, odist,
            FFTW_MEASURE
        );

        FFTW_UNIFIED_CHECK(plan, "initializePlanRealToComplex - r2c plan");

        std::string msg = std::format(
            "Successfully created FFTW r2c plan ({}) which uses {} threads for shape: {}x{}x{}",
            description.inPlace ? "in-place" : "out-of-place",
            description.ompThreads, description.shape.width, description.shape.height, description.shape.depth
        );

        log(msg, LogLevel::INFO);

        fftwf_free(out);
        if (!description.inPlace) fftwf_free(in);
        return plan;
    }
    catch (...) {
        if (out != nullptr) {
            fftwf_free(out);
        }
        if (in != nullptr && !description.inPlace) {
            fftwf_free(in);
        }
        throw;
    }
}


// WARNING this expects the output (realdata) to be allocated as if it will be used for inplacefft
// even if it wont be in place
fftwf_plan FFTWManager::initializePlanComplexToReal(const FFTWPlanDescription& description) {
    //has to be holding lock

    assert(getGlobalLogger() && "logger not yet set");

    fftwf_plan_with_nthreads(description.ompThreads); // each thread that calls the fftw_execute should run the fftw singlethreaded, but its called in parallel

    // Allocate temporary memory for plan creation
    complex_t* in = nullptr;
    real_t* out = nullptr;

    int rank = 3;
    int Nx = description.shape.width;
    int Ny = description.shape.height;
    int Nz = description.shape.depth;

    int n[3] = {Nz, Ny, Nx};

    // Complex input always has halved last dimension.
    // For out-of-place: real output is unpadded, onembed matches logical dimensions.
    // For in-place: real output must be padded on the last dimension to 2*(Nx/2+1)
    //              to match the complex input buffer, so onembed reflects the padded size.
    int inembed[3] = {Nz, Ny, Nx/2+1};
    int onembed[3];

    int istride = 1;
    int ostride = 1;

    int idist = Nz * Ny * (Nx/2+1);
    int odist;

    // if (description.inPlace) {
    onembed[0] = Nz;
    onembed[1] = Ny;
    onembed[2] = 2*(Nx/2+1);  // padded last dimension (in real_t units)
    odist = Nz * Ny * 2*(Nx/2+1);
    // } else {
    //     onembed[0] = Nz;
    //     onembed[1] = Ny;
    //     onembed[2] = Nx;           // unpadded last dimension
    //     odist = Nz * Ny * Nx;
    // }

    try {
        in = (complex_t*)fftwf_malloc(sizeof(complex_t) * Nz * Ny * (Nx/2+1));
        FFTW_MALLOC_UNIFIED_CHECK(in, sizeof(complex_t) * Nz * Ny * (Nx/2+1), "initializePlanComplexToReal");

        if (description.inPlace) {
            out = (real_t*)in;  // shared buffer: in provides the padded storage
        } else {
            out = (real_t*)fftwf_malloc(sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1));
            FFTW_MALLOC_UNIFIED_CHECK(out, sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1), "initializePlanComplexToReal");
        }

        // Create FFT plan using advanced c2r interface
        fftwf_plan plan = fftwf_plan_many_dft_c2r(
            rank, n, 1,
            in, inembed,
            istride, idist,
            out, onembed,
            ostride, odist,
            FFTW_MEASURE
        );

        FFTW_UNIFIED_CHECK(plan, "initializePlanComplexToReal - c2r plan");

        std::string msg = std::format(
            "Successfully created FFTW c2r plan ({}) which uses {} threads for shape: {}x{}x{}",
            description.inPlace ? "in-place" : "out-of-place",
            description.ompThreads, description.shape.width, description.shape.height, description.shape.depth
        );

        log(msg, LogLevel::INFO);

        fftwf_free(in);
        if (!description.inPlace) fftwf_free(out);
        return plan;
    }
    catch (...) {
        if (in != nullptr) {
            fftwf_free(in);
        }
        if (out != nullptr) {
            fftwf_free(out);
        }
        throw;
    }
}

fftwf_plan FFTWManager::initializePlan(const FFTWPlanDescription& description) {
    // not threadsafe!

    assert(getGlobalLogger() && "logger not yet set");

    fftwf_plan_with_nthreads(description.ompThreads); // each thread that calls the fftw_execute should run the fftw singlethreaded, but its called in parallel

    // Allocate temporary memory for plan creation
    complex_t* temp = nullptr;
    complex_t* tempout = nullptr;
    try {
        temp = (complex_t*)fftwf_malloc(sizeof(complex_t) * description.shape.getVolume());
        if (description.inPlace) tempout = temp;
        else tempout = (complex_t*)fftwf_malloc(sizeof(complex_t) * description.shape.getVolume());

        FFTW_MALLOC_UNIFIED_CHECK(temp, sizeof(complex_t) * description.shape.getVolume(), "initializePlan");

        // Create FFT plan
        fftwf_plan plan = fftwf_plan_dft_3d(description.shape.depth, description.shape.height, description.shape.width,
            temp, tempout, description.direction == PlanDirection::FORWARD ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_MEASURE);


        FFTW_UNIFIED_CHECK(plan, "initializePlan - forward plan");

        std::string msg = std::format(
            "Successfully created FFTW plan which uses {} threads for shape: {}x{}x{}",
            description.ompThreads, description.shape.width, description.shape.height, description.shape.depth
        );

        log(msg, LogLevel::INFO);

        fftwf_free(temp);
        if (!description.inPlace)fftwf_free(tempout);
        return plan;
    }
    catch (...) {
        if (temp != nullptr) {
            fftwf_free(temp);
        }
        if (!description.inPlace)fftwf_free(tempout);
        throw;
    }
}


void FFTWManager::destroyFFTPlans() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto& plan : fftwPlans) {
        fftwf_destroy_plan(plan.plan);
    }
    fftwPlans.clear();
}

const fftwf_plan* FFTWManager::findPlan(const FFTWPlanDescription& description) {
    init();

    // Fast path: shared lock for read-only lookup
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const FFTWPlan& plan : fftwPlans){
            if (plan.description == description){
                return &plan.plan;
            }
        }
    }

    // Slow path: exclusive lock for double-check + plan creation
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (const FFTWPlan& plan : fftwPlans){
        if (plan.description == description){
            return &plan.plan;
        }
    }

    // Create new plan and store it
    fftwf_plan newPlan;
    if (description.type == PlanType::COMPLEX) newPlan = initializePlan(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::FORWARD) newPlan = initializePlanRealToComplex(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::BACKWARD) newPlan = initializePlanComplexToReal(description);
    FFTWPlan plan{ std::move(newPlan), description };
    fftwPlans.push_back(std::move(plan));
    return &fftwPlans.back().plan;
}




//-----------------------------------------
// FFTWWisdomManager Implementation
//-----------------------------------------

FFTWWisdomManager::FFTWWisdomManager(const std::string& wisdomFilename) : wisdomFilename_(wisdomFilename) {}

FFTWWisdomManager::~FFTWWisdomManager() {}

std::string FFTWWisdomManager::getFullPath() const {
    // Expand ~ to user home directory
    std::string path = wisdomFilename_;
    if (path.starts_with("~")) {
        const char* home = std::getenv("HOME");
        if (home) {
            path = std::string(home) + path.substr(1);
        }
    }
    return path;
}

bool FFTWWisdomManager::wisdomFileExists() const {
    return fs::exists(getFullPath());
}

bool FFTWWisdomManager::importWisdom() {
    std::string fullPath = getFullPath();

    if (!fs::exists(fullPath)) {
        log(std::string("FFTW wisdom file not found at: ") + fullPath + ", skipping import", LogLevel::INFO);
        return false;
    }

    int success = fftwf_import_wisdom_from_filename(fullPath.c_str());
    if (success) {
        log(std::string("Successfully imported FFTW wisdom from: ") + fullPath, LogLevel::DEBUG);
        return true;
    } else {
        log(std::string("Failed to import FFTW wisdom from: ") + fullPath, LogLevel::WARN);
        return false;
    }
}

bool FFTWWisdomManager::exportWisdom() {
    std::string fullPath = getFullPath();

    // Ensure directory exists
    fs::path dirPath = fs::path(fullPath).parent_path();
    if (!dirPath.empty() && !fs::exists(dirPath)) {
        try {
            fs::create_directories(dirPath);
        } catch (const fs::filesystem_error& e) {
            log(std::string("Failed to create FFTW wisdom directory: ") + std::string(e.what()), LogLevel::ERROR);
            return false;
        }
    }

    int success = fftwf_export_wisdom_to_filename(fullPath.c_str());
    if (success) {
        std::string message = std::string("Successfully exported FFTW wisdom to: ") + fullPath;
        log(message, LogLevel::DEBUG);
        return true;
    } else {
        log(std::string("Failed to export FFTW wisdom to: ") + fullPath, LogLevel::ERROR);
        return false;
    }
}
