#include "CPUBackendManager.h"
#include "CPUBackend.h"
#include <dolphinbackend/Exceptions.h>
#include <fftw3.h>
#include <spdlog/fmt/fmt.h>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using dolphin::backend::buildCpuContext;

extern void log(const std::string& message, LogLevel level);

extern LogCallback& getGlobalLogger();

void CPUBackendManager::init(LogCallback fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    getGlobalLogger() = std::move(fn);

    FFTWWisdomManager wisdomManager(FFTW_WISDOM_PATH);
    fftwManager = std::make_unique<FFTWManager>(std::move(wisdomManager));
}

// IComputeBackend& CPUBackendManager::getComputeBackend(const BackendConfig& config) {
//     auto compute = createComputeBackend(configToConfig(config));
//     std::unique_lock<std::mutex> lock(mutex_);
//     computeBackends.push_back(std::move(compute));
//     return *computeBackends.back();
// }

// IBackendMemoryManager& CPUBackendManager::getBackendMemoryManager(const BackendConfig& config) {
//     auto manager = createMemoryManager(configToConfig(config));
//     std::unique_lock<std::mutex> lock(mutex_);
//     memoryManagers.push_back(std::move(manager));
//     return *memoryManagers.back();
// }

std::unique_ptr<CPUComputeBackend> CPUBackendManager::createComputeBackend(CPUBackendConfig config) {
    return std::make_unique<CPUComputeBackend>(config, *fftwManager);
}

std::unique_ptr<CPUBackendMemoryManager> CPUBackendManager::createMemoryManager(CPUBackendConfig config) {
    return std::make_unique<CPUBackendMemoryManager>(config, memory);
}

IBackend& CPUBackendManager::createBackendForCurrentThread(const BackendConfig& config) {
    CPUBackendConfig cpuconfig = configToConfig(config);
    auto compute = createComputeBackend(cpuconfig);
    auto mem = createMemoryManager(cpuconfig);
    auto backend = std::unique_ptr<CPUBackend>(new CPUBackend(std::move(compute), std::move(mem), cpuconfig));
    std::unique_lock<std::mutex> lock(mutex_);
    IBackend& ref = *backend;
    backends.push_back(std::move(backend));
    return ref;
}

CPUBackendConfig CPUBackendManager::configToConfig(const BackendConfig& config) const {
    CPUBackendConfig cpuconfig{true, config.nThreads};
    return cpuconfig;
}

// IBackend& CPUBackendManager::clone(IBackend& backend, const BackendConfig& config){
//     return backend;
// }

// multiple seperate cpu devices e.g. NUMA not supported
IBackend& CPUBackendManager::createBackendSharedMemoryForCurrentThread(IBackend& backend, const BackendConfig& config){
    return createBackendForCurrentThread(config);
}

int CPUBackendManager::getNumberDevices() const {
    return 1;
}

void CPUBackendManager::setThreadDistribution(const size_t& totalThreads, size_t& ioThreads, size_t& workerThreads, BackendConfig& ioconfig, BackendConfig& workerConfig) {
    ioconfig.nThreads = 1;
    workerConfig.nThreads = workerThreads == 0 ? static_cast<size_t>(2*totalThreads/3) : workerThreads;
    workerConfig.nThreads = workerConfig.nThreads == 0 ? 1 : workerConfig.nThreads;

    ioThreads = ioThreads == 0 ? totalThreads : ioThreads;
    workerThreads = 1; //TESTVALUE
}



//------------------------------------------
FFTWManager::FFTWManager(FFTWWisdomManager wisdomManager) : wisdomManager_(wisdomManager) {
}

FFTWManager::~FFTWManager() {
    if (!didInit_.load(std::memory_order_acquire)) return;

    destroyFFTPlans();
    try {
        wisdomManager_.exportWisdom();
    } catch (...) {}
    fftwf_cleanup_threads();
}

std::once_flag FFTWManager::initFlag_;

void FFTWManager::init(){
    std::call_once(initFlag_, [this]{
        fftwf_init_threads();
        try {
            wisdomManager_.importWisdom();
        } catch (...) {
            log("FFTW wisdom import failed, continuing without wisdom", LogLevel::WARN);
        }
        didInit_.store(true, std::memory_order_release);
    });
}


void FFTWManager::executeForwardFFT(const FFTWPlanDescription& description, fftwf_complex* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFT - plan creation", buildCpuContext());
    fftwf_execute_dft(*plan, in, out);
}

void FFTWManager::executeBackwardFFT(const FFTWPlanDescription& description, fftwf_complex* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "backwardFFT - plan creation", buildCpuContext());
    fftwf_execute_dft(*plan, in, out);
}

void FFTWManager::executeForwardFFTReal(const FFTWPlanDescription& description, real_t* in, fftwf_complex* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "forwardFFTReal - plan creation", buildCpuContext());
    fftwf_execute_dft_r2c(*plan, in, out);
}

void FFTWManager::executeBackwardFFTReal(const FFTWPlanDescription& description, fftwf_complex* in, real_t* out){
    auto* plan = findPlan(description);
    BACKEND_CHECK(plan != nullptr, "Failed to create FFT plan for shape", "CPU", "backwardFFTReal - plan creation", buildCpuContext());
    fftwf_execute_dft_c2r(*plan, in, out);
}



// WARNING this expects the input (realdata) to be allocated as if it will be used for inplacefft
// even if it wont be in place
fftwf_plan FFTWManager::initializePlanRealToComplex(const FFTWPlanDescription& description) {
    //has to be holding lock

    BACKEND_CHECK(getGlobalLogger(), "Logger not set", "CPU", "initializePlanRealToComplex - logger", buildCpuContext());

    fftwf_plan_with_nthreads(description.ompThreads);

    real_t* in = nullptr;
    complex_t* out = nullptr;

    int rank = 3;
    size_t Nx = description.shape.width;
    size_t Ny = description.shape.height;
    size_t Nz = description.shape.depth;

    int n[3] = {static_cast<int>(Nz), static_cast<int>(Ny), static_cast<int>(Nx)};

    // For out-of-place: real input is unpadded, inembed matches logical dimensions.
    // For in-place: real input must be padded on the last dimension to 2*(Nx/2+1)
    //              to accommodate the complex output, so inembed reflects the padded size.
    // onembed is always {Nz, Ny, Nx/2+1} (complex output has halved last dimension).
    int inembed[3];
    int onembed[3] = {static_cast<int>(Nz), static_cast<int>(Ny), static_cast<int>(Nx/2+1)};

    int istride = 1;
    int ostride = 1;

    int idist;
    int odist = static_cast<int>(Nz * Ny * (Nx/2+1));

    inembed[0] = static_cast<int>(Nz);
    inembed[1] = static_cast<int>(Ny);
    inembed[2] = static_cast<int>(2*(Nx/2+1));  // padded last dimension (in real_t units)
    idist = static_cast<int>(Nz * Ny * 2*(Nx/2+1));

    try {
        out = (complex_t*)fftwf_malloc(sizeof(complex_t) * Nz * Ny * (Nx/2+1));
        FFTW_MALLOC_UNIFIED_CHECK(out, sizeof(complex_t) * Nz * Ny * (Nx/2+1), "initializePlanRealToComplex", buildCpuContext());

        if (description.inPlace) {
            in = (real_t*)out;  // shared buffer: out provides the padded storage
        } else {
            in = (real_t*)fftwf_malloc(sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1));
            FFTW_MALLOC_UNIFIED_CHECK(in, sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1), "initializePlanRealToComplex", buildCpuContext());
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

        FFTW_UNIFIED_CHECK(plan, "initializePlanRealToComplex - r2c plan", buildCpuContext());

        std::string msg = fmt::format(
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

    BACKEND_CHECK(getGlobalLogger(), "Logger not set", "CPU", "initializePlanComplexToReal - logger", buildCpuContext());

    fftwf_plan_with_nthreads(description.ompThreads);

    complex_t* in = nullptr;
    real_t* out = nullptr;

    int rank = 3;
    size_t Nx = description.shape.width;
    size_t Ny = description.shape.height;
    size_t Nz = description.shape.depth;

    int n[3] = {static_cast<int>(Nz), static_cast<int>(Ny), static_cast<int>(Nx)};

    // Complex input always has halved last dimension.
    // For out-of-place: real output is unpadded, onembed matches logical dimensions.
    // For in-place: real output must be padded on the last dimension to 2*(Nx/2+1)
    //              to match the complex input buffer, so onembed reflects the padded size.
    int inembed[3] = {static_cast<int>(Nz), static_cast<int>(Ny), static_cast<int>(Nx/2+1)};
    int onembed[3];

    int istride = 1;
    int ostride = 1;

    int idist = static_cast<int>(Nz * Ny * (Nx/2+1));
    int odist;

    onembed[0] = static_cast<int>(Nz);
    onembed[1] = static_cast<int>(Ny);
    onembed[2] = static_cast<int>(2*(Nx/2+1));  // padded last dimension (in real_t units)
    odist = static_cast<int>(Nz * Ny * 2*(Nx/2+1));

    try {
        in = (complex_t*)fftwf_malloc(sizeof(complex_t) * Nz * Ny * (Nx/2+1));
        FFTW_MALLOC_UNIFIED_CHECK(in, sizeof(complex_t) * Nz * Ny * (Nx/2+1), "initializePlanComplexToReal", buildCpuContext());

        if (description.inPlace) {
            out = (real_t*)in;  // shared buffer: in provides the padded storage
        } else {
            out = (real_t*)fftwf_malloc(sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1));
            FFTW_MALLOC_UNIFIED_CHECK(out, sizeof(real_t) * 2 * Nz * Ny * (Nx/2+1), "initializePlanComplexToReal", buildCpuContext());
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

        FFTW_UNIFIED_CHECK(plan, "initializePlanComplexToReal - c2r plan", buildCpuContext());

        std::string msg = fmt::format(
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

void FFTWManager::initializePlan(const FFTWPlanDescription& description) {
    fftwf_plan newPlan;
    if (description.type == PlanType::COMPLEX) newPlan = initializePlanComplex(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::FORWARD) newPlan = initializePlanRealToComplex(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::BACKWARD) newPlan = initializePlanComplexToReal(description);

    addPlan(newPlan, description);
}

fftwf_plan FFTWManager::initializePlanComplex(const FFTWPlanDescription& description) {
    // not threadsafe!

    BACKEND_CHECK(getGlobalLogger(), "Logger not set", "CPU", "initializePlanComplex - logger", buildCpuContext());

    fftwf_plan_with_nthreads(description.ompThreads);

    complex_t* temp = nullptr;
    complex_t* tempout = nullptr;
    try {
        temp = (complex_t*)fftwf_malloc(sizeof(complex_t) * description.shape.getVolume());
        if (description.inPlace) tempout = temp;
        else tempout = (complex_t*)fftwf_malloc(sizeof(complex_t) * description.shape.getVolume());

        FFTW_MALLOC_UNIFIED_CHECK(temp, sizeof(complex_t) * description.shape.getVolume(), "initializePlan", buildCpuContext());

        // Create FFT plan
        fftwf_plan plan = fftwf_plan_dft_3d(static_cast<int>(description.shape.depth), static_cast<int>(description.shape.height), static_cast<int>(description.shape.width),
            temp, tempout, description.direction == PlanDirection::FORWARD ? FFTW_FORWARD : FFTW_BACKWARD, FFTW_MEASURE);

        FFTW_UNIFIED_CHECK(plan, "initializePlan - forward plan", buildCpuContext());

        std::string msg = fmt::format(
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
    initializePlan(description);

    for (const FFTWPlan& plan : fftwPlans){
        if (plan.description == description){
            return &plan.plan;
        }
    }
    throw dolphin::backend::BackendException(
        "FFTW plan not found after creation",
        "CPU",
        "findPlan - plan not found after initializePlan",
        buildCpuContext()
    );
}

void FFTWManager::addPlan(fftwf_plan& handle, const FFTWPlanDescription& description){
    FFTWPlan plan{ handle, description };
    fftwPlans.push_back(std::move(plan));
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

std::string FFTWWisdomManager::resolveWritablePath() const {
    // 1. Try the configured path (e.g. ~/.cache/dolphin/fftw/wisdom)
    std::string primaryPath = getFullPath();
    fs::path primaryDir = fs::path(primaryPath).parent_path();

    try {
        if (!primaryDir.empty()) {
            if (!fs::exists(primaryDir)) {
                fs::create_directories(primaryDir);
            }
            // Test writability
            fs::path testFile = primaryDir / ".dolphin_write_test";
            std::ofstream test(testFile);
            if (test) {
                test.close();
                fs::remove(testFile);
                return primaryPath;
            }
        }
    } catch (const std::exception& e) {
        log(std::string("FFTW wisdom path not accessible: ") + primaryPath + " (" + e.what() + ")", LogLevel::WARN);
    }

    // 2. Fall back to a directory next to the running application
    try {
        fs::path cwd = fs::current_path();
        fs::path fallbackDir = cwd / ".dolphin" / "fftw";
        fs::create_directories(fallbackDir);

        fs::path testFile = fallbackDir / ".dolphin_write_test";
        std::ofstream test(testFile);
        if (test) {
            test.close();
            fs::remove(testFile);
            std::string fallbackPath = (fallbackDir / "wisdom").string();
            log(std::string("FFTW wisdom using fallback path: ") + fallbackPath, LogLevel::WARN);
            return fallbackPath;
        }
    } catch (const std::exception& e) {
        log(std::string("FFTW wisdom fallback path not writable: ") + std::string(e.what()), LogLevel::WARN);
    }

    // 3. No writable location found
    log("FFTW wisdom will not be used (no writable path available)", LogLevel::WARN);
    return "";
}

bool FFTWWisdomManager::wisdomFileExists() const {
    try {
        return fs::exists(getFullPath());
    } catch (...) {
        return false;
    }
}

bool FFTWWisdomManager::importWisdom() {
    try {
        std::string fullPath = resolveWritablePath();
        if (fullPath.empty()) {
            return false;
        }

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
    } catch (const std::exception& e) {
        log(std::string("Error importing FFTW wisdom: ") + e.what(), LogLevel::WARN);
        return false;
    }
}

bool FFTWWisdomManager::exportWisdom() {
    try {
        std::string fullPath = resolveWritablePath();
        if (fullPath.empty()) {
            return false;
        }

        fs::path dirPath = fs::path(fullPath).parent_path();
        if (!dirPath.empty() && !fs::exists(dirPath)) {
            fs::create_directories(dirPath);
        }

        int success = fftwf_export_wisdom_to_filename(fullPath.c_str());
        if (success) {
            log(std::string("Successfully exported FFTW wisdom to: ") + fullPath, LogLevel::DEBUG);
            return true;
        } else {
            log(std::string("Failed to export FFTW wisdom to: ") + fullPath, LogLevel::ERROR);
            return false;
        }
    } catch (const std::exception& e) {
        log(std::string("Error exporting FFTW wisdom: ") + e.what(), LogLevel::WARN);
        return false;
    }
}
