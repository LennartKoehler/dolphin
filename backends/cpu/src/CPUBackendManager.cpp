#include "CPUBackendManager.h"
#include "CPUBackend.h"
#include <dolphinbackend/Exceptions.h>
#include <format>
#include <mutex>

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
    // workerConfig.nThreads = 1; //TESTVALUE

    ioThreads = ioThreads == 0 ? totalThreads : ioThreads;
    workerThreads = 1; //TESTVALUE
}



//------------------------------------------
FFTWManager::FFTWManager() {
    fftwf_init_threads();
}

FFTWManager::~FFTWManager() {
    fftwf_cleanup_threads();
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

    assert(g_logger && "logger not yet set");

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

        g_logger(msg, LogLevel::INFO);

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

    assert(g_logger && "logger not yet set");

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

        g_logger(msg, LogLevel::INFO);

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

    assert(g_logger && "logger not yet set");

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

        g_logger(msg, LogLevel::INFO);

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
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& plan : fftwPlans) {
        fftwf_destroy_plan(plan.plan);
    }
}

const fftwf_plan* FFTWManager::findPlan(const FFTWPlanDescription& description) {

    for (FFTWPlan& plan : fftwPlans){
        if (plan.description == description){
            return &plan.plan;
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    for (FFTWPlan& plan : fftwPlans){
        if (plan.description == description){
            return &plan.plan;
        }
    }

    // Create new plan and store it in the map
    fftwf_plan newPlan;
    if (description.type == PlanType::COMPLEX) newPlan = initializePlan(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::FORWARD) newPlan = initializePlanRealToComplex(description);
    else if (description.type == PlanType::REAL && description.direction == PlanDirection::BACKWARD) newPlan = initializePlanComplexToReal(description);
    FFTWPlan plan{ std::move(newPlan), description };
    fftwPlans.push_back(std::move(plan));
    return &fftwPlans.back().plan;  // Return reference to the stored plan
}

