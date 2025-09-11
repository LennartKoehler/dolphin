
# DOLPHIN API Reference

This comprehensive API reference provides detailed documentation for all classes, methods, and interfaces in the refactored DOLPHIN CPU/GPU architecture. It covers inheritance hierarchies, virtual methods, helper functions, and configuration interfaces.

## Table of Contents
- [Core Architecture Classes](#core-architecture-classes)
- [Backend-Specific Classes](#backend-specific-classes)
- [Algorithm Implementations](#algorithm-implementations)
- [Configuration Classes](#configuration-classes)
- [Service Layer](#service-layer)
- [Helper Functions](#helper-functions)
- [Frontend Interfaces](#frontend-interfaces)
- [Data Structures](#data-structures)
- [Performance Monitoring](#performance-monitoring)
- [Utility Classes](#utility-classes)

## Core Architecture Classes

### BaseDeconvolutionAlgorithmDerived

The central abstract base class that provides common functionality and defines the interface for backend-specific implementations.

**File:** [`include/deconvolution/algorithms/BaseDeconvolutionAlgorithmDerived.h`](include/deconvolution/algorithms/BaseDeconvolutionAlgorithmDerived.h)

#### Class Definition

```cpp
class BaseDeconvolutionAlgorithmDerived : public BaseDeconvolutionAlgorithm
{
public:
    // Constructor and Destructor
    BaseDeconvolutionAlgorithmDerived();
    virtual ~BaseDeconvolutionAlgorithmDerived();
    
    // Backend Interface - Common Implementation
    virtual bool preprocess(int channel_num, int psf_index) override;
    virtual void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    virtual bool postprocess(int channel_num, int psf_index) override;
    
    // Backend Interface - Backend-Specific Overrides
    virtual bool preprocessBackendSpecific(int channel_num, int psf_index) = 0;
    virtual void algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) = 0;
    virtual bool postprocessBackendSpecific(int channel_num, int psf_index) = 0;
    
    // Memory Management Interface
    virtual bool allocateBackendMemory(int channel_num) = 0;
    virtual void deallocateBackendMemory(int channel_num) = 0;
    virtual void cleanupBackendSpecific() = 0;
    
    // Configuration Interface
    virtual void configureAlgorithmSpecific(const DeconvolutionConfig& config) = 0;
    
    // Configuration Access
    DeconvolutionConfig getConfig() const;
    void setConfig(const DeconvolutionConfig& config);
    
    // Optimal Backend Detection
    static bool isBetterBackend(const BackendCandidate& backend1, const BackendCandidate& backend2);
    static BackendCandidate detectOptimalBackend(const SystemRequirements& requirements);
    
    // Performance Optimization
    virtual bool optimizeForWorkload(const WorkloadAnalysis& analysis);
    virtual bool applyPerformanceOptimizations(bool apply_gpu_optimizations);
    
    // Progress Reporting
    virtual void setProgressCallback(ProgressCallback callback);
    virtual double getProgress() const;
    
    // Error Handling
    virtual ErrorState getBackendErrorState() const;
    virtual void setBackendErrorState(const ErrorState& state);
    
    // Version Information
    static std::string getBackendVersion();
    static std::string getBackendName(BackendType type);
    
    // Static Factory Methods
    static AlgorithmRegistry& getAlgorithmRegistry();
    static PerformanceMonitor& getPerformanceMonitor();
    static LogService& getLogService();
    
protected:
    // Helper Methods (Backend Access)
    template<typename T>
    bool allocateArray(T*& array, size_t size);
    template<typename T>
    void deallocateArray(T*& array);
    
    // Memory Management
    virtual void* allocateSystemMemory(size_t size);
    virtual void deallocateSystemMemory(void* ptr);
    
    // Performance Tracking
    struct PerformanceMetrics {
        double preprocessing_time;
        double algorithm_time;
        double postprocessing_time;
        double memory_usage;
        int error_count;
    };
    
    PerformanceMetrics& getCurrentMetrics();
    void updateProgress(double progress);
    
    // Configuration Management
    bool updateBackendConfig(const DeconvolutionConfig& config);
    bool validateConfig(const DeconvolutionConfig& config) const;
    
    // Error Handling
    void setError(ErrorCode code, const std::string& message);
    void clearErrors();
    bool hasErrors() const;
    
private:
    // Implementation Details
    std::unique_ptr<BaseDeconvolutionAlgorithmImpl> implementation_;
    DeconvolutionConfig current_config_;
    PerformanceMetrics current_metrics_;
    ErrorState error_state_;
    
    BackendType detected_backend_;
    bool auto_optimization_enabled_;
    
    ProgressCallback progress_callback_;
    double current_progress_;
    
    AlgorithmRegistry& algorithm_registry_;
    PerformanceMonitor& performance_monitor_;
    LogService& log_service_;
};
```

#### Key Methods

##### Constructor and Initialization

```cpp
/**
 * @brief Construct a new BaseDeconvolutionAlgorithmDerived object
 * 
 * Initializes the algorithm with default configuration and detects optimal backend.
 * Validates system requirements and configures logging and monitoring.
 */
BaseDeconvolutionAlgorithmDerived::BaseDeconvolutionAlgorithmDerived()
    : auto_optimization_enabled_(true),
      current_progress_(0.0),
      detected_backend_(AUTO_DETECT),
      implementation_(nullptr)
{
    initialization_mutex_.lock();
    
    // Initialize static services
    algorithm_registry_ = &AlgorithmRegistry::getInstance();
    performance_monitor_ = &PerformanceMonitor::getInstance();
    log_service_ = &LogService::getInstance();
    
    // Detect optimal backend for current system
    SystemRequirements system_requirements = detectSystemRequirements();
    BackendCandidate optimal_backend = detectOptimalBackend(system_requirements);
    
    detected_backend_ = optimal_backend.type;
    
    // Initialize base configuration
    current_config_.gpu = backendToString(detected_backend_);
    current_config_.auto_optimize = auto_optimization_enabled_;
    
    // Initialize error state
    error_state_.clear();
    
    // Initialize monitoring
    performance_monitor_->startMonitoring();
    
    initialization_mutex_.unlock();
    
    log_service_->log(LogLevel::INFO, 
                      "BaseDeconvolutionAlgorithmDerived initialized with backend: " + 
                      getBackendName(detected_backend_));
}

/**
 * @brief Destroy the BaseDeconvolutionAlgorithmDerived object
 * 
 * Performs cleanup specific to the derived class, deallocates memory,
 * and stops performance monitoring.
 */
BaseDeconvolutionAlgorithmDerived::~BaseDeconvolutionAlgorithmDerived()
{
    // Stop monitoring
    performance_monitor_->stopMonitoring();
    
    // Cleanup implementation
    if (implementation_) {
        implementation_->cleanupBackendSpecific();
    }
    
    // Deallocate memory
    cleanupBackendSpecific();
    
    log_service_->log(LogLevel::INFO, "BaseDeconvolutionAlgorithmDerived destructed");
}
```

##### Backend Interface Methods

```cpp
/**
 * @brief Preprocess channel data before algorithm execution
 * 
 * @param channel_num Channel number to preprocess
 * @param psf_index Index of PSF to use for preprocessing
 * @return true if preprocessing succeeded, false otherwise
 */
bool BaseDeconvolutionAlgorithmDerived::preprocess(int channel_num, int psf_index)
{
    PerformanceTimer timer("preprocess");
    
    // Update progress
    updateProgress(0.0);
    
    // Validate input
    if (!validatePreprocessInput(channel_num, psf_index)) {
        setError(ERROR_INVALID_INPUT, "Invalid channel or PSF index");
        return false;
    }
    
    // Check backend-specific preprocessing
    if (!preprocessBackendSpecific(channel_num, psf_index)) {
        log_service_->log(LogLevel::ERROR, "Backend-specific preprocessing failed");
        setError(ERROR_BACKEND_PREPROCESS, "Backend preprocessing failed");
        return false;
    }
    
    // Log completion
    log_service_->log(LogLevel::INFO, 
                     "Preprocessing completed for channel " + std::to_string(channel_num));
    
    timer.stop();
    current_metrics_.preprocessing_time = timer.getElapsedSeconds();
    
    // Update progress
    updateProgress(0.2);
    
    return true;
}

/**
 * @brief Main algorithm processing
 * 
 * @param data Hyperstack reference to process
 * @param channel_num Channel number to process
 * @param H FFTW complex array (PSF in frequency domain)
 * @param g FFTW complex array (input image in frequency domain)
 * @param f FFTW complex array (output image in frequency domain)
 */
void BaseDeconvolutionAlgorithmDerived::algorithm(Hyperstack& data, 
                                                int channel_num, 
                                                fftw_complex* H, 
                                                fftw_complex* g, 
                                                fftw_complex* f)
{
    PerformanceTimer timer("algorithm");
    
    // Update progress
    updateProgress(0.2);
    
    // Validate algorithm inputs
    if (!validateAlgorithmInputs(data, channel_num, H, g, f)) {
        setError(ERROR_INVALID_INPUT, "Invalid algorithm inputs");
        return;
    }
    
    // Log algorithm start
    log_service_->log(LogLevel::INFO, 
                     "Starting algorithm processing for channel " + std::to_string(channel_num));
    
    // Forward pass: algorithm-specific implementation
    forward Processing = [&]() {
        runForwardPass(data, channel_num, H, g, f);
    };
    
    // Backend-specific forward processing
    if (detected_backend_ == BackendType::CPU) {
        forward Processing();
    } else if (detected_backend_ == BackendType::GPU) {
        // Handle GPU-specific processing with fallback
        if (!executeGPUForwardPass(data, channel_num, H, g, f)) {
            log_service_->log(LogLevel::WARNING, "GPU processing failed, falling back to CPU");
            detected_backend_ = BackendType::CPU;
            forward Processing();
        }
    }
    
    timer.stop();
    current_metrics_.algorithm_time = timer.getElapsedSeconds();
    
    // Update progress
    updateProgress(0.8);
    
    // Log algorithm completion
    log_service_->log(LogLevel::INFO, 
                     "Algorithm processing completed for channel " + std::to_string(channel_num));
}

/**
 * @brief Postprocess channel data after algorithm execution
 * 
 * @param channel_num Channel number to postprocess
 * @param psf_index Index of PSF to use for postprocessing
 * @return true if postprocessing succeeded, false otherwise
 */
bool BaseDeconvolutionAlgorithmDerived::postprocess(int channel_num, int psf_index)
{
    PerformanceTimer timer("postprocess");
    
    // Update progress
    updateProgress(0.8);
    
    // Validate postprocess inputs
    if (!validatePostprocessInput(channel_num, psf_index)) {
        setError(ERROR_INVALID_INPUT, "Invalid channel or PSF index for postprocessing");
        return false;
    }
    
    // Check backend-specific postprocessing
    if (!postprocessBackendSpecific(channel_num, psf_index)) {
        log_service_->log(LogLevel::ERROR, "Backend-specific postprocessing failed");
        setError(ERROR_BACKEND_POSTPROCESS, "Backend postprocessing failed");
        return false;
    }
    
    // Log completion
    log_service_->log(LogLevel::INFO, 
                     "Postprocessing completed for channel " + std::to_string(channel_num));
    
    timer.stop();
    current_metrics_.postprocessing_time = timer.getElapsedSeconds();
    
    // Update progress
    updateProgress(1.0);
    
    return true;
}
```

##### Memory Management Interface

```cpp
/**
 * @brief Allocate backend-specific memory for a channel
 * 
 * @param channel_num Channel number to allocate memory for
 * @return true if allocation succeeded, false otherwise
 */
bool BaseDeconvolutionAlgorithmDerived::allocateBackendMemory(int channel_num)
{
    ChannelMemory& channel_mem = channel_memory_[channel_num];
    
    // Check if already allocated
    if (channel_mem.allocated) {
        log_service_->log(LogLevel::WARNING, 
                         "Memory already allocated for channel " + std::to_string(channel_num));
        return true;
    }
    
    // Get channel dimensions
    std::tuple<int, int, int> dimensions = getImageDimensions();
    size_t volume = std::get<0>(dimensions) * std::get<1>(dimensions) * std::get<2>(dimensions);
    
    // Allocate arrays based on backend
    switch (detected_backend_) {
        case BackendType::CPU:
            channel_mem.forward_array = allocateArray<fftw_complex>(volume);
            channel_mem.inverse_array = allocateArray<fftw_complex>(volume);
            channel_mem.temp_array = allocateArray<fftw_complex>(volume);
            break;
            
        case BackendType::GPU:
            channel_mem.forward_array_gpu = allocateArray<cuComplex>(volume);
            channel_mem.inverse_array_gpu = allocateArray<cuComplex>(volume);
            channel_mem.temp_array_gpu = allocateArray<cuComplex>(volume);
            break;
            
        default:
            setError(ERROR_INVALID_BACKEND, "Unknown backend type");
            return false;
    }
    
    // Mark as allocated and record dimensions
    channel_mem.allocated = true;
    channel_mem.dimensions = dimensions;
    
    log_service_->log(LogLevel::INFO, 
                     "Allocated backend memory for channel " + std::to_string(channel_num));
    
    return true;
}

/**
 * @brief Deallocate backend-specific memory for a channel
 * 
 * @param channel_num Channel number to deallocate memory for
 */
void BaseDeconvolutionAlgorithmDerived::deallocateBackendMemory(int channel_num)
{
    ChannelMemory& channel_mem = channel_memory_[channel_num];
    
    // Check if allocated
    if (!channel_mem.allocated) {
        log_service_->log(LogLevel::WARNING, 
                         "Memory not allocated for channel " + std::to_string(channel_num));
        return;
    }
    
    // Deallocate arrays based on backend
    switch (detected_backend_) {
        case BackendType::CPU:
            deallocateArray(channel_mem.forward_array);
            deallocateArray(channel_mem.inverse_array);
            deallocateArray(channel_mem.temp_array);
            break;
            
        case BackendCase::GPU:
            deallocateArray(channel_mem.forward_array_gpu);
            deallocateArray(channel_mem.inverse_array_gpu);
            deallocateArray(channel_mem.temp_array_gpu);
            break;
    }
    
    // Mark as deallocated
    channel_mem.allocated = false;
    channel_mem.dimensions = {0, 0, 0};
    
    log_service_->log(LogLevel::INFO, 
                     "Deallocated backend memory for channel " + std::to_string(channel_num));
}

/**
 * @ Cleanup backend-specific resources
 */
void BaseDeconvolutionAlgorithmDerived::cleanupBackendSpecific()
{
    // Deallocate all channel memory
    for (auto& [channel_num, mem] : channel_memory_) {
        if (mem.allocated) {
            deallocateBackendMemory(channel_num);
        }
    }
    
    // Clear performance metrics
    current_metrics_ = PerformanceMetrics();
    
    log_service_->log(LogLevel::INFO, "Backend-specific cleanup completed");
}
```

##### Configuration Management

```cpp
/**
 * @brief Configure algorithm-specific parameters
 * 
 * @param config Deconvolution configuration to apply
 */
void BaseDeconvolutionAlgorithmDerived::configureAlgorithmSpecific(const DeconvolutionConfig& config)
{
    PerformanceTimer timer("configure_algorithm");
    
    // Validate configuration
    if (!validateConfig(config)) {
        log_service_->log(LogLevel::ERROR, "Invalid configuration for algorithm");
        setError(ERROR_INVALID_CONFIG, "Invalid configuration parameters");
        return;
    }
    
    // Update current configuration
    current_config_ = config;
    
    // Apply backend-specific configuration changes
    if (detected_backend_ == BackendType::GPU) {
        applyGPUConfigOptimizations(config);
    } else {
        applyCPUConfigOptimizations(config);
    }
    
    // Apply auto-optimization if enabled
    if (auto_optimization_enabled_ && config.auto_optimize) {
        optimizeForWorkload(analyzeCurrentWorkload());
    }
    
    timer.stop();
    
    log_service_->log(LogLevel::INFO, "Algorithm configuration updated");
}

/**
 * @brief Update backend-specific configuration
 * 
 * @param config Configuration to update with
 * @return true if update succeeded, false otherwise
 */
bool BaseDeconvolutionAlgorithmDerived::updateBackendConfig(const DeconvolutionConfig& config)
{
    // Validate backend compatibility
    if (!isBackendCompatible(detected_backend_, config)) {
        setError(ERROR_INCOMPATIBLE_BACKEND, 
                "Configuration not compatible with current backend");
        return false;
    }
    
    // Store old configuration for rollback if needed
    DeconvolutionConfig old_config = current_config_;
    
    try {
        // Update backend-specific settings
        updateBackendSpecificConfig(config);
        
        // Verify update succeeded
        if (!validateBackendConfig(detected_backend_, config)) {
            // Rollback to old configuration
            current_config_ = old_config;
            updateBackendSpecificConfig(old_config);
            
            return false;
        }
        
        // Update current configuration
        current_config_ = config;
        
        return true;
        
    } catch (const std::exception& e) {
        // Catch any exceptions and rollback
        log_service_->log(LogLevel::ERROR, "Error updating backend config: " + std::string(e.what()));
        current_config_ = old_config;
        updateBackendSpecificConfig(old_config);
        
        return false;
    }
}
```

##### System Requirements and Backend Detection

```cpp
/**
 * @brief Check system requirements for different backends
 * 
 * @return SystemRequirements Structure containing system capability information
 */
SystemRequirements BaseDeconvolutionAlgorithmDerived::detectSystemRequirements()
{
    SystemRequirements requirements;
    
    // Detect CPU capabilities
    requirements.cpu_cores = std::thread::hardware_concurrency();
    requirements.cpu_turbo = hasTurboBoost();
    requirements.cpu_avx2 = hasAVX2Support();
    
    // Detect memory capabilities
    requirements.total_memory = getSystemMemoryGB();
    requirements.available_memory = getAvailableMemoryGB();
    
    // Detect GPU capabilities
    requirements.has_gpu = hasGPUSupport();
    if (requirements.has_gpu) {
        requirements.gpu_memory = getGPUMemoryGB();
        requirements.cuda_version = getCUDAVersion();
        requirements.cuda_available = hasCUDAAvailability();
    }
    
    // Detect FFTW capabilities
    requirements.fftw_optimized = isFFTWOptimized();
    requirements.fftw_wisdom = hasFFTWWisdom();
    
    return requirements;
}

/**
 * @brief Detect optimal backend based on system requirements and configuration
 * 
 * @param requirements System requirements struct
 * @return BackendCandidate Information about optimal backend configuration
 */
BackendCandidate BaseDeconvolutionAlgorithmDerived::detectOptimalBackend(
    const SystemRequirements& requirements)
{
    BackendCandidate candidate;
    
    // Calculate backend scores for each available backend
    std::vector<BackendScore> scores;
    
    // Calculate CPU score
    double cpu_score = calculateCPUScore(requirements);
    scores.push_back({BackendType::CPU, cpu_score});
    
    // Calculate GPU score if available
    if (requirements.has_gpu && requirements.cuda_available) {
        double gpu_score = calculateGPUScore(requirements);
        scores.push_back({BackendType::GPU, gpu_score});
    }
    
    // Select best backend
    auto best_it = std::max_element(scores.begin(), scores.end(),
                                   [](const BackendScore& a, const BackendScore& b) {
                                       return a.score < b.score;
                                   });
    
    candidate.type = best_it->type;
    candidate.score = best_it->score;
    candidate.is_auto = true;  // Auto-detected
    candidate.confidence = calculateBackendConfidence(candidate.type, requirements);
    
    // Set configuration recommendations
    candidate.recommended_config = generateRecommendedConfig(candidate.type, requirements);
    
    return candidate;
}

/**
 * @brief Compare two backend candidates for selection
 * 
 * @param backend1 First backend candidate
 * @param backend2 Second backend candidate
 * @return true if backend1 is better than backend2
 */
bool BaseDeconvolutionAlgorithmDerived::isBetterBackend(
    const BackendCandidate& backend1, 
    const BackendCandidate& backend2)
{
    // Compare scores with confidence weighting
    double score1 = backend1.score * (1.0 + backend1.confidence * 0.2);  // Add 20% for high confidence
    double score2 = backend2.score * (1.0 + backend2.confidence * 0.2);
    
    return score1 > score2;
}
```

##### Error Handling and Logging

```cpp
/**
 * @brief Set error state with code and message
 * 
 * @param code Error code from ErrorCode enum
 * @param message Error message
 */
void BaseDeconvolutionAlgorithmDerived::setError(ErrorCode code, const std::string& message)
{
    ErrorState& error_state = getBackendErrorState();
    
    error_state.code = code;
    error_state.message = message;
    error_state.timestamp = std::chrono::system_clock::now();
    error_state.backend_type = detected_backend_;
    
    // Log error
    log_service_->log(LogLevel::ERROR, 
                     "Error " + std::to_string(static_cast<int>(code)) + ": " + message);
    
    // Update metrics
    current_metrics_.error_count++;
}

/**
 * @brief Clear all error states
 */
void BaseDeconvolutionAlgorithmDerived::clearErrors()
{
    error_state_.clear();
    log_service_->log(LogLevel::INFO, "Error state cleared");
}

/**
 * @brief Check if there are any errors
 * 
 * @return true if errors exist, false otherwise
 */
bool BaseDeconvolutionAlgorithmDerived::hasErrors() const
{
    return error_state_.code != ErrorCode::NO_ERROR ||
           !error_state_.message.empty();
}

/**
 * @brief Get current error state
 * 
 * @return ErrorState const& Reference to current error state
 */
ErrorState& BaseDeconvolutionAlgorithmDerived::getBackendErrorState()
{
    return error_;
}

/**
 * @brief Set error state with custom state
 * 
 * @param state New error state to set
 */
void BaseDeconvolutionAlgorithmDerived::setBackendErrorState(const ErrorState& state)
{
    error_state_ = state;
    
    if (error_state_.code != ErrorCode::NO_ERROR) {
        log_service_->log(LogLevel::ERROR, 
                         "Backend error: " + error_state_.message);
    }
}
```

### BaseDeconvolutionAlgorithmCPU

CPU-specific backend implementation derived from the base architecture class.

**File:** [`include/deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h`](include/deconvolution/algorithms/BaseDeconvolutionAlgorithmCPU.h)

#### Class Definition

```cpp
class BaseDeconvolutionAlgorithmCPU : public BaseDeconvolutionAlgorithmDerived
{
public:
    // Constructor and Destructor
    BaseDeconvolutionAlgorithmCPU();
    virtual ~BaseDeconvolutionAlgorithmCPU();
    
    // Backend Interface Implementation
    virtual bool preprocessBackendSpecific(int channel_num, int psf_index) override;
    virtual void algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override;
    virtual bool postprocessBackendSpecific(int channel_num, int psf_index) override;
    
    // Memory Management Implementation
    virtual bool allocateBackendMemory(int channel_num) override;
    virtual void deallocateBackendMemory(int channel_num) override;
    virtual void cleanupBackendSpecific() override;
    
    // Configuration Implementation
    virtual void configureAlgorithmSpecific(const DeconvolutionConfig& config) override;
    
    // FFTW Management
    bool initializeFFTWPlans();
    void cleanupFFTWPlans();
    bool reuseFFTWPlans(bool reuse);
    
    // Memory Pool Management
    bool enableMemoryPool(bool enable);
    bool configureMemoryPool(size_t pool_size);
    
    // CPU Optimization
    bool optimizeForCPUArchitecture(const CPUArchitecture& arch);
    bool configureOpenMPSchedule(const OpenMPConfig& config);
    
    // Memory Allocation Helpers
    template<typename T>
    T* allocateCPUArray(size_t size, bool aligned = true);
    template<typename T>
    void deallocateCPUArray(T* array);
    
    // FFT execution helpers
    bool executeForwardFFT(fftw_complex* input, fftw_complex* output);
    bool executeInverseFFT(fftw_complex* input, fftw_complex* output);
    
    // Diagnostics
    void runCPUPerformanceDiagnostics();
    CPUPerformanceMetrics getCPUPerformanceMetrics() const;
    
protected:
    // FFTW Plan Management
    struct FFTWPlan {
        fftw_plan forward_plan;
        fftw_plan inverse_plan;
        fftw_plan temp_plan;
        bool initialized;
        std::tuple<int, int, int> dimensions;
    };
    
    // Memory Pool Management
    struct MemoryPoolEntry {
        void* buffer;
        size_t size;
        size_t allocated;
        bool in_use;
    };
    
    std::map<int, FFTWPlan> fftw_plans_;
    std::vector<MemoryPoolEntry> memory_pool_;
    size_t memory_pool_size_;
    bool memory_pool_enabled_;
    
    CPUPerformanceMetrics current_metrics_;
    
    // CPU Architecture Information
    CPUArchitecture detected_arch_;
    
    // OpenMP Configuration
    OpenMPConfig omp_config_;
    
    // FFTW Initialization
    bool initializeFFTWForDimensions(const std::tuple<int, int, int>& dimensions);
    bool createFFTWPlan(const std::tuple<int, int, int>& dimensions, FFTWPlan& plan);
    
    // Memory Pool Implementation
    bool findMemoryPoolEntry(size_t required_size, size_t& pool_index);
    bool allocateFromPool(size_t size, size_t& pool_index);
    void releaseFromPool(size_t pool_index);
    
    // CPU Optimization Helpers
    bool optimizeForCacheSize(size_t cache_size);
    bool optimizeForMemoryBottlenecks();
    
private:
    // Disallow copy construction and assignment
    BaseDeconvolutionAlgorithmCPU(const BaseDeconvolutionAlgorithmCPU&) = delete;
    BaseDeconvolutionAlgorithmCPU& operator=(const BaseDeconvolutionAlgorithmCPU&) = delete;
};
```

#### Key Methods

##### FFTW Plan Management

```cpp
/**
 * @brief Initialize FFTW plans for current dimensions
 * 
 * @return true if initialization succeeded, false otherwise
 */
bool BaseDeconvolutionAlgorithmCPU::initializeFFTWPlans()
{
    try {
        // Get current image dimensions
        std::tuple<int, int, int> dimensions = getCurrentImageDimensions();
        
        // Check if plans already exist for these dimensions
        if (fftw_plans_[0].initialized && 
            fftw_plans_[0].dimensions == dimensions) {
            log_service_->log(LogLevel::INFO, "FFTW plans already initialized for current dimensions");
            return true;
        }
        
        // Create new FFTW plans
        if (!createFFTWPlan(dimensions, fftw_plans_[0])) {
            setError(ERROR_FFTW_INITIALIZATION, "Failed to create FFTW plans");
            return false;
        }
        
        log_service_->log(LogLevel::INFO, "FFTW plans initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        setError(ERROR_FFTW_INITIALIZATION, 
                "FFT );


[Response interrupted by API Error]