# Code Mode Rules - DOLPHIN Project

## Project-Specific Coding Patterns

### Configuration System (Critical)
```cpp
// Always inherit from Config base class
struct MyConfig : public Config {
    // Use readParameter<T>() for REQUIRED fields (throws if missing)
    int requiredField = readParameter<int>(jsonData, "fieldName");
    
    // Use readParameterOptional<T>() for optional fields
    std::optional<int> optionalField;
    readParameterOptional(jsonData, "fieldName", optionalField);
};

// ALWAYS use CreateFromJSONFile() for loading
auto config = MyConfig::createFromJSONFile("path/to/config.json");
```

### Algorithm Registration Pattern
```cpp
// In factory constructor ONLY
void registerAlgorithm(const std::string& name, AlgorithmCreator creator) {
    algorithms_[name] = creator;
}

// Register like this:
registerAlgorithm("RichardsonLucy", []() {
    return std::make_unique<RLDeconvolutionAlgorithm>();
});
// NOT: registerAlgorithm("rl", ...) - must match CLI args exactly
```

### PSF Management (Non-Obvious)
```cpp
// Use PSFManager, don't instantiate PSF directly
PSFManager psfmanager;
PSFPackage psfpackage = psfmanager.handleSetupConfig(*config);

// PSF arrays resolve conflicts by index order (lower index wins)
// Position [10,5] with two PSFs = only first PSF applies
```

### Memory Management Pattern
```cpp
// Always use shared_ptr for PSFs and algorithms
std::shared_ptr<BaseDeconvolutionAlgorithm> algorithm = DAF.create(deconvConfig->algorithmName, *deconvConfig);

// Hyperstack copying uses copy constructor extensively
Hyperstack result = deconvAlgorithm->run(hyperstack, psfpackage.psfs);
```

### Exception Handling Rules
- Configuration loading throws `std::runtime_error` with field name for missing required params
- Never catch exceptions in main processing pipeline - let them propagate to CLI/GUI frontend
- GPU compilation errors indicate CUDA build order problem (build cube library first)

### File I/O Patterns
```cpp
// TIFF reading auto-detects file type
if (imagePath.substr(imagePath.find_last_of(".") + 1) == "tif") {
    // Single file
} else {
    // Directory
}

// Results ALWAYS go to ../result/ relative to binary location
result.saveAsTifFile("../result/deconv.tif");
```

## Architecture Constraints

### Factory Pattern Dependencies
- Adding algorithms requires updating BOTH factory registration AND CLI option parsing
- Algorithm names are global constants defined in factory constructor
- Cannot add algorithms at runtime - only via code changes

### GPU Build Dependencies
- CUDA version requires building `lib/cube` subproject FIRST
- Conditional compilation with `#ifdef CUDA_AVAILABLE` guards GPU-specific code
- GPU architecture hard-coded to "75;80;90" in CMake

### Configuration Inheritance
- All config classes inherit from `Config` base for template-based parameter loading
- JSON loading paths are relative to CMAKE_BINARY_DIR, not project root
- Deconvolution section must exist in JSON even when using defaults

## Performance-Critical Patterns

### OpenMP Usage
```cpp
// Project uses OpenMP for parallelization - ensure -fopenmp flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp -march=native")
```

### Subimage Processing
```cpp
// subimageSize = 0 auto-adjusts to PSF size (recommended)
// Manual subimage sizing requires careful PSF position mapping
algorithm->run(hyperstack, psfpackage.psfs); // Handles both cases
```

## Code Style Requirements

### Hungarian Notation in Configs
```cpp
// Project uses Hungarian notation in configuration structs (violates modern C++)
int subimageSize; // NOT: sub_image_size
std::string psfConfigPath; // NOT: psf_config_path
double lambda; // NOT: regularization_lambda
```

### Header/Source Separation
```cpp
// Headers contain minimal implementation
// All actual implementations go to .cpp files (unusual for C++ projects)
// Header: declare interface, .cpp: implement all logic
```

### Template-Based Parameter Loading
```cpp
// Never parse JSON manually in config classes
// Always use Config base class templates:
// readParameter<T>() for required, readParameterOptional<T>() for optional
```

### Factory Registration Order
```cpp
// Algorithm registration must happen in constructor, not during runtime
// Factory is singleton - register algorithms at static initialization time
DeconvolutionAlgorithmFactory::getInstance() // Static initialization