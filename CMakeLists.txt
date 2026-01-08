cmake_minimum_required(VERSION 3.10)
project(Deconvtool VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set build type to Release by default for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags for different build types
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma")
# Include directories and libraries for FFTW, OpenMP, and TIFF
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags for Valgrind-safe build
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     add_compile_options(
#         -O0             # Disable optimizations
#         -g              # Debug symbols
#         -march=x86-64   # Only baseline instructions
#         -mno-avx        # Disable AVX
#         -mno-avx2       # Disable AVX2
#         -mno-fma        # Disable FMA
#         -Wall           # Enable warnings
#         -Wextra
#     )
# endif()

# link_directories(/usr/local/lib)
find_package(OpenMP REQUIRED)
find_package(TIFF REQUIRED)
find_package(ITK REQUIRED)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# ------------------------------------------------------------
# Collect sources
# ------------------------------------------------------------

# Create Dolphin library with full API
set(DOLPHIN_SOURCES
    src/Dolphin.cpp
    src/PSFCreator.cpp
    src/ServiceFactory.cpp
    src/PSFGenerationService.cpp
    src/DeconvolutionService.cpp
    src/DeconvolutionStrategyFactory.cpp
    src/ThreadPool.cpp
    src/backend/BackendFactory.cpp
    
    src/hyperstack/HyperstackImage.cpp
    src/hyperstack/Image3D.cpp
    src/hyperstack/ImageMetaData.cpp
    src/hyperstack/HyperstackIO.cpp
    src/hyperstack/HyperstackMetaData.cpp

    src/IO/TiffReader.cpp
    src/IO/TiffWriter.cpp

    src/psf/PSF.cpp
    src/psf/generators/GaussianPSFGenerator.cpp
    src/psf/generators/BornWolfModel.cpp
    src/psf/generators/GibsonLanniPSFGenerator.cpp
    src/psf/generators/SimpsonIntegrator.cpp
    src/psf/configs/PSFConfig.cpp
    src/psf/configs/GibsonLanniPSFConfig.cpp
    src/psf/configs/GaussianPSFConfig.cpp

    src/deconvolution/DeconvolutionConfig.cpp
    src/deconvolution/DeconvolutionAlgorithmFactory.cpp
    src/deconvolution/Preprocessor.cpp
    src/deconvolution/Postprocessor.cpp
    src/deconvolution/DeconvolutionProcessor.cpp
    
    src/deconvolution/deconvolutionStrategies/ImageSplit.cpp
    src/deconvolution/deconvolutionStrategies/DeconvolutionStrategyPair.cpp
    src/deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.cpp
    src/deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.cpp
    src/deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.cpp

    src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/TestAlgorithm.cpp


    src/frontend/SetupConfig.cpp
)



# Create backend library
set(BACKEND_SOURCES
    src/backend/ComplexData.cpp)
add_library(backendlib SHARED ${BACKEND_SOURCES})

target_include_directories(backendlib
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/backend>
        $<INSTALL_INTERFACE:include/backend>)




# Create shared library for Dolphin API
add_library(dolphin SHARED ${DOLPHIN_SOURCES})

target_include_directories(dolphin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${TIFF_INCLUDE_DIR}
        ${ITK_INCLUDE_DIRS}
)

target_link_libraries(dolphin
    PUBLIC
        backendlib
        ${TIFF_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${ITK_LIBRARIES}
)

# Set version for the library
set_target_properties(dolphin PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Install targets
install(TARGETS dolphin backendlib
    EXPORT dolphinTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/ 
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Create and install the config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets
export(EXPORT dolphinTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/dolphinTargets.cmake"
    NAMESPACE dolphin::
)

# Configure the config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dolphinConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinConfig.cmake"
    INSTALL_DESTINATION lib/cmake/dolphin
)

# Install the config files
install(EXPORT dolphinTargets
    FILE dolphinTargets.cmake
    NAMESPACE dolphin::
    DESTINATION lib/cmake/dolphin
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinConfigVersion.cmake"
    DESTINATION lib/cmake/dolphin
)






option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})
     # Ensure backends are built before tests
    # add_subdirectory(backends/cpu)
    # if(CUDA_AVAILABLE)
    #     add_subdirectory(backends/cuda)
    # endif()
    set(CMAKE_CTEST_ARGUMENTS "--verbose")


    enable_testing()
    add_subdirectory(tests)

endif()



# find_package(CUDAToolkit QUIET)
# if(CUDAToolkit_FOUND)
#     message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
#     set(CUDA_AVAILABLE TRUE)
# else()
#     message(STATUS "CUDA toolkit not found. Building CPU-only version.")
#     set(CUDA_AVAILABLE FALSE)
# endif()




# ------------------------------------------------------------
# CPU executable (using CLI frontend)
# ------------------------------------------------------------
# add_executable(dolphin src/main.cpp)
# target_link_libraries(dolphin
#     PRIVATE
#         dolphin
#         pthread
#         gomp
#         stdc++fs)

# add_subdirectory(backends/cpu)
# add_subdirectory(backends/openmp)

# ------------------------------------------------------------
# GPU executable (only if CUDA found)
# ------------------------------------------------------------
# if(CUDA_AVAILABLE)
#     enable_language(CUDA)
#     message(STATUS "CUDA found, building dolphincuda with CUBE.")
#     add_compile_definitions(CUDA_AVAILABLE)
#     # add_subdirectory(backends/cuda)
#     # Add CUBE subproject
#     # target_link_libraries(cuda_backend
#     #     PUBLIC
#     #         dolphinlib  # cudalib depends on dolphinlib headers
#     # )

# else()
#     message(STATUS "CUDA not found. Building only CPU version.")
# endif()