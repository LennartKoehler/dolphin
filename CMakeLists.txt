cmake_minimum_required(VERSION 3.10)
project(Deconvtool VERSION 1.0.0)


# set(CMAKE_BUILD_TYPE NoVectorization)
# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)



# Optimization flags for different build types
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -march=native")
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma")
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma")
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(project_options INTERFACE)

target_compile_options(project_options INTERFACE
  # Debug
  $<$<CONFIG:Debug>:-g;-O0;-fno-omit-frame-pointer>
  # DebugVectorization
  $<$<CONFIG:DebugVectorization>:-g;-O0;-fno-omit-frame-pointer>
  #NoVectorization
  $<$<CONFIG:NoVectorization>:-O3;-ffast-math;-DNDEBUG;-fno-tree-vectorize;-fno-tree-slp-vectorize>
  # Release
  $<$<CONFIG:Release>:-O3;-ffast-math;-DNDEBUG;-funroll-loops;-mavx2;-mfma>
  # RelWithDebInfo
  $<$<CONFIG:RelWithDebInfo>:-O3;-g;-ffast-math;-DNDEBUG;-funroll-loops>
)

# architecture flags only for Release (avoid -march=native for Debug)
target_compile_options(project_options INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-mavx2;-mfma>
)

# link_directories(/usr/local/lib)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so" ".dll" ".dylib")
find_package(TIFF REQUIRED)
find_package(ITK REQUIRED)

set(FFTW_PATH "/usr/local/fftw-debug/lib" CACHE PATH "Path to FFTW installation")
message(STATUS "Using the followinf FFTW lib: ${FFTW_PATH}")

message(STATUS "ITK version: ${ITK_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "ITK_BUILD_TYPE: ${ITK_BUILD_TYPE}")
message(STATUS "ITK_DIR: ${ITK_DIR}")
get_target_property(itk_release ITKCommon IMPORTED_LOCATION_RELEASE)
get_target_property(itk_debug   ITKCommon IMPORTED_LOCATION_DEBUG)

message(STATUS "ITKCommon RELEASE: ${itk_release}")
message(STATUS "ITKCommon DEBUG:   ${itk_debug}")

add_subdirectory(backends/dolphinbackend)
# add_subdirectory(backends/openmp)
add_subdirectory(backends/cpu)



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
    src/Config.cpp
    src/SetupConfig.cpp
    
    src/image/Image3D.cpp
    src/image/ImageMetaData.cpp

    src/backend/BackendFactory.cpp

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

    src/deconvolution/algorithms/ConvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/TestAlgorithm.cpp
)



# Create shared library for Dolphin API
add_library(dolphin STATIC ${DOLPHIN_SOURCES})


target_link_libraries(dolphin PUBLIC project_options)



target_include_directories(dolphin
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends/cpu/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backends/dolphinbackend/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/nlohmann/json/single_include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lib/spdlog/include/>
        $<INSTALL_INTERFACE:include>
        ${TIFF_INCLUDE_DIR}
        ${ITK_INCLUDE_DIRS}
)

target_link_libraries(dolphin
    PUBLIC
        dolphinbackend
        cpu_backend
        ${TIFF_LIBRARIES}
        ${ITK_LIBRARIES}
)

# Compiler flags for Valgrind-safe build
# if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options( dolphin PROPERTIES
#         -O0             # Disable optimizations
#         -g              # Debug symbols
#         -march=x86-64   # Only baseline instructions
#         -mno-avx        # Disable AVX
#         -mno-avx2       # Disable AVX2
#         -mno-fma        # Disable FMA
#         -Wall           # Enable warnings
#         -Wextra
#         -g
#         -O0
#         -fno-omit-frame-pointer
#     )
# endif()


# Set version for the library
set_target_properties(dolphin PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
set_target_properties(dolphin PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_RPATH "$ORIGIN/../lib"
)



# Install targets
install(TARGETS project_options dolphin dolphinbackend cpu_backend
    EXPORT dolphinTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY lib/nlohmann/json/single_include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(DIRECTORY lib/dolphinbackend/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)
install(DIRECTORY include/dolphin
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
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

    set(CMAKE_CTEST_ARGUMENTS "--verbose")


    enable_testing()
    add_subdirectory(tests)
endif()


