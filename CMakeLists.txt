cmake_minimum_required(VERSION 3.10)
project(Deconvtool)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp -march=native")

# Include directories and libraries for FFTW, OpenCV, OpenMP, and TIFF
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TIFF REQUIRED)
find_package(CUDAToolkit REQUIRED)


# Common include dirs
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
# ------------------------------------------------------------
# Collect sources
# ------------------------------------------------------------
set(COMMON_SOURCES
    src/Dolphin.cpp
    src/algorithms/ConvolutionAlgorithm.cpp
    src/hyperstack/HyperstackImage.cpp
    src/hyperstack/Image3D.cpp
    src/hyperstack/ImageMetaData.cpp
    src/hyperstack/HyperstackIO.cpp
    src/hyperstack/HyperstackMetaData.cpp
    src/psf/PSF.cpp
    src/psf/PSFConfig.cpp
    src/psf/algorithms/SimpleGaussianPSFGeneratorAlgorithm.cpp
    src/psf/algorithms/GaussianPSFGenerator.cpp
    src/psf/algorithms/BornWolfModel.cpp
    src/psf/algorithms/GibsonLanniPSFGenerator.cpp
    src/psf/algorithms/KirchhoffDiffractionSimpson.cpp
    src/psf/configs/GibsonLanniPSFConfig.cpp
    src/psf/configs/GaussianPSFConfig.cpp
    src/deconvolution/BaseDeconvolutionAlgorithm.cpp
    src/deconvolution/DeconvolutionConfig.cpp
    src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
    src/utilities/UtlFFT.cpp
    src/utilities/UtlGrid.cpp
    src/utilities/UtlImage.cpp
    src/utilities/UtlIO.cpp
)

# ------------------------------------------------------------
# Build shared library containing all algorithms
# ------------------------------------------------------------
add_library(dolphinlib ${COMMON_SOURCES})

target_link_libraries(dolphinlib
    PUBLIC
        ${OpenCV_LIBS}
        ${TIFF_LIBRARIES}
        OpenMP::OpenMP_CXX
        fftw3_threads
        fftw3
)

target_include_directories(dolphinlib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# ------------------------------------------------------------
# CPU executable
# ------------------------------------------------------------
add_executable(dolphin src/main.cpp)
target_link_libraries(dolphin PRIVATE dolphinlib pthread gomp stdc++fs)

# ------------------------------------------------------------
# GPU executable (only if CUDA found)
# ------------------------------------------------------------
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found, building dolphincuda with CUBE.")

    # Add CUBE subproject
    add_subdirectory(${CMAKE_SOURCE_DIR}/lib/cube)

    add_executable(dolphincuda src/main.cpp)
    target_link_libraries(dolphincuda
        PRIVATE
            dolphinlib
            pthread
            gomp
            stdc++fs
            CUBE          # provided by lib/cube/CMakeLists.txt
            CUDA::cufft
            CUDA::cudart
            CUDA::cufftw
    )
    target_compile_definitions(dolphincuda PRIVATE CUDA_AVAILABLE)
else()
    message(STATUS "CUDA not found. Building only CPU version.")
endif()
