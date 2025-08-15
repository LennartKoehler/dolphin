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
if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found. Compiling with CUBE library. Creating second executable.")
else()
    message(STATUS "CUDA not found. Compiling without CUBE library.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${TIFF_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include)

# Define source files for CPU executable

# LK make these into their own library, so with or without cuda can just import that library
add_executable(dolphin ${CMAKE_SOURCE_DIR}/src/main.cpp
        src/Dolphin.cpp
        include/Dolphin.h
        src/algorithms/ConvolutionAlgorithm.cpp
        src/hyperstack/HyperstackImage.cpp
        include/HyperstackImage.h
        src/hyperstack/Image3D.cpp
        include/Image3D.h
        src/hyperstack/ImageMetaData.cpp
        include/ImageMetaData.h
        src/psf/PSF.cpp
        include/PSF.h
        include/BasePSFGenerator.h
        src/psf/algorithms/SimpleGaussianPSFGeneratorAlgorithm.cpp
        include/SimpleGaussianPSFGeneratorAlgorithm.h
        include/PSFGeneratorFactory.h
        include/DeconvolutionAlgorithmFactory.h
        include/RegularizedInverseFilterDeconvolutionAlgorithm.h
        include/RLDeconvolutionAlgorithm.h
        src/deconvolution/DeconvolutionConfig.cpp
        include/DeconvolutionConfig.h
        include/BaseDeconvolutionAlgorithm.h
        src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
        src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
        src/hyperstack/HyperstackIO.cpp
        src/hyperstack/HyperstackMetaData.cpp
        src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
        include/InverseFilterDeconvolutionAlgorithm.h
        include/UtlFFT.h
        src/utilities/UtlFFT.cpp
        include/UtlGrid.h
        src/utilities/UtlGrid.cpp
        include/UtlImage.h
        src/utilities/UtlImage.cpp
        include/UtlIO.h
        src/utilities/UtlIO.cpp
        include/Channel.h
        src/deconvolution/BaseDeconvolutionAlgorithm.cpp
        include/GaussianPSFGenerator.h
        src/psf/algorithms/GaussianPSFGenerator.cpp
        src/psf/algorithms/BornWolfModel.cpp
        include/BornWolfModel.h
        include/RLTVDeconvolutionAlgorithm.h
        src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
        src/psf/algorithms/GibsonLanniPSFGenerator.cpp
        src/psf/configs/GibsonLanniPSFConfig.cpp
        src/psf/configs/GaussianPSFConfig.cpp


        include/GibsonLanniPSFGenerator.h
        include/RLADDeconvolutionAlgorithm.h
        src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
        include/PSFConfig.h
        
)
# Link CPU executable with the required libraries (FFTW)
target_link_libraries(dolphin
        -lpthread
        -lgomp
        ${OpenCV_LIBS}
        stdc++fs
        ${TIFF_LIBRARIES}
        OpenMP::OpenMP_CXX
        fftw3
        fftw3_threads
)

# Create second executable only if CUDA available
if(CUDAToolkit_FOUND)
    # Include the CUBE dir
    add_subdirectory(${CMAKE_SOURCE_DIR}/lib/cube)
    link_directories(${CMAKE_SOURCE_DIR}/lib/cube/build)
    include_directories(${CMAKE_SOURCE_DIR}/lib/cube/include)
    # Define source files for GPU executable
    add_executable(dolphincuda ${CMAKE_SOURCE_DIR}/src/main.cpp
                src/Dolphin.cpp
                include/Dolphin.h
                src/algorithms/ConvolutionAlgorithm.cpp
                src/hyperstack/HyperstackImage.cpp
            include/HyperstackImage.h
            src/hyperstack/Image3D.cpp
            include/Image3D.h
            src/hyperstack/ImageMetaData.cpp
            include/ImageMetaData.h
            src/psf/PSF.cpp
            include/PSF.h
            include/BasePSFGenerator.h
            src/psf/algorithms/SimpleGaussianPSFGeneratorAlgorithm.cpp
            include/SimpleGaussianPSFGeneratorAlgorithm.h
            include/PSFGeneratorFactory.h
            include/DeconvolutionAlgorithmFactory.h
            include/RegularizedInverseFilterDeconvolutionAlgorithm.h
            include/RLDeconvolutionAlgorithm.h
            src/deconvolution/DeconvolutionConfig.cpp
            include/DeconvolutionConfig.h
            include/BaseDeconvolutionAlgorithm.h
            src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
            src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
            src/hyperstack/HyperstackIO.cpp
            src/hyperstack/HyperstackMetaData.cpp
            src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
            include/InverseFilterDeconvolutionAlgorithm.h
            include/UtlFFT.h
            src/utilities/UtlFFT.cpp
            include/UtlGrid.h
            src/utilities/UtlGrid.cpp
            include/UtlImage.h
            src/utilities/UtlImage.cpp
            include/UtlIO.h
            src/utilities/UtlIO.cpp
            include/Channel.h
            src/deconvolution/BaseDeconvolutionAlgorithm.cpp
            include/GaussianPSFGenerator.h
            src/psf/algorithms/GaussianPSFGenerator.cpp
            src/psf/algorithms/GibsonLanniPSFGenerator.cpp
            include/GibsonLanniPSFGenerator.h
            src/psf/algorithms/BornWolfModel.cpp
            include/BornWolfModel.h
            include/RLTVDeconvolutionAlgorithm.h
            src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
            include/RLADDeconvolutionAlgorithm.h
            src/psf/configs/GibsonLanniPSFConfig.cpp
            src/psf/configs/GaussianPSFConfig.cpp
            src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
            include/PSFConfig.h
            
    )
    # Set macro for GPU executable, for compiling correct libs in source code
    target_compile_definitions(dolphincuda PRIVATE CUDA_AVAILABLE)
    # Link CPU executable with the required libraries (cuFFTW)
    target_link_libraries(dolphincuda
            -lpthread
            -lgomp
            ${OpenCV_LIBS}
            stdc++fs
            ${TIFF_LIBRARIES}
            OpenMP::OpenMP_CXX
            ${CMAKE_SOURCE_DIR}/lib/cube/build/libCUBE.a
            cuda
            CUDA::cufft
            CUDA::cudart
            CUDA::cufftw
    )
endif()

