cmake_minimum_required(VERSION 3.10)  # Minimale Version von CMake
project(Deconvtool)  # Name des Projekts

# Setze den C++-Standard, den das Projekt verwendet
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp -march=native")

# Bibliotheksverzeichnis für FFTW3, falls nicht standard
link_directories(/usr/lib/x86_64-linux-gnu)

# Finde das OpenCV-Paket
find_package(OpenCV REQUIRED)
# Findet die FFTW3-Bibliothek
# Manuell setzen der Include- und Bibliothekspfade
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
# Finde FFTW3
#find_package(FFTW3 REQUIRED)
# Finden Sie OpenMP
find_package(OpenMP REQUIRED)

# Füge die Include-Verzeichnisse hinzu
#include_directories(${FFTW3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Füge das Verzeichnis hinzu, das die OpenCV-Header enthält
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(TIFF REQUIRED)

include_directories(${TIFF_INCLUDE_DIR})

# Definiere die Quelldatei(en) für das ausführbare Programm
add_executable(deconvtool ${CMAKE_SOURCE_DIR}/src/main.cpp
        src/hyperstack/HyperstackImage.cpp
        include/HyperstackImage.h
        src/hyperstack/Image3D.cpp
        include/Image3D.h
        src/hyperstack/ImageMetaData.cpp
        include/ImageMetaData.h
        src/psf/PSF.cpp
        include/PSF.h
        include/BasePSFGeneratorAlgorithm.h
        src/psf/algorithms/SimpleGaussianPSFGeneratorAlgorithm.cpp
        include/SimpleGaussianPSFGeneratorAlgorithm.h
        include/PSFGenerator.h
        include/DeconvolutionAlgorithm.h
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
        include/GaussianPSFGeneratorAlgorithm.h
        src/psf/algorithms/GaussianPSFGeneratorAlgorithm.cpp
        src/psf/algorithms/BornWolfModel.cpp
        include/BornWolfModel.h
)
# Link your executable with the required libraries
target_link_libraries(deconvtool
        fftw3_threads
        -lpthread
        -lgomp
)

# Verknüpfe das Programm mit den OpenCV-Bibliotheken
target_link_libraries(deconvtool ${OpenCV_LIBS})
# Linkt FFTW3 zur Zielanwendung
target_link_libraries(deconvtool fftw3)
target_link_libraries(deconvtool stdc++fs)
target_link_libraries(deconvtool ${TIFF_LIBRARIES})
# Verlinken Sie OpenMP-Bibliotheken
target_link_libraries(deconvtool OpenMP::OpenMP_CXX)

