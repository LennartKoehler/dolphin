cmake_minimum_required(VERSION 3.8)
project(CUBE LANGUAGES CXX CUDA)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3") # -rdc=true
set(CMAKE_CUDA_ARCHITECTURES "75;80;90")
set(CMAKE_BUILD_TYPE Release)  # or Debug

# Add include directories
include_directories(include)

find_package(OpenMP REQUIRED)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
# Define source files
set(CPP_SOURCES
        src/main.cpp
        src/utl.cu
        src/operations.cu
)

set(CUDA_SOURCES
        src/kernels/device.cu
        src/kernels/global.cu
)

# Create the executable
add_library(CUBE SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
#add_definitions(-DENABLE_CUBE_DEBUG)
#add_definitions(-DENABLE_CUBEUTL_DEBUG)

target_link_directories(CUBE PUBLIC include)

# Link your executable with the required libraries
target_link_libraries(CUBE
        OpenMP::OpenMP_CXX
        pthread
        gomp
        ${CUDA_CUDA_LIBRARY}
        CUDA::cufft
        CUDA::cufftw
        ${OpenCV_LIBS}
)

# Ensure that CUDA and required libraries are linked properly
target_compile_features(CUBE PRIVATE cxx_std_11)
