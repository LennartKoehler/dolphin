cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
project(CUBE LANGUAGES CXX CUDA)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "86;75;90")


option(DOUBLE_PRECISION "Enable double precision" OFF)

find_package(CUDAToolkit 12.8 REQUIRED)
# Define source files
set(CPP_SOURCES
        src/utl.cu
        src/operations.cu
)

set(CUDA_SOURCES
        src/kernels/global.cu
)

# Create the executable
add_library(CUBE STATIC ${CPP_SOURCES} ${CUDA_SOURCES})


target_link_libraries(CUBE PRIVATE project_options)

if(DOUBLE_PRECISION)
    target_compile_definitions(CUBE PRIVATE DOUBLE_PRECISION)
endif()

target_include_directories(CUBE
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link your executable with the required libraries
target_link_libraries(CUBE
   PUBLIC
        ${CUDA_CUDA_LIBRARY}
        CUDA::cudart_static
)

# Ensure that CUDA and required libraries are linked properly
target_compile_features(CUBE PRIVATE cxx_std_11)
set_target_properties(CUBE PROPERTIES
    INTERFACE_BUILD_RPATH "\$ORIGIN/../lib"
    INTERFACE_INSTALL_RPATH "\$ORIGIN/../lib"
)
