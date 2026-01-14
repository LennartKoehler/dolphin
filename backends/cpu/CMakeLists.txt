# backends/cpu/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(CPUBackend)


find_library(FFTW3_LIB NAMES fftw3 PATHS /usr/lib /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(FFTW3_THREADS_LIB NAMES fftw3_threads PATHS /usr/lib /usr/lib/x86_64-linux-gnu REQUIRED)


add_library(cpu_backend SHARED CPUBackend.cpp)

target_compile_options(cpu_backend PRIVATE
    -O3
    -march=native
    -fopt-info-vec-optimized
    -g -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma
)
# Include the main project headers
target_include_directories(cpu_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)
# Link FFTW shared libraries and OpenMP
target_link_libraries(cpu_backend PRIVATE
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    backendlib
)

# Specify library output directory
set_target_properties(cpu_backend PROPERTIES
    OUTPUT_NAME "cpu_backend"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)
