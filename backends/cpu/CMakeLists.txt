# backends/cpu/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(CPUBackend)

# CPU backend is always a shared library (plugin)
add_library(cpu_backend SHARED CPUBackend.cpp)

# Include the main project headers
target_include_directories(cpu_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
)

# Find FFTW shared libraries
# Prefer system-installed shared libraries (.so) over static (.a)
# Prefer system shared libraries
find_library(FFTW3_LIB NAMES fftw3 PATHS /usr/lib /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(FFTW3_THREADS_LIB NAMES fftw3_threads PATHS /usr/lib /usr/lib/x86_64-linux-gnu REQUIRED)

message(STATUS "Using FFTW3 library: ${FFTW3_LIB}")
message(STATUS "Using FFTW3 threads library: ${FFTW3_THREADS_LIB}")

if(NOT FFTW3_LIB OR NOT FFTW3_THREADS_LIB)
    message(FATAL_ERROR "Could not find shared FFTW3 libraries. Please install libfftw3-dev (or rebuild FFTW3 with --with-pic).")
endif()

# Link FFTW shared libraries and OpenMP
target_link_libraries(cpu_backend PRIVATE
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    OpenMP::OpenMP_CXX
)

# Set proper output name
set_target_properties(cpu_backend PROPERTIES OUTPUT_NAME "cpu_backend")

# Optionally, specify library output directory
set_target_properties(cpu_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/cpu
)
