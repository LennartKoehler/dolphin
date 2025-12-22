# backends/openmp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(OpenMPBackend)

# OpenMP backend is always a shared library (plugin)
add_library(openmp_backend SHARED OpenMPBackend.cpp)

# Include the main project headers
target_include_directories(openmp_backend PUBLIC
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
target_link_libraries(openmp_backend PRIVATE
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    OpenMP::OpenMP_CXX
    backendlib
)

# Set proper output name
set_target_properties(openmp_backend PROPERTIES OUTPUT_NAME "openmp_backend")

# Specify library output directory
set_target_properties(openmp_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/openmp
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/backends/openmp
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/backends/openmp
)