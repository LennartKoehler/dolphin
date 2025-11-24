# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)


enable_testing()

# Include backends subdirectory for backend-specific tests
add_subdirectory(backends)

# Original combined test (optional)
add_executable(dolphin_tests
    main.cpp
    memory_exception_example.cpp
)

# Link libraries to tests
target_link_libraries(dolphin_tests
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs
)

# Include directories for tests
target_include_directories(dolphin_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add original combined test to test suite
add_test(NAME DolphinTests COMMAND dolphin_tests)
set_tests_properties(DolphinTests PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Standalone cuFFT vs FFTW comparison test
find_package(CUDAToolkit REQUIRED)

# Find FFTW using pkg-config or manual search
find_package(PkgConfig)
if (PkgConfig_FOUND)
    pkg_check_modules(FFTW fftw3)
endif()

# If pkg-config didn't find FFTW, try manual search
if (NOT FFTW_FOUND)
    find_path(FFTW_INCLUDE_DIR
        NAMES fftw3.h
        PATHS /usr/include /usr/local/include /opt/local/include
    )
    find_library(FFTW_LIBRARY
        NAMES fftw3
        PATHS /usr/lib /usr/local/lib /opt/local/lib
    )
    find_library(FFTW_THREADS_LIBRARY
        NAMES fftw3_threads
        PATHS /usr/lib /usr/local/lib /opt/local/lib
    )
    if (FFTW_INCLUDE_DIR AND FFTW_LIBRARY)
        set(FFTW_FOUND TRUE)
    endif()
endif()

if (NOT FFTW_FOUND)
    message(FATAL_ERROR "FFTW library not found. Please install libfftw3-dev")
endif()

add_executable(cufft_vs_fftw_comparison
    cufft_vs_fftw_comparison.cpp
)

# Link libraries for standalone test
target_link_libraries(cufft_vs_fftw_comparison
    PRIVATE
        CUDA::cufft
        CUDA::cufftw
        ${FFTW_LIBRARIES}
        ${FFTW_THREADS_LIBRARY}
        pthread
        CUDA::cudart
        ${CMAKE_THREAD_LIBS_INIT}
)

# Include directories for standalone test
target_include_directories(cufft_vs_fftw_comparison
    PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
        ${FFTW_INCLUDE_DIRS}
)

# Add standalone test to test suite
add_test(NAME CuFFTvsFFTW COMMAND cufft_vs_fftw_comparison)
set_tests_properties(CuFFTvsFFTW PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})


