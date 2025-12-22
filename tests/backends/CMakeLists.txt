# Backend Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Enable testing
enable_testing()

# CPU Backend Test
add_executable(cpu_test
    cpu_test.cpp
)

# Link libraries to CPU test
target_link_libraries(cpu_test
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs
)

# Include directories for CPU test
target_include_directories(cpu_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add CPU test to test suite
add_test(NAME CPUBackendTest COMMAND cpu_test)
set_tests_properties(CPUBackendTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# OpenMP Backend Test
add_executable(openmp_test
    openmp_test.cpp
)

# Link libraries to OpenMP test
target_link_libraries(openmp_test
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs
)

# Include directories for OpenMP test
target_include_directories(openmp_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add OpenMP test to test suite
add_test(NAME OpenMPBackendTest COMMAND openmp_test)
set_tests_properties(OpenMPBackendTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# CUDA Backend Test (conditional)
option(BUILD_CUDA_TESTS "Build CUDA backend tests" ON)
if(BUILD_CUDA_TESTS)
    # Check if CUDA is available
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        add_executable(cuda_test
            cuda_test.cpp
        )
        
        # Link libraries to CUDA test
        target_link_libraries(cuda_test
            PRIVATE
                dolphinlib
                backendlib
                pthread
                gomp
                stdc++fs
        )
        
        # Include directories for CUDA test
        target_include_directories(cuda_test
            PRIVATE
                ${CMAKE_SOURCE_DIR}/../include
                ${CMAKE_SOURCE_DIR}/../src
        )
        
        # Add CUDA compilation flags if needed
        target_compile_definitions(cuda_test PRIVATE CUDA_AVAILABLE)
        
        # Add CUDA test to test suite
        add_test(NAME CUDABackendTest COMMAND cuda_test)
        set_tests_properties(CUDABackendTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    else()
        message(STATUS "CUDA not found - CUDA backend test will not be built")
    endif()
endif()

# FFT Performance Test
add_executable(fft_performance_test
    fft_performance_test.cpp
)

# Link libraries to FFT performance test
target_link_libraries(fft_performance_test
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs
)

# Include directories for FFT performance test
target_include_directories(fft_performance_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add FFT performance test to test suite
add_test(NAME FFTPerformanceTest COMMAND fft_performance_test)
set_tests_properties(FFTPerformanceTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# cuFFT vs FFTW comparison test (conditional on CUDA availability)
if(BUILD_CUDA_TESTS AND CUDAToolkit_FOUND)
    find_package(OpenMP REQUIRED)
    
    # Find FFTW libraries manually since pkg-config might not work correctly
    find_path(FFTW_INCLUDE_DIR
        NAMES fftw3.h
        PATHS /usr/include /usr/local/include /opt/local/include
    )
    
    find_library(FFTW_LIBRARY
        NAMES fftw3f
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )
    
    find_library(FFTW_THREADS_LIBRARY
        NAMES fftw3f_threads
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )
    
    find_library(FFTW_OMP_LIBRARY
        NAMES fftw3f_omp
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )
    
    if (FFTW_INCLUDE_DIR AND FFTW_LIBRARY AND FFTW_THREADS_LIBRARY)
        add_executable(cufft_vs_fftw_comparison
            ../cufft_vs_fftw_comparison.cpp
        )
        
        # Set C++ standard and compiler flags
        set_property(TARGET cufft_vs_fftw_comparison PROPERTY CXX_STANDARD 17)
        target_compile_options(cufft_vs_fftw_comparison PRIVATE ${OpenMP_CXX_FLAGS})
        
        # Link libraries for comparison test
        target_link_libraries(cufft_vs_fftw_comparison
            PRIVATE
                CUDA::cufft
                CUDA::cufftw
                ${FFTW_LIBRARY}
                ${FFTW_THREADS_LIBRARY}
                OpenMP::OpenMP_CXX
                pthread
                CUDA::cudart
        )
        
        # Add FFTW_OMP_LIBRARY if it was found
        if(FFTW_OMP_LIBRARY)
            target_link_libraries(cufft_vs_fftw_comparison PRIVATE ${FFTW_OMP_LIBRARY})
        endif()
        
        # Include directories for comparison test
        target_include_directories(cufft_vs_fftw_comparison
            PRIVATE
                ${CUDAToolkit_INCLUDE_DIRS}
                ${FFTW_INCLUDE_DIR}
        )
        
        # Add comparison test to test suite
        add_test(NAME CuFFTvsFFTW COMMAND cufft_vs_fftw_comparison)
        set_tests_properties(CuFFTvsFFTW PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
        
        message(STATUS "cuFFT vs FFTW comparison test will be built")
        message(STATUS "FFTW library: ${FFTW_LIBRARY}")
        message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
        if(FFTW_OMP_LIBRARY)
            message(STATUS "FFTW OpenMP library: ${FFTW_OMP_LIBRARY}")
        endif()
    else()
        message(STATUS "FFTW libraries not found - cuFFT vs FFTW comparison test will not be built")
        message(STATUS "FFTW include dir: ${FFTW_INCLUDE_DIR}")
        message(STATUS "FFTW library: ${FFTW_LIBRARY}")
        message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
    endif()
else()
    message(STATUS "CUDA not available - cuFFT vs FFTW comparison test will not be built")
endif()