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