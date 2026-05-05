# Backend Tests CMakeLists.txt
# Tests that depend on dolphin (full application)
# FFTW-only tests live in backends/cpu/tests/
# cuFFT-only tests live in backends/cuda/tests/
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/backends)

# FFT Performance Test (uses dolphin BackendFactory)
add_executable(fft_performance_test
    cpu/fft_performance_test.cpp
)

target_link_libraries(fft_performance_test
    PRIVATE
        dolphin
)

target_include_directories(fft_performance_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

add_test(NAME FFTPerformanceTest COMMAND $<TARGET_FILE:fft_performance_test>)
set_tests_properties(FFTPerformanceTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# FFT Backend Test (uses dolphin BackendFactory + TiffReader)
add_executable(fft_backend_test
    fft_backend_test.cpp
)
target_link_libraries(fft_backend_test
    PRIVATE
        dolphin
)
target_include_directories(fft_backend_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME FFTBackendTest COMMAND $<TARGET_FILE:fft_backend_test>)
set_tests_properties(FFTBackendTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# CPU Data Reinterpret Test (uses dolphin BackendFactory + TiffReader)
add_executable(cpu_data_reinterpret_test
    cpu/cpu_data_reinterpret_test.cpp
)
target_link_libraries(cpu_data_reinterpret_test
    PRIVATE
        dolphin
)
target_include_directories(cpu_data_reinterpret_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME CPUDataReinterpret COMMAND $<TARGET_FILE:cpu_data_reinterpret_test>)
set_tests_properties(CPUDataReinterpret PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFT Real vs Complex Test (uses dolphin BackendFactory + TiffReader)
add_executable(fft_real_vs_complex_test
    cpu/fft_real_vs_complex_test.cpp
)
target_link_libraries(fft_real_vs_complex_test
    PRIVATE
        dolphin
)
target_include_directories(fft_real_vs_complex_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME FFTComplexVsReal COMMAND $<TARGET_FILE:fft_real_vs_complex_test>)
set_tests_properties(FFTComplexVsReal PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# DolphinBackend Memory Exception Test (uses only dolphinbackend)
add_executable(memory_exception_test
    dolphinbackend/memory_exception_example.cpp
)
target_link_libraries(memory_exception_test
    PRIVATE
        dolphinbackend
)
target_include_directories(memory_exception_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
)
add_test(NAME MemoryExceptionTest COMMAND $<TARGET_FILE:memory_exception_test>)
set_tests_properties(MemoryExceptionTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFTW In-Place vs Out-of-Place R2C Test (uses dolphin BackendFactory)
add_executable(fftw_inplace_vs_outofplace_r2c_test
    cpu/fftw_inplace_vs_outofplace_r2c_test.cpp
)
target_link_libraries(fftw_inplace_vs_outofplace_r2c_test
    PRIVATE
        dolphin
)
target_include_directories(fftw_inplace_vs_outofplace_r2c_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/cpu/include
        ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
)
add_test(NAME FFTWInplaceVsOutofplaceR2C COMMAND $<TARGET_FILE:fftw_inplace_vs_outofplace_r2c_test>)
set_tests_properties(FFTWInplaceVsOutofplaceR2C PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
