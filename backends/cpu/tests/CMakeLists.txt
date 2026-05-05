cmake_minimum_required(VERSION 3.10)

set(CMAKE_cpu_RUNTIME_LIBRARY Static)
set(CMAKE_BUILD_TYPE Debug)
set(GLIBCXX_USE_CXX11_ABI 0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/cpu/tests)

enable_testing()



add_executable(cpu_test
    test.cpp
)

target_link_libraries(cpu_test
    PRIVATE
        cpu_backend
)

# RPATH so it finds bundled libs
set_target_properties(cpu_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
add_test(NAME cpuBackendTest COMMAND $<TARGET_FILE:cpu_test>)

target_compile_options(cpu_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(cpu_test PRIVATE "-static-libstdc++" "-static-libgcc")

# ---------------------------------------------------------------------------
# Multiple backends / multi-device tests (each in its own source file)
# ---------------------------------------------------------------------------

function(add_cpu_test test_name test_source)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE cpu_backend)
    set_target_properties(${test_name} PROPERTIES
        BUILD_RPATH "\$ORIGIN/../lib"
        INSTALL_RPATH "\$ORIGIN/../lib"
    )
    target_compile_options(${test_name} PRIVATE "-static-libstdc++" "-static-libgcc")
    target_link_options(${test_name} PRIVATE "-static-libstdc++" "-static-libgcc")
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endfunction()

add_cpu_test(cpu_test_multiple_backends      test_multiple_backends.cpp)
add_cpu_test(cpu_test_independent_fft          test_independent_fft.cpp)
add_cpu_test(cpu_test_clone_shared_memory      test_clone_shared_memory.cpp)
add_cpu_test(cpu_test_different_thread_configs test_different_thread_configs.cpp)
add_cpu_test(cpu_test_concurrent_backends     test_concurrent_backends.cpp)
add_cpu_test(cpu_test_shared_memory_tracking  test_shared_memory_tracking.cpp)
add_cpu_test(cpu_test_move_data               test_move_data_between_backends.cpp)

# ---------------------------------------------------------------------------
# FFTW-only tests (no dolphin dependency)
# ---------------------------------------------------------------------------

find_package(OpenMP REQUIRED)

# LIKWID support (optional)
find_library(LIKWID_LIBRARY
    NAMES likwid
    PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
)
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

# FFTW Multithreading Performance Test
add_executable(fftw_multithreading_test
    fftw_multithreading_test.cpp
)

set_property(TARGET fftw_multithreading_test PROPERTY CXX_STANDARD 17)
target_compile_options(fftw_multithreading_test PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(fftw_multithreading_test
    PRIVATE
        cpu_backend
        OpenMP::OpenMP_CXX
        pthread
)

target_include_directories(fftw_multithreading_test
    PRIVATE
        ${FFTW_INCLUDE_DIR}
)

# Add LIKWID support if available
if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    target_compile_options(fftw_multithreading_test PRIVATE -DLIKWID_PERFMON)
    target_link_libraries(fftw_multithreading_test PRIVATE ${LIKWID_LIBRARY})
    target_include_directories(fftw_multithreading_test PRIVATE ${LIKWID_INCLUDE_DIR})
    message(STATUS "LIKWID support enabled for fftw_multithreading_test")
else()
    message(STATUS "LIKWID not found - building fftw_multithreading_test without LIKWID support")
endif()

add_test(NAME FFTWMultithreadingTest COMMAND $<TARGET_FILE:fftw_multithreading_test>)
set_tests_properties(FFTWMultithreadingTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFTW Size Decomposition Test
add_executable(fftw_size_decomposition_test
    fftw_size_decomposition_test.cpp
)

target_link_libraries(fftw_size_decomposition_test
    PRIVATE
    project_options)

set_property(TARGET fftw_size_decomposition_test PROPERTY CXX_STANDARD 17)
target_compile_options(fftw_size_decomposition_test PRIVATE ${OpenMP_CXX_FLAGS})

target_link_libraries(fftw_size_decomposition_test
    PRIVATE
        cpu_backend
        OpenMP::OpenMP_CXX
        pthread
)

if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    target_compile_options(fftw_size_decomposition_test PRIVATE -DLIKWID_PERFMON)
    target_link_libraries(fftw_size_decomposition_test PRIVATE ${LIKWID_LIBRARY})
    target_include_directories(fftw_size_decomposition_test PRIVATE ${LIKWID_INCLUDE_DIR})
    message(STATUS "LIKWID support enabled for fftw_size_decomposition_test")
else()
    message(STATUS "LIKWID not found - building fftw_size_decomposition_test without LIKWID support")
endif()

add_test(NAME FFTWSizeDecompositionTest COMMAND $<TARGET_FILE:fftw_size_decomposition_test>)
set_tests_properties(FFTWSizeDecompositionTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFTW In-Place vs Out-of-Place R2C Test (uses dolphin BackendFactory - in tests/backends/cpu/)
# add_executable(fftw_inplace_vs_outofplace_r2c_test ...)
# This test is built from tests/backends/ since it depends on dolphin

# Real vs Complex FFTW Test
add_executable(real_complex_fftw
    real_complex_fftw.cpp
)

target_link_libraries(real_complex_fftw
    PRIVATE
        cpu_backend
        OpenMP::OpenMP_CXX
        pthread
)

if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    target_compile_options(real_complex_fftw PRIVATE -DLIKWID_PERFMON)
    target_link_libraries(real_complex_fftw PRIVATE ${LIKWID_LIBRARY})
    target_include_directories(real_complex_fftw PRIVATE ${LIKWID_INCLUDE_DIR})
endif()

add_test(NAME RealComplexFftw COMMAND $<TARGET_FILE:real_complex_fftw>)
set_tests_properties(RealComplexFftw PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "FFTW multithreading test will be built")
message(STATUS "FFTW size decomposition test will be built")
message(STATUS "FFTW library: ${FFTW_LIBRARY}")
message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    message(STATUS "LIKWID library: ${LIKWID_LIBRARY}")
    message(STATUS "LIKWID include dir: ${LIKWID_INCLUDE_DIR}")
else()
    message(STATUS "LIKWID library not found - tests will be built without LIKWID support")
endif()
