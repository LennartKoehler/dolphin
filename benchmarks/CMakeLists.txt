cmake_minimum_required(VERSION 3.14)

option(ENABLE_BENCHMARKS "Build benchmark executables" ON)
if(NOT ENABLE_BENCHMARKS)
    return()
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/cpu)

find_package(OpenMP REQUIRED)

find_library(LIKWID_LIBRARY
    NAMES likwid
    PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
)
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

function(add_cpu_benchmark name source)
    add_executable(${name} ${source})
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
    target_compile_options(${name} PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(${name} PRIVATE
        cpu_backend
        OpenMP::OpenMP_CXX
        pthread
    )
    target_include_directories(${name} PRIVATE ${FFTW_INCLUDE_DIR})
    if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
        target_compile_options(${name} PRIVATE -DLIKWID_PERFMON)
        target_link_libraries(${name} PRIVATE ${LIKWID_LIBRARY})
        target_include_directories(${name} PRIVATE ${LIKWID_INCLUDE_DIR})
    endif()
endfunction()

add_cpu_benchmark(fftw_multithreading_benchmark  cpu/fftw_multithreading_test.cpp)
add_cpu_benchmark(fftw_size_decomposition_benchmark cpu/fftw_size_decomposition_test.cpp)
add_cpu_benchmark(real_complex_fftw_benchmark    cpu/real_complex_fftw.cpp)

if(CUDAToolkit_FOUND)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks/cuda)

    find_path(FFTW_INCLUDE_DIR_BENCH
        NAMES fftw3.h
        PATHS /usr/include /usr/local/include /opt/local/include
    )
    find_library(FFTW_LIBRARY_BENCH
        NAMES fftw3f
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )
    find_library(FFTW_THREADS_LIBRARY_BENCH
        NAMES fftw3f_threads
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )
    find_library(FFTW_OMP_LIBRARY_BENCH
        NAMES fftw3f_omp
        PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
    )

    if (FFTW_INCLUDE_DIR_BENCH AND FFTW_LIBRARY_BENCH AND FFTW_THREADS_LIBRARY_BENCH)
        add_executable(cufft_vs_fftw_benchmark
            cuda/cufft_vs_fftw_comparison.cpp
        )
        set_property(TARGET cufft_vs_fftw_benchmark PROPERTY CXX_STANDARD 17)
        target_compile_options(cufft_vs_fftw_benchmark PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(cufft_vs_fftw_benchmark PRIVATE
            CUDA::cufft
            CUDA::cufftw
            ${FFTW_LIBRARY_BENCH}
            ${FFTW_THREADS_LIBRARY_BENCH}
            OpenMP::OpenMP_CXX
            pthread
            CUDA::cudart
        )
        if(FFTW_OMP_LIBRARY_BENCH)
            target_link_libraries(cufft_vs_fftw_benchmark PRIVATE ${FFTW_OMP_LIBRARY_BENCH})
        endif()
        target_include_directories(cufft_vs_fftw_benchmark PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS}
            ${FFTW_INCLUDE_DIR_BENCH}
        )
    endif()
endif()
