cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_BUILD_TYPE Debug)
set(GLIBCXX_USE_CXX11_ABI 0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/cuda/tests)

enable_testing()



add_executable(cuda_test
    cuda_test.cpp
)

add_executable(cuda_multiple_devices_test
    multiple_devices_test.cpp
)


target_link_libraries(cuda_multiple_devices_test
    PRIVATE
        cuda_backend)
target_link_libraries(cuda_test
    PRIVATE
        cuda_backend
)

# RPATH so it finds bundled libs
set_target_properties(cuda_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
# RPATH so it finds bundled libs
set_target_properties(cuda_multiple_devices_test PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
add_test(NAME CUDABackendTest COMMAND $<TARGET_FILE:cuda_test>)

add_test(NAME CUDAMultipleDevices COMMAND $<TARGET_FILE:cuda_multiple_devices_test>)

target_compile_options(cuda_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(cuda_test PRIVATE "-static-libstdc++" "-static-libgcc")


target_compile_options(cuda_multiple_devices_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(cuda_multiple_devices_test PRIVATE "-static-libstdc++" "-static-libgcc")

# ---------------------------------------------------------------------------
# cuFFT vs FFTW comparison test
# ---------------------------------------------------------------------------

if(CUDAToolkit_FOUND)
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
            cufft_vs_fftw_comparison.cpp
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
        add_test(NAME CuFFTvsFFTW COMMAND $<TARGET_FILE:cufft_vs_fftw_comparison>)
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
