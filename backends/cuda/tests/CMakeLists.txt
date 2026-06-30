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

set_target_properties(cuda_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
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
