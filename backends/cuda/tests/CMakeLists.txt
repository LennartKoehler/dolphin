cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_BUILD_TYPE Debug)
set(GLIBCXX_USE_CXX11_ABI 0)
enable_testing()



add_executable(cuda_test
    cuda_test.cpp
)

add_executable(multiple_devices_test
    multiple_devices_test.cpp)


target_link_libraries(multiple_devices_test
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
set_target_properties(multiple_devices_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
add_test(NAME CUDABackendTest COMMAND cuda_test)

add_test(NAME MultipleDevices COMMAND multiple_devices_test)

target_compile_options(cuda_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(cuda_test PRIVATE "-static-libstdc++" "-static-libgcc")


target_compile_options(multiple_devices_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(multiple_devices_test PRIVATE "-static-libstdc++" "-static-libgcc")

