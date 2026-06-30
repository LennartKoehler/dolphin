cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/cuda/tests)

enable_testing()

add_executable(cuda_backend_test
    test_cuda_backend.cpp
)

target_link_libraries(cuda_backend_test
    PRIVATE
        cuda_backend
        GTest::gtest_main
)

target_compile_options(cuda_backend_test PRIVATE -fno-fast-math)

gtest_discover_tests(cuda_backend_test)

set_target_properties(cuda_backend_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
