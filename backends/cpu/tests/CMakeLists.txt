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

set_target_properties(cpu_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
add_test(NAME cpuBackendTest COMMAND $<TARGET_FILE:cpu_test>)

target_compile_options(cpu_test PRIVATE "-static-libstdc++" "-static-libgcc")
target_link_options(cpu_test PRIVATE "-static-libstdc++" "-static-libgcc")

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
add_cpu_test(cpu_test_clone_shared_memory      test_clone_shared_memory.cpp)
add_cpu_test(cpu_test_different_thread_configs test_different_thread_configs.cpp)
add_cpu_test(cpu_test_concurrent_backends     test_concurrent_backends.cpp)
add_cpu_test(cpu_test_shared_memory_tracking  test_shared_memory_tracking.cpp)
add_cpu_test(cpu_test_move_data               test_move_data_between_backends.cpp)
