set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/backends/cpu/tests)

add_executable(cpu_backend_test
    test_cpu_backend.cpp
)

target_link_libraries(cpu_backend_test
    PRIVATE
        cpu_backend
        dolphinbackend
        GTest::gtest_main
        project_options
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cpu_backend_test PRIVATE -fno-fast-math)
endif()

gtest_discover_tests(cpu_backend_test DISCOVERY_TIMEOUT 30)

set_target_properties(cpu_backend_test PROPERTIES
    BUILD_RPATH "\$ORIGIN/../lib"
    INSTALL_RPATH "\$ORIGIN/../lib"
)
