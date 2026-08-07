# CLI Frontend CMakeLists.txt
# Can be used standalone (find_package(dolphin)) or as a subdirectory
# of the main dolphin project (BUILD_CLI=ON).

if(NOT DEFINED DOLPHIN_CLI_STANDALONE)
    if(DEFINED PROJECT_NAME AND PROJECT_NAME STREQUAL "dolphin")
        set(DOLPHIN_CLI_STANDALONE OFF)
    else()
        set(DOLPHIN_CLI_STANDALONE ON)
    endif()
endif()

if(DOLPHIN_CLI_STANDALONE)
    cmake_minimum_required(VERSION 3.10)
    project(DolphinCLI)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    set(CMAKE_EXPORT_COMPILE_COMMANDS On)

    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    find_package(CUDAToolkit QUIET)
    find_package(dolphin REQUIRED)

    set(DOLPHIN_LINK_TARGET dolphin::dolphin)
    set(DOLPHIN_OPTIONS_TARGET dolphin::project_options)
    set(CLI11_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/CLI11/include")
    set(CLI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/cli/")
else()
    set(DOLPHIN_LINK_TARGET dolphin)
    set(DOLPHIN_OPTIONS_TARGET project_options)
    set(CLI11_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/CLI11/include")
    set(CLI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/cli/")
endif()

add_executable(dolphin_cli
    src/main.cpp
    src/CLIFrontend.cpp
)

target_include_directories(dolphin_cli
    PRIVATE
        ${CLI11_INCLUDE_DIR}
        ${CLI_INCLUDE_DIR}
)

target_link_libraries(dolphin_cli
    PRIVATE
        ${DOLPHIN_LINK_TARGET}
        ${DOLPHIN_OPTIONS_TARGET}
)

set_target_properties(dolphin_cli PROPERTIES OUTPUT_NAME dolphin)

if(TARGET cuda_backend)
    set_target_properties(dolphin_cli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# ---------------------------------------------------------------------------
# CLI Frontend tests
# ---------------------------------------------------------------------------
if(NOT DOLPHIN_CLI_STANDALONE AND ENABLE_TESTS)
    add_executable(test_cli_frontend
        tests/test_cli_frontend.cpp
        src/CLIFrontend.cpp
    )

    target_include_directories(test_cli_frontend
        PRIVATE
            ${CLI11_INCLUDE_DIR}
            ${CLI_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/tests/utils
    )

    target_link_libraries(test_cli_frontend
        PRIVATE
            ${DOLPHIN_LINK_TARGET}
            ${DOLPHIN_OPTIONS_TARGET}
            GTest::gtest_main
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_cli_frontend PRIVATE -fno-fast-math)
    endif()

    if(TARGET cuda_backend)
        set_target_properties(test_cli_frontend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()

    gtest_discover_tests(test_cli_frontend DISCOVERY_TIMEOUT 30)
endif()
