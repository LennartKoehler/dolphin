# CLI Frontend CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(DolphinCLI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_EXPORT_COMPILE_COMMANDS On)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Match dolphin's dynamic CRT (/MD) to avoid LNK2038 mismatch
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

find_package(CUDAToolkit QUIET)
find_package(dolphin REQUIRED)
# Create CLI executable
add_executable(dolphin_cli
    src/main.cpp
    src/CLIFrontend.cpp

)
# Include directories
target_include_directories(dolphin_cli
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lib/CLI11/include
        ${CMAKE_SOURCE_DIR}/include/cli/
)
# Link against the Dolphin library and required dependencies
target_link_libraries(dolphin_cli
    PRIVATE
        dolphin::dolphin
        dolphin::project_options
)

if(TARGET CUDA::cufft_static)
    enable_language(CUDA)
    set_target_properties(dolphin_cli PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

