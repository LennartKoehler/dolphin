# CLI Frontend CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DolphinCLI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
        ${CMAKE_SOURCE_DIR}/include
)
# Link against the Dolphin library and required dependencies
target_link_libraries(dolphin_cli
    PRIVATE
        dolphin::dolphin
)

