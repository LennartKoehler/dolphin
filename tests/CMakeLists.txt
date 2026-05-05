# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

# Place test binaries in a directory structure mirroring the source tree
# e.g. tests/IO/read_write_test -> bin/tests/IO/read_write_test
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# Backend tests (dolphin-dependent backend tests)
add_subdirectory(backends)

# IO tests
add_subdirectory(IO)

# Deconvolution tests
add_subdirectory(deconvolution)

# Image tests
add_subdirectory(image)

# Integration tests
add_subdirectory(integration)
