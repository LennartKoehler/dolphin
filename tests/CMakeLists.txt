# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Enable testing
enable_testing()

# Include backends subdirectory for backend-specific tests
add_subdirectory(backends)

# Original combined test (optional)
add_executable(dolphin_tests
    main.cpp
    memory_exception_example.cpp
)

# Link libraries to tests
target_link_libraries(dolphin_tests
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs
)

# Include directories for tests
target_include_directories(dolphin_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add original combined test to test suite
add_test(NAME DolphinTests COMMAND dolphin_tests)
set_tests_properties(DolphinTests PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})