# Integration Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration)

set(CMAKE_BUILD_TYPE Release)

# Cube Shape Test
add_executable(cube_shape_test
    cubeShapeTest.cpp
)
target_link_libraries(cube_shape_test
    PRIVATE
        dolphin
        project_options
)
add_test(NAME CubeShapeTest COMMAND $<TARGET_FILE:cube_shape_test>)
set_tests_properties(CubeShapeTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Main Integration Test
add_executable(main_test
    main_test.cpp
)
target_link_libraries(main_test
    PRIVATE
        dolphin
)
target_include_directories(main_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME MainTest COMMAND $<TARGET_FILE:main_test>)
