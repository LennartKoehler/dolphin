# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)


set(CMAKE_BUILD_TYPE Release)

enable_testing()

add_executable(cube_shape_test
    cubeShapeTest.cpp)

target_link_libraries(cube_shape_test
    PRIVATE
        dolphin)

target_link_libraries(cube_shape_test
    PRIVATE
        project_options)


add_test(NAME CubeShapeTest COMMAND cube_shape_test)
set_tests_properties(CubeShapeTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

