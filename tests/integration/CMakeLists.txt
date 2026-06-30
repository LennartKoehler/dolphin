cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/integration)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(cube_shape_test
    SOURCES cubeShapeTest.cpp
    LIBS dolphin project_options
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(main_test
    SOURCES main_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)
