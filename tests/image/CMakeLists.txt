set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/image)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(image_comparison_test
    SOURCES image_comparison_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)
