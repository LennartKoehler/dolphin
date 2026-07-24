set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/deconvolution)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(feathering_test
    SOURCES feathering_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(padding_test
    SOURCES padding_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

add_subdirectory(deconvolutionStrategies)
