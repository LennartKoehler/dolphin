set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/IO)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(read_write_test
    SOURCES read_write_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(tiff_generation_test
    SOURCES tiff_generation_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(mirror_padding_test
    SOURCES mirror_padding_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(tiff_reader_test
    SOURCES tiff_reader_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)
