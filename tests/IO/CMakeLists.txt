# IO Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/IO)

# Read/Write Test
add_executable(read_write_test
    read_write_test.cpp
)
target_link_libraries(read_write_test
    PRIVATE
        dolphin
)
target_include_directories(read_write_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME ReadWriteTest COMMAND $<TARGET_FILE:read_write_test>)

# TIFF Generation Test
add_executable(tiff_generation_test
    tiff_generation_test.cpp
)
target_link_libraries(tiff_generation_test
    PRIVATE
        dolphin
)
target_include_directories(tiff_generation_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME TiffGenerationTest COMMAND $<TARGET_FILE:tiff_generation_test>)
set_tests_properties(TiffGenerationTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Mirror Padding Test
add_executable(mirror_padding_test
    mirror_padding_test.cpp
)
target_link_libraries(mirror_padding_test
    PRIVATE
        dolphin
)
target_include_directories(mirror_padding_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME MirrorPaddingTest COMMAND $<TARGET_FILE:mirror_padding_test>)
