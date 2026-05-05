# Image Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/image)

# Image Comparison Test
add_executable(image_comparison_test
    image_comparison_test.cpp
)
target_link_libraries(image_comparison_test
    PRIVATE
        dolphin
)
target_include_directories(image_comparison_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME ImageComparisonTest COMMAND $<TARGET_FILE:image_comparison_test>)
set_tests_properties(ImageComparisonTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
