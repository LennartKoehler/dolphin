# Deconvolution Strategies Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/deconvolution/deconvolutionStrategies)

# ImageSplit Test
add_executable(imageSplit_test
    imageSplit_test.cpp
)
target_link_libraries(imageSplit_test
    PRIVATE
        dolphin
)
add_test(NAME ImageSplitTest COMMAND $<TARGET_FILE:imageSplit_test>)

# GetLabelGroups Test
add_executable(getLabelGroups_test
    getLabelGroups_test.cpp
)
target_link_libraries(getLabelGroups_test
    PRIVATE
        dolphin
)
target_include_directories(getLabelGroups_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME GetLabelGroupsTest COMMAND $<TARGET_FILE:getLabelGroups_test>)
set_tests_properties(GetLabelGroupsTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
