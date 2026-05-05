# Deconvolution Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/deconvolution)

# Feathering Test
add_executable(feathering_test
    feathering_test.cpp
)
target_link_libraries(feathering_test
    PRIVATE
        dolphin
)
target_include_directories(feathering_test
    PRIVATE
        BEFORE ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME FeatheringTest COMMAND $<TARGET_FILE:feathering_test>)

# Padding Test
add_executable(padding_test
    padding_test.cpp
)
target_link_libraries(padding_test
    PRIVATE
        dolphin
)
target_include_directories(padding_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)
add_test(NAME PaddingTest COMMAND $<TARGET_FILE:padding_test>)
set_tests_properties(PaddingTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Deconvolution Strategies subdirectory
add_subdirectory(deconvolutionStrategies)
