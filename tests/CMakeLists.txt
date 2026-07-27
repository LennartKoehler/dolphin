set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

function(dolphin_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE GTest::gtest_main project_options ${ARG_LIBS})
    if(ARG_INCLUDES)
        target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${name} PRIVATE -fno-fast-math)
    endif()
    target_compile_definitions(${name} PRIVATE
        CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    )
    gtest_discover_tests(${name} DISCOVERY_TIMEOUT 30)
endfunction()

add_subdirectory(backends)
add_subdirectory(IO)
add_subdirectory(deconvolution)
add_subdirectory(image)
add_subdirectory(integration)
add_subdirectory(unit)

# ---------------------------------------------------------------------------
# Combined test binary — runs every test in a single executable
# ---------------------------------------------------------------------------
set(ALL_TEST_SOURCES
    unit/test_cuboid_shape.cpp
    unit/test_image3d.cpp
    unit/test_range_map.cpp
    unit/test_boxcoord.cpp
    unit/test_config.cpp
    unit/test_algorithm_factory.cpp
    unit/test_algorithms.cpp
    unit/test_backend_factory.cpp
    unit/test_psf_generator.cpp
    unit/test_threadpool.cpp
    unit/test_image_padding.cpp
    unit/test_image_operations.cpp
    unit/test_preprocessor.cpp
    backends/fft_backend_test.cpp
    backends/dolphinbackend/memory_exception_example.cpp
    backends/independent_fft_test.cpp
    integration/cubeShapeTest.cpp
    integration/main_integration_test.cpp
    image/image_comparison_test.cpp
    deconvolution/feathering_test.cpp
    deconvolution/padding_test.cpp
    deconvolution/deconvolutionStrategies/imageSplit_test.cpp
    deconvolution/deconvolutionStrategies/getLabelGroups_test.cpp
    deconvolution/deconvolutionStrategies/memory_test.cpp
    IO/read_write_test.cpp
    IO/tiff_generation_test.cpp
    IO/mirror_padding_test.cpp
    IO/tiff_reader_test.cpp
    ${CMAKE_SOURCE_DIR}/backends/cpu/tests/test_cpu_backend.cpp
)

set(ALL_TEST_LIBS
    GTest::gtest_main
    dolphin
    dolphinbackend
    cpu_backend
    project_options
)

set(ALL_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/backends/cpu/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/lib/nlohmann/json/single_include
    ${CMAKE_SOURCE_DIR}/lib/spdlog/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

if(TARGET cuda_backend)
    list(APPEND ALL_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/backends/cuda/tests/test_cuda_backend.cpp
    )
    list(APPEND ALL_TEST_LIBS cuda_backend)
    list(APPEND ALL_TEST_INCLUDES ${CMAKE_SOURCE_DIR}/backends/cuda/include)
endif()

add_executable(dolphin_all_tests ${ALL_TEST_SOURCES})
target_link_libraries(dolphin_all_tests PRIVATE ${ALL_TEST_LIBS})
target_include_directories(dolphin_all_tests PRIVATE ${ALL_TEST_INCLUDES})
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dolphin_all_tests PRIVATE -fno-fast-math)
endif()
target_compile_definitions(dolphin_all_tests PRIVATE
    CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
)
