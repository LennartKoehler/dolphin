cmake_minimum_required(VERSION 3.14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/unit)

set(UNIT_TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/backends/cpu/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/lib/nlohmann/json/single_include
    ${CMAKE_SOURCE_DIR}/lib/spdlog/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(test_cuboid_shape
    SOURCES test_cuboid_shape.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_image3d
    SOURCES test_image3d.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_range_map
    SOURCES test_range_map.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_boxcoord
    SOURCES test_boxcoord.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_config
    SOURCES test_config.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_algorithm_factory
    SOURCES test_algorithm_factory.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_algorithms
    SOURCES test_algorithms.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_backend_factory
    SOURCES test_backend_factory.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_psf_generator
    SOURCES test_psf_generator.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_threadpool
    SOURCES test_threadpool.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_image_padding
    SOURCES test_image_padding.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_image_operations
    SOURCES test_image_operations.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)

dolphin_add_test(test_preprocessor
    SOURCES test_preprocessor.cpp
    LIBS dolphin
    INCLUDES ${UNIT_TEST_INCLUDES}
)
