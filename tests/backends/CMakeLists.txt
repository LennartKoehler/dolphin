set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/backends)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(fft_backend_test
    SOURCES fft_backend_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(memory_exception_test
    SOURCES dolphinbackend/memory_exception_example.cpp
    LIBS dolphinbackend
    INCLUDES ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(independent_fft_test
    SOURCES independent_fft_test.cpp
    LIBS cpu_backend dolphinbackend
    INCLUDES ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include ${CMAKE_SOURCE_DIR}/backends/cpu/include
)
