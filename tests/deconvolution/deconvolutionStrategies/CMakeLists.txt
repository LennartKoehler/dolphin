set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/deconvolution/deconvolutionStrategies)

set(TEST_INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
    ${CMAKE_SOURCE_DIR}/dolphin_image/include
    ${CMAKE_SOURCE_DIR}/tests/utils
)

dolphin_add_test(memory_test
    SOURCES memory_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

dolphin_add_test(imageSplit_test
    SOURCES imageSplit_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)

# Exhaustive geometric-contract tests (thousands of splits).
# Labeled "exhaustive" so regular CI can skip them:
#   ctest -LE exhaustive                # skip exhaustive
#   ctest -L exhaustive --output-on-failure  # run only exhaustive
dolphin_add_test(imageSplit_contract_test
    SOURCES imageSplit_contract_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
    LABELS exhaustive
)

dolphin_add_test(getLabelGroups_test
    SOURCES getLabelGroups_test.cpp
    LIBS dolphin
    INCLUDES ${TEST_INCLUDES}
)
