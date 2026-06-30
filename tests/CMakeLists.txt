cmake_minimum_required(VERSION 3.14)

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_rcs ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

function(dolphin_add_test name)
    cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;INCLUDES" ${ARGN})
    add_executable(${name} ${ARG_SOURCES})
    target_link_libraries(${name} PRIVATE GTest::gtest_main ${ARG_LIBS})
    if(ARG_INCLUDES)
        target_include_directories(${name} PRIVATE ${ARG_INCLUDES})
    endif()
    target_compile_definitions(${name} PRIVATE
        CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    )
    gtest_discover_tests(${name})
endfunction()

add_subdirectory(backends)
add_subdirectory(IO)
add_subdirectory(deconvolution)
add_subdirectory(image)
add_subdirectory(integration)
add_subdirectory(unit)
