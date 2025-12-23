# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)


enable_testing()

# Include backends subdirectory for backend-specific tests
add_subdirectory(backends)

# Original combined test (optional)
add_executable(dolphin_tests
    memory_exception_example.cpp
    labeledImage/tiff_generation_test.cpp
)

# Link libraries to tests
target_link_libraries(dolphin_tests
    PRIVATE
        dolphin 
)

# Include directories for tests
target_include_directories(dolphin_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add original combined test to test suite
add_test(NAME DolphinTests COMMAND dolphin_tests)
set_tests_properties(DolphinTests PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})



add_executable(feathering_test
    feathering_test.cpp)

target_link_libraries(feathering_test
    PRIVATE
        dolphin
        )

# Add ITK include directories and configuration
# target_include_directories(feathering_test
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/include
# )

add_test(NAME FeatheringTest COMMAND feathering_test)


# Standalone TIFF generation test
add_executable(tiff_generation_test
    labeledImage/tiff_generation_test.cpp
)

# Link libraries to TIFF generation test
target_link_libraries(tiff_generation_test
    PRIVATE
        dolphin
)

# Include directories for TIFF generation test
target_include_directories(tiff_generation_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

# Add TIFF generation test to test suite
add_test(NAME TiffGenerationTest COMMAND tiff_generation_test)
set_tests_properties(TiffGenerationTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# FFTW Multithreading Performance Test
find_package(OpenMP REQUIRED)

# Find FFTW libraries
find_path(FFTW_INCLUDE_DIR
    NAMES fftw3.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

find_library(FFTW_LIBRARY
    NAMES fftw3f
    PATHS /usr/include /usr/local/include /opt/local/include
)



find_library(FFTW_THREADS_LIBRARY
    NAMES fftw3f_threads
    PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
)

find_library(LIKWID_LIBRARY
    NAMES likwid
    PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
)
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

if (FFTW_INCLUDE_DIR AND FFTW_LIBRARY AND FFTW_THREADS_LIBRARY)
    add_executable(fftw_multithreading_test
        fftw_multithreading_test.cpp
    )
    target_compile_options(fftw_multithreading_test PRIVATE -DLIKWID_PERFMON)
    
    # Set C++ standard and compiler flags
    set_property(TARGET fftw_multithreading_test PROPERTY CXX_STANDARD 17)
    target_compile_options(fftw_multithreading_test PRIVATE ${OpenMP_CXX_FLAGS})
    
    # Link libraries for multithreading test
    target_link_libraries(fftw_multithreading_test
        PRIVATE
            ${FFTW_LIBRARY}
            ${FFTW_THREADS_LIBRARY}
            ${LIKWID_LIBRARY}
            OpenMP::OpenMP_CXX
            pthread
    )
    
    # Include directories for multithreading test
    target_include_directories(fftw_multithreading_test
        PRIVATE
            ${LIKWID_INCLUDE_DIR}
            ${FFTW_INCLUDE_DIR}
    )
    
    # Add multithreading test to test suite
    add_test(NAME FFTWMultithreadingTest COMMAND fftw_multithreading_test)
    set_tests_properties(FFTWMultithreadingTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    
    # FFTW Size Decomposition Test
    add_executable(fftw_size_decomposition_test
        fftw_size_decomposition_test.cpp
    )
    target_compile_options(fftw_size_decomposition_test PRIVATE -DLIKWID_PERFMON)
    
    # Set C++ standard and compiler flags
    set_property(TARGET fftw_size_decomposition_test PROPERTY CXX_STANDARD 17)
    target_compile_options(fftw_size_decomposition_test PRIVATE ${OpenMP_CXX_FLAGS})
    
    # Link libraries for size decomposition test
    target_link_libraries(fftw_size_decomposition_test
        PRIVATE
            ${FFTW_LIBRARY}
            ${FFTW_THREADS_LIBRARY}
            ${LIKWID_LIBRARY}
            OpenMP::OpenMP_CXX
            pthread
    )
    
    # Include directories for size decomposition test
    target_include_directories(fftw_size_decomposition_test
        PRIVATE
            ${LIKWID_INCLUDE_DIR}
            ${FFTW_INCLUDE_DIR}
    )
    
    # Add size decomposition test to test suite
    add_test(NAME FFTWSizeDecompositionTest COMMAND fftw_size_decomposition_test)
    set_tests_properties(FFTWSizeDecompositionTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    
    message(STATUS "FFTW multithreading test will be built")
    message(STATUS "FFTW size decomposition test will be built")
    message(STATUS "FFTW library: ${FFTW_LIBRARY}")
    message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
else()
    message(STATUS "FFTW libraries not found - FFTW tests will not be built")
    message(STATUS "FFTW include dir: ${FFTW_INCLUDE_DIR}")
    message(STATUS "FFTW library: ${FFTW_LIBRARY}")
    message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
endif()
