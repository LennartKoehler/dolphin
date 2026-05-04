# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)


enable_testing()

add_subdirectory(mainTest)
add_subdirectory(cubeShapeTest)
add_executable(dolphin_tests
    memory_exception_example.cpp
    tiff_generation_test.cpp
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

add_executable(read_write_test
    read_write_test.cpp)
target_link_libraries(read_write_test
    PRIVATE
        dolphin

)


# Include directories for TIFF generation test
target_include_directories(read_write_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)

add_test(NAME ReadWriteTest COMMAND read_write_test)




# ImageSplit Test
add_executable(imageSplit_test
    imageSplit_test.cpp)
target_link_libraries(imageSplit_test
    PRIVATE
        dolphin
)

add_test(NAME ImageSplitTest COMMAND imageSplit_test)


add_executable(feathering_test
    feathering_test.cpp)

target_link_libraries(feathering_test
    PRIVATE
        dolphin
        )

target_include_directories(feathering_test
    PRIVATE
        BEFORE ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)


add_test(NAME FeatheringTest COMMAND feathering_test)



# Standalone TIFF generation test
add_executable(real_complex_fftw
    real_complex_fftw.cpp
)

# Link libraries to TIFF generation test
target_link_libraries(real_complex_fftw
    PRIVATE
        dolphin
)


# Add TIFF generation test to test suite
add_test(NAME RealComplexFftw COMMAND real_complex_fftw)
set_tests_properties(RealComplexFftw PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})



# Standalone TIFF generation test
add_executable(tiff_generation_test
    tiff_generation_test.cpp
)

# Link libraries to TIFF generation test
target_link_libraries(tiff_generation_test
    PRIVATE
        dolphin
)


# Add TIFF generation test to test suite
add_test(NAME TiffGenerationTest COMMAND tiff_generation_test)
set_tests_properties(TiffGenerationTest PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# FFTW Multithreading Performance Test
find_package(OpenMP REQUIRED)







find_library(LIKWID_LIBRARY
    NAMES likwid
    PATHS /usr/lib /usr/local/lib /opt/local/lib /usr/lib/x86_64-linux-gnu
)
find_path(LIKWID_INCLUDE_DIR
    NAMES likwid.h
    PATHS /usr/include /usr/local/include /opt/local/include
)

add_executable(fftw_multithreading_test
    fftw_multithreading_test.cpp
)

# Set C++ standard and compiler flags
set_property(TARGET fftw_multithreading_test PROPERTY CXX_STANDARD 17)
target_compile_options(fftw_multithreading_test PRIVATE ${OpenMP_CXX_FLAGS})

# Base libraries for multithreading test
target_link_libraries(fftw_multithreading_test
    PRIVATE
        dolphin
        OpenMP::OpenMP_CXX
        pthread
)

# Base include directories for multithreading test
target_include_directories(fftw_multithreading_test
    PRIVATE
        ${FFTW_INCLUDE_DIR}
)

# Add LIKWID support if available
if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    target_compile_options(fftw_multithreading_test PRIVATE -DLIKWID_PERFMON)
    target_link_libraries(fftw_multithreading_test PRIVATE ${LIKWID_LIBRARY})
    target_include_directories(fftw_multithreading_test PRIVATE ${LIKWID_INCLUDE_DIR})
    message(STATUS "LIKWID support enabled for fftw_multithreading_test")
else()
    message(STATUS "LIKWID not found - building fftw_multithreading_test without LIKWID support")
endif()

# Add multithreading test to test suite
add_test(NAME FFTWMultithreadingTest COMMAND fftw_multithreading_test)
set_tests_properties(FFTWMultithreadingTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFTW Size Decomposition Test
add_executable(fftw_size_decomposition_test
    fftw_size_decomposition_test.cpp
)
target_link_libraries(fftw_size_decomposition_test
    PRIVATE
    project_options)

# Set C++ standard and compiler flags
set_property(TARGET fftw_size_decomposition_test PROPERTY CXX_STANDARD 17)
target_compile_options(fftw_size_decomposition_test PRIVATE ${OpenMP_CXX_FLAGS})

# Base libraries for size decomposition test
target_link_libraries(fftw_size_decomposition_test
    PRIVATE
        dolphin
        OpenMP::OpenMP_CXX
        pthread
)



# Add LIKWID support if available
if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    target_compile_options(fftw_size_decomposition_test PRIVATE -DLIKWID_PERFMON)
    target_link_libraries(fftw_size_decomposition_test PRIVATE ${LIKWID_LIBRARY})
    target_include_directories(fftw_size_decomposition_test PRIVATE ${LIKWID_INCLUDE_DIR})
    message(STATUS "LIKWID support enabled for fftw_size_decomposition_test")
else()
    message(STATUS "LIKWID not found - building fftw_size_decomposition_test without LIKWID support")
endif()

# Add size decomposition test to test suite
add_test(NAME FFTWSizeDecompositionTest COMMAND fftw_size_decomposition_test)
set_tests_properties(FFTWSizeDecompositionTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFT Backend Test - Tests forward and backward FFT using the backend
add_executable(fft_backend_test
    fft_backend_test.cpp
)
target_link_libraries(fft_backend_test
    PRIVATE
        dolphin
)
target_include_directories(fft_backend_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME FFTBackendTest COMMAND fft_backend_test)
set_tests_properties(FFTBackendTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})


add_executable(cpu_data_reinterpret_test
    cpu_data_reinterpret_test.cpp
)
target_link_libraries(cpu_data_reinterpret_test
    PRIVATE
        dolphin
)
target_include_directories(cpu_data_reinterpret_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME CPUDataReinterpret COMMAND cpu_data_reinterpret_test)
set_tests_properties(CPUDataReinterpret PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(fft_real_vs_complex_test
    fft_real_vs_complex_test.cpp
)
target_link_libraries(fft_real_vs_complex_test
    PRIVATE
        dolphin
)
target_include_directories(fft_real_vs_complex_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME FFTComplexVsReal COMMAND fft_real_vs_complex_test)
set_tests_properties(FFTComplexVsReal PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

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
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME ImageComparisonTest COMMAND image_comparison_test)
set_tests_properties(ImageComparisonTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# FFTW In-Place vs Out-of-Place R2C Test
add_executable(fftw_inplace_vs_outofplace_r2c_test
    fftw_inplace_vs_outofplace_r2c_test.cpp
)
target_link_libraries(fftw_inplace_vs_outofplace_r2c_test
    PRIVATE
        dolphin
)
target_include_directories(fftw_inplace_vs_outofplace_r2c_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/backends/cpu/include
        ${CMAKE_SOURCE_DIR}/backends/dolphinbackend/include
)
add_test(NAME FFTWInplaceVsOutofplaceR2C COMMAND fftw_inplace_vs_outofplace_r2c_test)
set_tests_properties(FFTWInplaceVsOutofplaceR2C PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

#  Padding Test
add_executable(padding_test
    padding_test.cpp
)
target_link_libraries(padding_test
    PRIVATE
        dolphin
)
target_include_directories(padding_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME PaddingTest COMMAND padding_test)
set_tests_properties(PaddingTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

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
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../src
)
add_test(NAME GetLabelGroupsTest COMMAND getLabelGroups_test)
set_tests_properties(GetLabelGroupsTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

message(STATUS "FFTW multithreading test will be built")
message(STATUS "FFTW size decomposition test will be built")
message(STATUS "FFTW library: ${FFTW_LIBRARY}")
message(STATUS "FFTW threads library: ${FFTW_THREADS_LIBRARY}")
if (LIKWID_LIBRARY AND LIKWID_INCLUDE_DIR)
    message(STATUS "LIKWID library: ${LIKWID_LIBRARY}")
    message(STATUS "LIKWID include dir: ${LIKWID_INCLUDE_DIR}")
else()
    message(STATUS "LIKWID library not found - tests will be built without LIKWID support")
endif()

