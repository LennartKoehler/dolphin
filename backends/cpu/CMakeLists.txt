# backends/cpu/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(CPUBackend VERSION 1.0.0)


if (NOT DEFINED FFTW_PATH)
    set(FFTW_PATH "/usr/local/fftw-debug/lib" CACHE PATH "Path to FFTW installation")
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so" ".dll" ".dylib")
find_library(FFTW3_LIB NAMES fftw3f PATHS ${FFTW_PATH} NO_DEFAULT_PATH REQUIRED)

find_library(FFTW3_THREADS_LIB NAMES fftw3f_threads fftw3f_omp PATHS ${FFTW_PATH} NO_DEFAULT_PATH REQUIRED)

# find_package(OpenMP REQUIRED) # TODO make optional

# add_library(dolphinbackend STATIC IMPORTED)
# set_target_properties(dolphinbackend PROPERTIES
#     IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/lib/libdolphinbackend.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/include"
#     INTERFACE_LINK_LIBRARIES ""  # empty if it has no dependencies
# )

add_library(cpu_backend STATIC
    src/CPUBackend.cpp
    src/CPUBackendManager.cpp
)

target_include_directories(cpu_backend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)




# Link FFTW shared libraries
target_link_libraries(cpu_backend PUBLIC
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    )
if(OpenMP)
    target_link_libraries(cpu_backend PUBLIC
    OpenMP::OpenMP_CXX)
endif()

target_link_libraries(cpu_backend PUBLIC project_options)

target_link_libraries(cpu_backend PUBLIC
    dolphinbackend)


# target_compile_options(cpu_backend PRIVATE
#     -O3
#     -march=native
#     -fopt-info-vec-optimized
#     -g -ffast-math -DNDEBUG -funroll-loops -mavx2 -mfma
# )

# # Specify library output directory
# set_target_properties(cpu_backend PROPERTIES
#     OUTPUT_NAME "cpu_backend"
#     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
#     LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
#     LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
# )

# # Install library and headers
# install(TARGETS cpu_backend
#     EXPORT cpu_backendTargets
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include
# )

# install(DIRECTORY include/cpu_backend/
#     DESTINATION include/cpu_backend
# )

# # Export CMake targets for find_package
# install(EXPORT cpu_backendTargets
#     FILE cpu_backendTargets.cmake
#     NAMESPACE cpu_backend::
#     DESTINATION lib/cmake/cpu_backend
# )

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/cpu_backendConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion
# )

# configure_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpu_backendConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/cpu_backendConfig.cmake"
#     @ONLY
# )

# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/cpu_backendConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/cpu_backendConfigVersion.cmake"
#     DESTINATION lib/cmake/cpu_backend
# )

# option(ENABLE_TESTS "Enable tests" ON)
# if (${ENABLE_TESTS})

#     set(CMAKE_CTEST_ARGUMENTS "--verbose")


#     enable_testing()
#     add_subdirectory(tests)

# endif()



