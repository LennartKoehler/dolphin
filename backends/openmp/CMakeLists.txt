# backends/openmp/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(OpenMPBackend VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT DEFINED FFTW_PATH)
    set(FFTW_PATH "/usr/local/fftw-debug/lib" CACHE PATH "Path to FFTW installation")
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a" ".lib" ".so" ".dll" ".dylib")
message(STATUS "FFTW_PATH=${FFTW_PATH}")

find_library(FFTW3_LIB NAMES fftw3f PATHS ${FFTW_PATH} NO_DEFAULT_PATH REQUIRED)
find_library(FFTW3_THREADS_LIB NAMES fftw3f_threads PATHS ${FFTW_PATH} NO_DEFAULT_PATH REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

# find_library(OpenMP REQUIRED)



# add_library(dolphinbackend STATIC IMPORTED)

# set_target_properties(dolphinbackend PROPERTIES
#     IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/lib/libdolphinbackend.a"
#     INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/include"
#     INTERFACE_LINK_LIBRARIES ""  # empty if it has no dependencies
# )

add_library(openmp_backend STATIC
    src/OpenMPBackend.cpp
)

target_include_directories(openmp_backend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/openmp_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
)




# Link FFTW shared libraries
target_link_libraries(openmp_backend PUBLIC
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    # OpenMP::OpenMP_CXX
    )

target_link_libraries(openmp_backend PUBLIC project_options)

target_link_libraries(openmp_backend PUBLIC
    dolphinbackend)



