# backends/cpu/CMakeLists.txt
# FFTW wisdom file path - can be customized for different operating systems
# Default: "~/.fftw/wisdom" on Unix, ".fftw/wisdom" in user directory on Windows
if (NOT DEFINED FFTW_WISDOM_PATH)
    if (WIN32)
        set(FFTW_WISDOM_PATH ".fftw/wisdom" CACHE PATH "Path to FFTW wisdom file")
    else()
        set(FFTW_WISDOM_PATH "~/.cache/dolphin/fftw/wisdom" CACHE PATH "Path to FFTW wisdom file")
    endif()
endif()
# Try vcpkg/CMake config first (threads compiled into main lib on Windows)
find_package(FFTW3f CONFIG QUIET)

if(FFTW3f_FOUND)
    message(STATUS "Found FFTW3f via CMake config")
    set(FFTW3_LIB FFTW3::fftw3f)
    set(FFTW3_THREADS_LIB "")
else()
    # Manual find for Unix/homebrew/local installs
    find_library(FFTW3_LIB NAMES fftw3f PATHS ${FFTW_PATH} REQUIRED)
    find_library(FFTW3_THREADS_LIB NAMES fftw3f_threads fftw3f_omp PATHS ${FFTW_PATH} REQUIRED)

    find_path(FFTW_INCLUDE_DIR NAMES fftw3.h
        PATHS ${FFTW_PATH}/../include /usr/include /usr/local/include /opt/homebrew/include
        REQUIRED)
endif()

# If using OpenMP version of FFTW, find OpenMP
if(FFTW3_THREADS_LIB MATCHES "omp")
    message(STATUS "Linking OpenMP")
    find_package(OpenMP REQUIRED)
    set(USE_OPENMP TRUE)
endif()

add_library(cpu_backend STATIC
    src/CPUBackend.cpp
    src/CPUBackendManager.cpp
)

target_include_directories(cpu_backend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/spdlog/include>
)

if(FFTW_INCLUDE_DIR)
    target_include_directories(cpu_backend PUBLIC $<BUILD_INTERFACE:${FFTW_INCLUDE_DIR}>)
endif()




# Link FFTW shared libraries
target_link_libraries(cpu_backend PUBLIC
    ${FFTW3_LIB}
    ${FFTW3_THREADS_LIB}
    )
if(USE_OPENMP)
    target_link_libraries(cpu_backend PUBLIC
    OpenMP::OpenMP_CXX)
endif()

target_link_libraries(cpu_backend PRIVATE
    project_options)

# Define FFTW wisdom file path as a compile definition
# Can be overridden at build time with -DFFTW_WISDOM_PATH="custom/path"
target_compile_definitions(cpu_backend PRIVATE
    FFTW_WISDOM_PATH="${FFTW_WISDOM_PATH}"
)

target_link_libraries(cpu_backend PUBLIC
    dolphinbackend)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()



