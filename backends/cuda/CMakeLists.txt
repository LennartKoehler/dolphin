cmake_minimum_required(VERSION 3.1)

set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Debug)
project(cudaBackend)

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/cube ${CMAKE_BINARY_DIR}/cube/build)
set(DOUBLE_PRECISION OFF) # explicit



#---------------------------------------
add_library(project_options INTERFACE)

target_compile_options(project_options INTERFACE
  # Debug
  $<$<CONFIG:Debug>:-g;-O0;-fno-omit-frame-pointer>
  # Release
  $<$<CONFIG:Release>:-O3;-ffast-math;-DNDEBUG;-funroll-loops;-mavx2;-mfma>
  # RelWithDebInfo
  $<$<CONFIG:RelWithDebInfo>:-O3;-g;-ffast-math;-DNDEBUG;-funroll-loops>
)

# architecture flags only for Release (avoid -march=native for Debug)
target_compile_options(project_options INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU>>:-mavx2;-mfma>
)

#---------------------------------------








add_library(dolphinbackend STATIC IMPORTED)

set_target_properties(dolphinbackend PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/lib/libdolphinbackend.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/lib/dolphinbackend/include"
)


find_package(CUDAToolkit 12.8 REQUIRED)

add_library(cuda_backend SHARED
    src/CUDABackend.cpp
    src/CUDABackendManager.cpp
)

target_include_directories(cuda_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_backend>
        ${CMAKE_SOURCE_DIR}/lib/cube/include
)




target_link_libraries(cuda_backend PUBLIC 
    CUDA::cufft
    CUBE
    dolphinbackend
)
set_target_properties(CUBE PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(cuda_backend PROPERTIES
    OUTPUT_NAME "cuda_backend"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)


target_link_libraries(cuda_backend PUBLIC project_options)



option(ENABLE_TESTS "Enable tests" ON)
if (${ENABLE_TESTS})

    set(CMAKE_CTEST_ARGUMENTS "--verbose")


    enable_testing()
    add_subdirectory(tests)

endif()


