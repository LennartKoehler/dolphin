find_package(CUDAToolkit 12.8 QUIET)

if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit not found. Disabling CUDA features.")

    set(BUILD_CUDA OFF PARENT_SCOPE)

    return()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/cube ${CMAKE_BINARY_DIR}/cube/build)

add_library(cuda_backend STATIC
    src/CUDABackend.cpp
    src/CUDABackendManager.cpp
)

set_target_properties(cuda_backend PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(cuda_backend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/lib/spdlog/include>
)




if(TARGET CUDA::cufft_static)
    target_link_libraries(cuda_backend PUBLIC CUDA::cufft_static)
else()
    target_link_libraries(cuda_backend PUBLIC CUDA::cufft)
endif()

target_link_libraries(cuda_backend PUBLIC
    CUBE
    dolphinbackend
)
set_target_properties(CUBE PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cuda_backend PRIVATE project_options)



if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

