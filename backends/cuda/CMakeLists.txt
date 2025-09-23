# find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/cube ${CMAKE_BINARY_DIR}/cube/build)

add_library(cuda_backend SHARED
    CUDABackend.cpp
)

target_include_directories(cuda_backend PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/lib/cube/include  # Add CUBE's include directory

)

# Link CUDA cufftw (comes with CUDA toolkit)
target_link_libraries(cuda_backend PRIVATE 
    CUBE
    CUDA::cudart
    CUDA::cufft
    CUDA::cufftw
    OpenMP::OpenMP_CXX
)

set_target_properties(cuda_backend PROPERTIES OUTPUT_NAME "cuda_backend")




