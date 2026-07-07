#pragma once
#include "CUBE.h"



namespace CUBE_UTL_COPY {
    cudaError_t copyDataFromHostToDevice(size_t Nx, size_t Ny, size_t Nz,complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    cudaError_t copyDataFromDeviceToHost(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    cudaError_t copyDataFromDeviceToDevice(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
}
