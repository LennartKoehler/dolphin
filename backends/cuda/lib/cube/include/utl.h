#pragma once
#include "CUBE.h"



namespace CUBE_UTL_COPY {
    cudaError_t copyDataFromHostToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    cudaError_t copyDataFromDeviceToHost(int Nx, int Ny, int Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    cudaError_t copyDataFromDeviceToDevice(int Nx, int Ny, int Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
}

