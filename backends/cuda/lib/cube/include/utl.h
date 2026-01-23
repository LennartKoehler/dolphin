#pragma once
#include "CUBE.h"



namespace CUBE_UTL_COPY {
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
    void copyDataFromDeviceToDevice(int Nx, int Ny, int Nz, complex_t* dest, complex_t* src, cudaStream_t stream = 0);
}

