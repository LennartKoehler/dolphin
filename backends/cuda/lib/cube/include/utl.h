#pragma once
#include <cuComplex.h>
#include <cufft.h>
#include <cufftw.h>



namespace CUBE_UTL_COPY {
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src, cudaStream_t stream = 0);
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz, fftw_complex* dest, fftw_complex* src, cudaStream_t stream = 0);
    void copyDataFromDeviceToDevice(int Nx, int Ny, int Nz, fftw_complex* dest, fftw_complex* src, cudaStream_t stream = 0);
}

