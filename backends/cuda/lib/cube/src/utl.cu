#include "utl.h"
#include "kernels.h"
#include <cuComplex.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#ifdef ENABLE_CUBEUTL_DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) // Nichts tun
#endif




namespace CUBE_UTL_COPY {
    // Copying fftw_complex datatype to GPU
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(fftw_complex)*Nx*Ny*Nz, cudaMemcpyHostToDevice, stream);

        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(fftw_complex)*Nx*Ny*Nz<<"B] Copy Data from Host to Device");

    }
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(fftw_complex)*Nx*Ny*Nz, cudaMemcpyDeviceToHost, stream);
        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(fftw_complex)*Nx*Ny*Nz<<"B] Copy Data from Device to Host");
    }
    void copyDataFromDeviceToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(fftw_complex)*Nx*Ny*Nz, cudaMemcpyDeviceToDevice, stream);
        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(fftw_complex)*Nx*Ny*Nz<<"B] Copy Data from Device to Host");
    }
}





