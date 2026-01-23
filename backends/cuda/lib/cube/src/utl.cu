#include "utl.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

#ifdef ENABLE_CUBEUTL_DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) // Nichts tun
#endif



namespace CUBE_UTL_COPY {
    // Copying complex_t datatype to GPU
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyHostToDevice, stream);

        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(complex_t)*Nx*Ny*Nz<<"B] Copy Data from Host to Device");

    }
    
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyDeviceToHost, stream);
        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(complex_t)*Nx*Ny*Nz<<"B] Copy Data from Device to Host");
    }
    
    void copyDataFromDeviceToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyDeviceToDevice, stream);
        
        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(complex_t)*Nx*Ny*Nz<<"B] Copy Data from Device to Host");
    }
}