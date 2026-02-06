#include "utl.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


namespace CUBE_UTL_COPY {
    // Copying complex_t datatype to GPU
    cudaError_t copyDataFromHostToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t event;
        cudaEventCreate(&event);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyHostToDevice, stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }
        
        cudaEventDestroy(event);
        return cudaSuccess;
    }
    
    cudaError_t copyDataFromDeviceToHost(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t event;
        cudaEventCreate(&event);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyDeviceToHost, stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }
        
        cudaEventDestroy(event);
        return cudaSuccess;
    }
    
    cudaError_t copyDataFromDeviceToDevice(int Nx, int Ny, int Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
        cudaEvent_t event;
        cudaEventCreate(&event);

        cudaMemcpyAsync(dest, src, sizeof(complex_t)*Nx*Ny*Nz, cudaMemcpyDeviceToDevice, stream);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }
        
        cudaEventDestroy(event);
        return cudaSuccess;
    }
}