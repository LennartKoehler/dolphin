/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/
#include "utl.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


namespace CUBE_UTL_COPY {
    // Copying complex_t datatype to GPU
    cudaError_t copyDataFromHostToDevice(size_t Nx, size_t Ny, size_t Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
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
    
    cudaError_t copyDataFromDeviceToHost(size_t Nx, size_t Ny, size_t Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
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
    
    cudaError_t copyDataFromDeviceToDevice(size_t Nx, size_t Ny, size_t Nz,complex_t* dest, complex_t* src, cudaStream_t stream) {
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
