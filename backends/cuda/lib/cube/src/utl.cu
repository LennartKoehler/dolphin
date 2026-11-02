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
#include "launch.h"
#include <iostream>
#include <cstdlib>
#include <ctime>


namespace CUBE_UTL_COPY {

    // Shared implementation for all device transfers: async copy + sync
    static cudaError_t copyData(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaMemcpyKind kind, cudaStream_t stream) {
        return launchSync([=] { cudaMemcpyAsync(dest, src, sizeof(complex_t) * Nx * Ny * Nz, kind, stream); }, stream);
    }

    // Copying complex_t datatype to GPU
    cudaError_t copyDataFromHostToDevice(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaStream_t stream) {
        return copyData(Nx, Ny, Nz, dest, src, cudaMemcpyHostToDevice, stream);
    }

    cudaError_t copyDataFromDeviceToHost(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaStream_t stream) {
        return copyData(Nx, Ny, Nz, dest, src, cudaMemcpyDeviceToHost, stream);
    }

    cudaError_t copyDataFromDeviceToDevice(size_t Nx, size_t Ny, size_t Nz, complex_t* dest, complex_t* src, cudaStream_t stream) {
        return copyData(Nx, Ny, Nz, dest, src, cudaMemcpyDeviceToDevice, stream);
    }
}
