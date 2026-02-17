#include <cuda_runtime_api.h>
#include <operations.h>
#include <kernels.h>
#include <thread>
#include <iostream>



// Global CUDA kernel configuration
namespace {
    // Global thread block configuration
    const dim3 GLOBAL_THREADS_PER_BLOCK(4, 8, 8);
    
    // Helper function to compute blocks per grid
    inline dim3 computeBlocksPerGrid(int Nx, int Ny, int Nz) {
        return dim3(
            (Nx + GLOBAL_THREADS_PER_BLOCK.x - 1) / GLOBAL_THREADS_PER_BLOCK.x,
            (Ny + GLOBAL_THREADS_PER_BLOCK.y - 1) / GLOBAL_THREADS_PER_BLOCK.y,
            (Nz + GLOBAL_THREADS_PER_BLOCK.z - 1) / GLOBAL_THREADS_PER_BLOCK.z
        );
    }
}

namespace CUBE_MAT {

    cudaError_t complexMatMul(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

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

    cudaError_t complexScalarMul(int Nx, int Ny, int Nz, complex_t* A, complex_t B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexScalarMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

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
    


    cudaError_t sumToOneReal(int Nx, int Ny, int Nz, complex_t** A, int nImages, int imageVolume, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        sumToOneReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, nImages, imageVolume);

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

    cudaError_t complexAddition(int Nx, int Ny, int Nz, complex_t** A, complex_t* sums, int nImages, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexAdditionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, sums, nImages);

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

    cudaError_t complexAddition(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexAdditionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

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


    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMul(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

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
    
    cudaError_t complexElementwiseMatMulConjugate(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatMulConjugateGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

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
    
    cudaError_t complexElementwiseMatDiv(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatDivGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

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
    
    cudaError_t complexElementwiseMatDivStabilized(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatDivStabilizedGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

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

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(int Nx, int Ny, int Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream) {
        if (!psf || !laplacian_fft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        calculateLaplacianGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, psf, laplacian_fft);

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
    
    cudaError_t gradX(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientXGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradX);

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
    
    cudaError_t gradY(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientYGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradY);

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
    
    cudaError_t gradZ(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientZGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradZ);

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
    
    cudaError_t computeTV(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv, cudaStream_t stream) {
        if (!gx || !gy || !gz || !tv) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        computeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, lambda, gx, gy, gz, tv);

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
    
    cudaError_t normalizeTV(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        normalizeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon);

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

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
        if (!Afft || !laplacianfft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        calculateLaplacianTiledGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK>>>(Nx, Ny, Nz, Afft, laplacianfft);

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

namespace CUBE_FTT {

    
    cudaError_t octantFourierShift(int Nx, int Ny, int Nz, complex_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        octantFourierShiftGlobal<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, data);
        cudaError_t errp = cudaPeekAtLastError();
        if (errp != cudaSuccess) {
            cudaEventDestroy(event);
            return errp;
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
    
    cudaError_t padMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat)
    {
        if (!oldMat || !newMat) {
            return cudaErrorInvalidValue;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        // Sicherheitsprüfung: Neue Dimensionen müssen größer oder gleich den alten sein
        if (newNx < oldNx || newNy < oldNy || newNz < oldNz) {
            return cudaErrorInvalidValue;
        }

        // Offset für Padding (Startkoordinaten der alten Matrix in der neuen Matrix)
        int offsetX = (newNx - oldNx) / 2;
        int offsetY = (newNy - oldNy) / 2;
        int offsetZ = (newNz - oldNz) / 2;

        // Initialisiere die neue Matrix mit Nullen
        for (int i = 0; i < newNx * newNy * newNz; ++i) {
            newMat[i][0] = 0.0; // Realteil
            newMat[i][1] = 0.0; // Imaginärteil
        }

        // Kopiere die Werte der alten Matrix in die Mitte der neuen Matrix
        for (int z = 0; z < oldNz; ++z) {
            for (int y = 0; y < oldNy; ++y) {
                for (int x = 0; x < oldNx; ++x) {
                    // Index in der alten Matrix
                    int oldIndex = z * oldNy * oldNx + y * oldNx + x;

                    // Index in der neuen Matrix
                    int newIndex =
                        (z + offsetZ) * newNy * newNx +
                        (y + offsetY) * newNx +
                        (x + offsetX);

                    // Kopiere den Wert
                    newMat[newIndex][0] = oldMat[oldIndex][0]; // Realteil
                    newMat[newIndex][1] = oldMat[oldIndex][1]; // Imaginärteil
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        float time = duration.count();

        
        
        return cudaSuccess;
    }
    
    cudaError_t normalizeData(int Nx, int Ny, int Nz, complex_t* d_data, cudaStream_t stream) {
        if (!d_data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        int num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 1024;
        int num_blocks = (num_elements + block_size - 1) / block_size;

        normalizeDataGlobal<<<num_blocks, block_size, 0, stream>>>(Nx, Ny, Nz, d_data);
        cudaError_t errp = cudaPeekAtLastError();
        if (errp != cudaSuccess) {
            cudaEventDestroy(event);
            return errp;
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

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        deviceTestKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);

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