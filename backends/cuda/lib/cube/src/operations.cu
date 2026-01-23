#include <cuda_runtime_api.h>
#include <operations.h>
#include <kernels.h>
#include <thread>
#include <iostream>

#ifdef ENABLE_CUBE_DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) // Nichts tun
#endif

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

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        complexMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] MatMul in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }

    cudaError_t complexScalarMul(int Nx, int Ny, int Nz, complex_t* A, complex_t B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        cudaError_t err = cudaGetLastError();

        cudaEventRecord(start, stream);

        complexScalarMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] scalar Mul in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t complexAddition(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        cudaError_t err = cudaGetLastError();

        cudaEventRecord(start, stream);

        complexAdditionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] add in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }


    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMul(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        cudaError_t err = cudaGetLastError();

        cudaEventRecord(start, stream);

        complexElementwiseMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t complexElementwiseMatMulConjugate(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start, stream);

        complexElementwiseMatMulConjugateGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul conjugated in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    
    cudaError_t complexElementwiseMatDiv(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start, stream);

        complexElementwiseMatDivGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    
    cudaError_t complexElementwiseMatDivStabilized(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start, stream);

        complexElementwiseMatDivStabilizedGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

        
        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise stabilized MatDiv in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(int Nx, int Ny, int Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream) {
        if (!psf || !laplacian_fft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        calculateLaplacianGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, psf, laplacian_fft);
        

        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t gradX(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        gradientXGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradX);
        

        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientX in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t gradY(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        gradientYGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradY);
        

        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientY in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t gradZ(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        gradientZGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradZ);
        

        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientZ in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t computeTV(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv, cudaStream_t stream) {
        if (!gx || !gy || !gz || !tv) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        computeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, lambda, gx, gy, gz, tv);
        

        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Total Variation in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    
    cudaError_t normalizeTV(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        normalizeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon);
        

        cudaEventRecord(stop, stream);
        // cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] normalizing Total Variation in CUDA ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
        if (!Afft || !laplacianfft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        cudaEventRecord(start);

        calculateLaplacianTiledGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK>>>(Nx, Ny, Nz, Afft, laplacianfft);
        

        cudaEventRecord(stop);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA tiled with shared mem ("<<GLOBAL_THREADS_PER_BLOCK.x*GLOBAL_THREADS_PER_BLOCK.y*GLOBAL_THREADS_PER_BLOCK.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}

namespace CUBE_FTT {
    // Fourier Shift, Padding and Normalization
    cudaError_t octantFourierShiftCPU(int Nx, int Ny, int Nz, complex_t* data) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        int width = Nx;
        int height = Ny;
        int depth = Nz;
        auto start = std::chrono::high_resolution_clock::now();

        int halfWidth = width / 2;
        int halfHeight = height / 2;
        int halfDepth = depth / 2;

        // Sequential version (removed OpenMP parallelization)
        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Calculate the indices for the swap
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = ((z + halfDepth) % depth) * height * width + ((y + halfHeight) % height) * width + ((x + halfWidth) % width);

                    // Perform the swap only if the indices are different
                    if (idx1 != idx2) {
                        // Swap real parts
                        real_t temp_real = data[idx1][0];
                        data[idx1][0] = data[idx2][0];
                        data[idx2][0] = temp_real;

                        // Swap imaginary parts
                        real_t temp_imag = data[idx1][1];
                        data[idx1][1] = data[idx2][1];
                        data[idx2][1] = temp_imag;
                    }
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        float time = duration.count();

        DEBUG_LOG("[TIME]["<<time/1000000<<" ms] Octant(Fourier)Shift in CPP");
        
        return cudaSuccess;
    }
    
    cudaError_t octantFourierShift(int Nx, int Ny, int Nz, complex_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);


        cudaEventRecord(start, stream);

        octantFourierShiftGlobal<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, data);
        cudaError_t errp = cudaPeekAtLastError();
        

        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Octant(Fouriere)Shift in CUDA ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return (errp == cudaSuccess) ? err : errp;
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

        DEBUG_LOG("[TIME]["<<time/1000000<<" ms] padded Mat in CPP");
        
        return cudaSuccess;
    }
    
    cudaError_t normalizeData(int Nx, int Ny, int Nz, complex_t* d_data, cudaStream_t stream) {
        if (!d_data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 1024;
        int num_blocks = (num_elements + block_size - 1) / block_size;

        cudaEventRecord(start, stream);

        normalizeDataGlobal<<<num_blocks, block_size, 0, stream>>>(Nx, Ny, Nz, d_data);
        cudaError_t errp = cudaPeekAtLastError();
        

        cudaEventRecord(stop, stream);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Normalizing data in CUDA ("<<block_size*num_blocks<< " Threads)");
        
        return (errp == cudaSuccess) ? err : errp;
    }
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        cudaEventRecord(start);

        deviceTestKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        

        cudaEventRecord(stop);
        //cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Device kernel(s) finished ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}