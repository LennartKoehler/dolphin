#include <cuda_runtime_api.h>
#include <operations.h>
#include <kernels.h>
#include <thread>
#include <iostream>
#include <omp.h>

#ifdef ENABLE_CUBE_DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) // Nichts tun
#endif

namespace CUBE_MAT {
    // Normal Matrix Multiplication
    cudaError_t complexMatMulFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        double start = omp_get_wtime();

        // Iterate over the 3D matrix
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    int index = x * Ny * Nz + y * Nz + z; // Correct index for 3D matrix

                    fftw_complex sum = {0.0, 0.0};

                    // Perform matrix multiplication with summation over k
                    for (int k = 0; k < Nz; ++k) {  // Loop over k for multiplication
                        int indexA = x * Ny * Nz + y * Nz + k;
                        int indexB = k * Ny * Nz + y * Nz + z;

                        float realA = A[indexA][0];
                        float imagA = A[indexA][1];
                        float realB = B[indexB][0];
                        float imagB = B[indexB][1];

                        // Accumulate the real and imaginary parts
                        sum[0] += realA * realB - imagA * imagB;
                        sum[1] += realA * imagB + imagA * realB;
                    }

                    // Store the result
                    C[index][0] = sum[0];
                    C[index][1] = sum[1];
                }
            }
        }

        double end = omp_get_wtime();
        DEBUG_LOG("[TIME][" << (end - start) * 1000 << " ms] MatMul in Cpp");
        
        return cudaSuccess;
    }
    cudaError_t complexMatMulFftwComplexOmpCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        double start = omp_get_wtime();

        // Use OpenMP to parallelize the 3D matrix multiplication
#pragma omp parallel for collapse(3) schedule(static)
        for (int x = 0; x < Nx; ++x) {
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    int index = x * Ny * Nz + y * Nz + z; // Correct index for 3D matrix

                    fftw_complex sum = {0.0, 0.0};

                    // Perform matrix multiplication with summation over k
                    for (int k = 0; k < Nz; ++k) {  // Loop over k for multiplication
                        int indexA = x * Ny * Nz + y * Nz + k;
                        int indexB = k * Ny * Nz + y * Nz + z;

                        float realA = A[indexA][0];
                        float imagA = A[indexA][1];
                        float realB = B[indexB][0];
                        float imagB = B[indexB][1];

                        // Accumulate the real and imaginary parts
                        sum[0] += realA * realB - imagA * imagB;
                        sum[1] += realA * imagB + imagA * realB;
                    }

                    // Store the result
                    C[index][0] = sum[0];
                    C[index][1] = sum[1];
                }
            }
        }

        double end = omp_get_wtime();
        DEBUG_LOG("[TIME][" << (end - start) * 1000 << " ms] MatMul in Cpp with Omp ("
                  << omp_get_max_threads() << " Threads)");
        
        return cudaSuccess;
    }
    cudaError_t complexMatMulFftwComplexCUDA(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
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

        complexMatMulFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] MatMul in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexMatMulCuComplexCUDA(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Kernel dimension 3D for Nx * Ny * Nz matrix stored in a 1D array
        dim3 threadsPerBlock(10, 10, 10); // Optimal for testing, adjust for your hardware
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        // Start the event for timing
        cudaEventRecord(start);

        // Launch the kernel
        complexMatMulCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);

        // Synchronize the device and check for errors after kernel execution
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Check for any CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        // Calculate and print the time elapsed
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] MatMul in CUDA (cuComplex) ("
                  << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z << "x"
                  << blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z << ")");
        
        return cudaSuccess;
    }
    cudaError_t complexMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type) {
        if (!A || !B || !C || !type) {
            return cudaErrorInvalidValue;
        }
        
        if (strcmp(type, "cpp") == 0) {
            return complexMatMulFftwComplexCPU(Nx, Ny, Nz, A, B, C);
        }else if (strcmp(type, "omp") == 0) {
            return complexMatMulFftwComplexOmpCPU(Nx, Ny, Nz, A, B, C);
        }else if (strcmp(type, "cuda") == 0) {
            return complexMatMulFftwComplexCUDA(Nx, Ny, Nz, A, B, C);
        }
        else {
            return cudaErrorInvalidValue;
        }
    }

    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMulCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
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

        complexElementwiseMatMulCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatMulCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
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



        complexElementwiseMatMulCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA (cufftComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
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
        cudaError_t err = cudaGetLastError();

        cudaEventRecord(start);

        complexElementwiseMatMulFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);

        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);


        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t complexElementwiseMatMulConjugateCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C)  {
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

        complexElementwiseMatMulConjugateCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul conjugated in CUDA (cufftComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatMulConjugateFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C)  {
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

        complexElementwiseMatMulConjugateFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatMul conjugated in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatDivCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
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

        complexElementwiseMatDivCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatDivCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
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

        complexElementwiseMatDivCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatDivFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon) {
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

        complexElementwiseMatDivFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }


    cudaError_t complexElementwiseMatDivNaiveCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
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

        complexElementwiseMatDivNaiveCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise naive MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatDivStabilizedCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon){
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

        complexElementwiseMatDivStabilizedCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise stabilized MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
    cudaError_t complexElementwiseMatDivStabilizedFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon){
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

        complexElementwiseMatDivStabilizedFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, A, B, C, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            return err;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] elementwise stabilized MatDiv in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return cudaSuccess;
    }
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacianCufftComplex(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
        if (!Afft || !laplacianfft) {
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

        calculateLaplacianCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, Afft, laplacianfft);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradXCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX) {
        if (!image || !gradX) {
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

        gradientXCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradX);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientX in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradYCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY) {
        if (!image || !gradY) {
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

        gradientYCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradY);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientY in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradZCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ) {
        if (!image || !gradZ) {
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

        gradientZCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradZ);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientZ in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t computeTVCufftComplex(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
        if (!gx || !gy || !gz || !tv) {
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

        computeTVCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, lambda, gx, gy, gz, tv);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t normalizeTVCufftComplex(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
        if (!gradX || !gradY || !gradZ) {
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

        normalizeTVCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] normalizing Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t calculateLaplacianFftwComplex(int Nx, int Ny, int Nz, fftw_complex* Afft, fftw_complex* laplacianfft) {
        if (!Afft || !laplacianfft) {
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

        calculateLaplacianFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, Afft, laplacianfft);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradXFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX) {
        if (!image || !gradX) {
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

        gradientXFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradX);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientX in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradYFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradY) {
        if (!image || !gradY) {
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

        gradientYFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradY);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientY in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t gradZFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradZ) {
        if (!image || !gradZ) {
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

        gradientZFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, image, gradZ);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating GradientZ in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t computeTVFftwComplex(int Nx, int Ny, int Nz, double lambda, fftw_complex *gx, fftw_complex *gy, fftw_complex *gz, fftw_complex *tv) {
        if (!gx || !gy || !gz || !tv) {
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

        computeTVFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, lambda, gx, gy, gz, tv);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
    cudaError_t normalizeTVFftwComplex(int Nx, int Ny, int Nz, fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, double epsilon) {
        if (!gradX || !gradY || !gradZ) {
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

        normalizeTVFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] normalizing Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianCufftComplexTiled(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
        if (!Afft || !laplacianfft) {
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

        calculateLaplacianCufftComplexTiledGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, Afft, laplacianfft);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA tiled with shared mem (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}

namespace CUBE_FTT {
    // FFT
    cudaError_t cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan) {
        if (!input || !output || !plan) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cufftResult result;

        cudaEventRecord(start);

        result = cufftExecC2C(plan, input, output, CUFFT_FORWARD);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Forward FFT in cuFFT");
        
        return err; 
    }
    cudaError_t cufftInverse(int Nx, int Ny, int Nz, cufftComplex* input, cufftComplex* output, cufftHandle plan) {
        if (!input || !output || !plan) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cufftResult result;

        cudaEventRecord(start);

        result = cufftExecC2C(plan, input, output, CUFFT_INVERSE);
        int num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 256;  // Blockgröße (kann angepasst werden)
        int num_blocks = (num_elements + block_size - 1) / block_size;  // Berechne die Anzahl der Blöcke

        normalizeComplexData<<<num_blocks, block_size>>>(Nx, Ny, Nz, output);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Inverse FFT in cuFFT");
        
        return err;
    }

    // Fourier Shift, Padding and Normalization
    cudaError_t octantFourierShiftFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* data) {
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

        // Parallelize the nested loops using OpenMP with collapsing to reduce overhead
#pragma omp parallel for collapse(3)
        for (int z = 0; z < halfDepth; ++z) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    // Calculate the indices for the swap
                    int idx1 = z * height * width + y * width + x;
                    int idx2 = ((z + halfDepth) % depth) * height * width + ((y + halfHeight) % height) * width + ((x + halfWidth) % width);

                    // Perform the swap only if the indices are different
                    if (idx1 != idx2) {
                        // Swap real parts
                        double temp_real = data[idx1][0];
                        data[idx1][0] = data[idx2][0];
                        data[idx2][0] = temp_real;

                        // Swap imaginary parts
                        double temp_imag = data[idx1][1];
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
    cudaError_t octantFourierShiftFftwComplex(int Nx, int Ny, int Nz, fftw_complex* data) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        //size_t freeMem, totalMem;
        //cudaMemGetInfo(&freeMem, &totalMem);
        //std::cout << "Free memory: " << freeMem << " Total memory: " << totalMem << std::endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);


        cudaEventRecord(start);

        octantFourierShiftFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, data);
        cudaError_t errp = cudaPeekAtLastError();
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Octant(Fouriere)Shift in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return (errp == cudaSuccess) ? err : errp;
    }
    cudaError_t octantFourierShiftCufftComplex(int Nx, int Ny, int Nz, cufftComplex* data) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        //size_t freeMem, totalMem;
        //cudaMemGetInfo(&freeMem, &totalMem);
        //std::cout << "Free memory: " << freeMem << " Total memory: " << totalMem << std::endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);


        cudaEventRecord(start);

        octantFourierShiftCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(Nx, Ny, Nz, data);
        cudaError_t errp = cudaPeekAtLastError();
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Octant(Fouriere)Shift in CUDA (cufftComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return (errp == cudaSuccess) ? err : errp;
    }
    cudaError_t padFftwMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, fftw_complex* oldMat, fftw_complex* newMat)
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
#pragma omp parallel for
        for (int i = 0; i < newNx * newNy * newNz; ++i) {
            newMat[i][0] = 0.0; // Realteil
            newMat[i][1] = 0.0; // Imaginärteil
        }

        // Kopiere die Werte der alten Matrix in die Mitte der neuen Matrix
#pragma omp parallel for
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

        DEBUG_LOG("[TIME]["<<time/1000000<<" ms] padded FftwComplex Mat in CPP");
        
        return cudaSuccess;
    }
    cudaError_t padCufftMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, cufftComplex* d_oldMat, cufftComplex* d_newMat)
    {
        // Sicherheitsprüfung: Neue Dimensionen müssen größer oder gleich den alten sein
        if (newNx < oldNx || newNy < oldNy || newNz < oldNz) {
            return cudaErrorInvalidValue;
        }

        // Offset berechnen
        int offsetX = (newNx - oldNx) / 2;
        int offsetY = (newNy - oldNy) / 2;
        int offsetZ = (newNz - oldNz) / 2;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Block- und Grid-Dimensionen festlegen
        dim3 blockDim(16, 8, 8); // Größe der Blöcke
        dim3 gridDim(
            (newNx + blockDim.x - 1) / blockDim.x,
            (newNy + blockDim.y - 1) / blockDim.y,
            (newNz + blockDim.z - 1) / blockDim.z);

        cudaEventRecord(start);
        padCufftMatGlobal<<<gridDim, blockDim>>>(
            oldNx, oldNy, oldNz,
            newNx, newNy, newNz,
            d_oldMat, d_newMat,
            offsetX, offsetY, offsetZ);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] padded CufftComplex Mat in CUDA ("<<blockDim.x*blockDim.y*blockDim.z<<"x"<<gridDim.x*gridDim.y*gridDim.z<<")");
        
        return err;
    }
    cudaError_t normalizeFftwComplexData(int Nx, int Ny, int Nz, fftw_complex* d_data) {
        if (!d_data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 1024;
        int num_blocks = (num_elements + block_size - 1) / block_size;

        cudaEventRecord(start);

        normalizeFftwComplexDataGlobal<<<num_blocks, block_size>>>(Nx, Ny, Nz, d_data);
        cudaError_t errp = cudaPeekAtLastError();
        
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Normalizing FFT data in CUDA (fftw_complex) ("<<block_size*num_blocks<< " Threads)");
        
        return (errp == cudaSuccess) ? err : errp;
    }
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
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
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms] Device kernel(s) finished ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")");
        
        return err;
    }
}






