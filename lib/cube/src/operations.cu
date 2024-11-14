#include <cuda_runtime_api.h>
#include <operations.h>
#include <kernels.h>
#include <thread>

// Runs on naive CPP (CPU), CPP with OMP (CPU) or CUDA (GPU)
void complexMatMultCpp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    double start = omp_get_wtime();
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < N; ++z) {
                int index = x * N * N + y * N + z;
                fftw_complex sum = {0.0, 0.0};

                for (int k = 0; k < N; ++k) {
                    int indexA = x * N * N + y * N + k;
                    int indexB = k * N * N + y * N + z;

                    float realA = A[indexA][0];
                    float imagA = A[indexA][1];
                    float realB = B[indexB][0];
                    float imagB = B[indexB][1];

                    sum[0] += realA * realB - imagA * imagB;  // Realteil
                    sum[1] += realA * imagB + imagA * realB;  // Imaginärteil
                }

                C[index][0] = sum[0];
                C[index][1] = sum[1];
            }
        }
    }
    double end = omp_get_wtime();
    std::cout << "[TIME]["<< (end - start)*1000 << " ms" <<"] MatMul in Cpp" << std::endl;
}
void complexMatMulCppOmp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    double start = omp_get_wtime();
#pragma omp parallel for collapse(3) schedule(static)
    for (int x = 0; x < N; ++x) {
        for (int y = 0; y < N; ++y) {
            for (int z = 0; z < N; ++z) {
                int index = x * N * N + y * N + z;
                fftw_complex sum = {0.0, 0.0};

                for (int k = 0; k < N; ++k) {
                    int indexA = x * N * N + y * N + k;
                    int indexB = k * N * N + y * N + z;

                    float realA = A[indexA][0];
                    float imagA = A[indexA][1];
                    float realB = B[indexB][0];
                    float imagB = B[indexB][1];

                    sum[0] += realA * realB - imagA * imagB;  // real
                    sum[1] += realA * imagB + imagA * realB;  // img
                }

                C[index][0] = sum[0];
                C[index][1] = sum[1];
            }
        }
    }
    double end = omp_get_wtime();
    std::cout << "[TIME]["<< (end - start)*1000 << " ms" <<"] MatMul in Cpp with Omp ("<< omp_get_max_threads() <<" Threads)" << std::endl;
}
void complexMatMulCudaFftwComplex(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);


    cudaEventRecord(start);

    complexMatMulFftwComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] MatMul in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexMatMulFftwComplex(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type) {
    if (strcmp(type, "cpp") == 0) {
        complexMatMultCpp(N, A, B, C);
    }else if (strcmp(type, "omp") == 0) {
        complexMatMulCppOmp(N, A, B, C);
    }else if (strcmp(type, "cuda") == 0) {
        complexMatMulCudaFftwComplex(N, A, B, C);
    }
    else {
        fprintf(stderr, "Unknown operation type %s\n", type);
    }
}


// Runs always on CUDA (GPU)
void complexMatMulCudaCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexMatMulCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] MatMul in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatMulCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatMulCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatDivCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatDivCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatMulCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatMulCuComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise MatMul in CUDA (cufftComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatMulConjugateCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C)  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatMulConjugateCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise MatMul conjugated in CUDA (cufftComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatDivCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatDivCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatDivNaiveCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatDivNaiveCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise naive MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void complexElementwiseMatDivStabilizedCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    complexElementwiseMatDivStabilizedCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] elementwise stabilized MatDiv in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}


// Regularization
void calculateLaplacianCufftComplex(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    calculateLaplacianCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, Afft, laplacianfft);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void gradXCufftComplex(int N, cufftComplex* image, cufftComplex* gradX) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    gradientXCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, image, gradX);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating GradientX in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void gradYCufftComplex(int N, cufftComplex* image, cufftComplex* gradY) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    gradientYCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, image, gradY);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating GradientY in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void gradZCufftComplex(int N, cufftComplex* image, cufftComplex* gradZ) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    gradientZCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, image, gradZ);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating GradientZ in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void computeTVCufftComplex(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    computeTVCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, lambda, gx, gy, gz, tv);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}
void normalizeTVCufftComplex(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    normalizeTVCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, gradX, gradY, gradZ, epsilon);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] normalizing Total Variation in CUDA (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}


// Tiled
void calculateLaplacianCufftComplexTiled(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    calculateLaplacianCufftComplexTiledGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, Afft, laplacianfft);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] calculating Laplacian in CUDA tiled with shared mem (cuComplex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}


// FFT
void cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cufftResult result;

    cudaEventRecord(start);

    result = cufftExecC2C(plan, input, output, CUFFT_FORWARD);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (result != CUFFT_SUCCESS) {
        std::cerr << "[ERROR] forward fft: " << result << std::endl;
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] Forward FFT in cuFFT"<< std::endl;
}
void cufftInverse(cufftComplex* input, cufftComplex* output, cufftHandle plan, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cufftResult result;

    cudaEventRecord(start);

    result = cufftExecC2C(plan, input, output, CUFFT_INVERSE);
    int num_elements = N * N * N;  // Beispiel: Gesamtzahl der Elemente
    int block_size = 256;  // Blockgröße (kann angepasst werden)
    int num_blocks = (num_elements + block_size - 1) / block_size;  // Berechne die Anzahl der Blöcke

    normalizeComplexData<<<num_blocks, block_size>>>(output, num_elements);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (result != CUFFT_SUCCESS) {
        std::cerr << "[ERROR] inverse fft: " << result << std::endl;
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] Inverse FFT in cuFFT"<< std::endl;
}


// Fourier Shift (fftw on CPU and cufft on GPU)
void octantFourierShiftFftwComplex(int N, fftw_complex* data) {
    int width = N;
    int height = N;
    int depth = N;
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

    std::cout
 << "[TIME]["<<time/1000000<<" ms] Octant(Fourier)Shift in CPP"<< std::endl;
}
void octantFourierShiftCufftComplex(int N, cufftComplex* data) {
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "Free memory: " << freeMem << " Total memory: " << totalMem << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);


    cudaEventRecord(start);

    octantFourierShiftCufftComplexGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, data);
    cudaError_t errp = cudaPeekAtLastError();
    if (errp != cudaSuccess) {
        std::cerr << "Fourier shift kernel launch error: " << cudaGetErrorString(errp) << std::endl;
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] Octant(Fouriere)Shift in CUDA (fftw_complex) ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}


// Testing __device__ kernels
void deviceTestKernel(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
    dim3 threadsPerBlock(10, 10, 10); //=1000 (faster than max 1024)
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (N + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaEventRecord(start);

    deviceTestKernelGlobal<<<blocksPerGrid, threadsPerBlock>>>(N, A, B, C);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms] Device kernel(s) finished ("<<threadsPerBlock.x*threadsPerBlock.y*threadsPerBlock.z<<"x"<<blocksPerGrid.x*blocksPerGrid.y*blocksPerGrid.z<<")" << std::endl;
}





