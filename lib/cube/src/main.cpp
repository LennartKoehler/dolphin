#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <utl.h>
#include <fftw3.h>
#include <operations.h>
#include <cstring>
#include <cufft.h>
#include <kernels.h>

//CUDA Boost Engine - CUBE
int main() {

    printDeviceProperties();

    // NxNxN matrix
    const int N = 262;
    // Memory size of matix
    int matrixSize = N * N * N * sizeof(fftw_complex);

    // Host-Arrays initialization
    fftw_complex *h_a = (fftw_complex*) fftw_malloc(matrixSize);;
    fftw_complex *h_b = (fftw_complex*) fftw_malloc(matrixSize);;
    fftw_complex *h_c = (fftw_complex*) fftw_malloc(matrixSize);;
    fftw_complex *h_d = (fftw_complex*) fftw_malloc(matrixSize);;
    // Creates a matrix filled with 2+i0
    createFftwRandomMat(N, h_a);
    createFftwRandomMat(N, h_b);
    createFftwUniformMat(N, h_c);

    // Init values in matrix
    checkUniformity(h_a, N);
    checkUniformity(h_b, N);
    checkUniformity(h_c, N);


    /*
    // Fourier Shift Test
        fftw_complex *h_c2 = (fftw_complex*) fftw_malloc(matrixSize);;
        std::memcpy(h_c2, h_c,  matrixSize);
        // Device-Pointer
        cufftComplex *d_c;

        // Allocate memory on GPU
        cudaMalloc((void**)&d_c, matrixSize);

        convertFftwToCufftComplexOnDevice(h_c, d_c, N);

        printFirstElem(h_c);
        printFirstElem(h_c2);

        octantFourierShiftCufftComplex(N, d_c);
        octantFourierShiftCufftComplex(N, d_c);
        convertCufftToFftwComplexOnHost(h_c, d_c, N);

        checkOctantFourierShift(N, h_c2, h_c);
        printFirstElem(h_c);
        printFirstElem(h_c2);
        octantFourierShiftFftwComplex(N, h_c);
        displayHeatmap(h_c, N);
    /*

    /*
    // Comparison MatMul on CPU and GPU
        complexMatMultCpp(N, h_a, h_b, h_c);
        complexMatMulCppOmp(N, h_a, h_b, h_c);

        // Device-Pointer
        fftw_complex *d_a, *d_b, *d_c;
        // Allocate memory on GPU
        cudaMalloc((void**)&d_a, matrixSize);
        cudaMalloc((void**)&d_b, matrixSize);
        cudaMalloc((void**)&d_c, matrixSize);

        // Copy Data from CPU to GPU
        copyDataFromHostToDevice(d_a, h_a, matrixSize);
        copyDataFromHostToDevice(d_b, h_b, matrixSize);

        complexMatMulFftwComplex(N, d_a, d_b, d_c, "cuda");
        // Copy Data back from GPU to CPU
        copyDataFromDeviceToHost(h_c, d_c, matrixSize);

        checkUniformity(h_c, N);
    */


    // This will be part of RL in DeconvTool in the interation loop
    printFirstElem(h_a);
    printFirstElem(h_b);

    //printSpecificElem(h_c, 68644);
        cufftComplex *d_cua, *d_cub, *d_cuc, *d_cud;
        cudaMalloc((void**)&d_cua, N * N * N * sizeof(cufftComplex));
        cudaMalloc((void**)&d_cub, N * N * N * sizeof(cufftComplex));
        cudaMalloc((void**)&d_cuc, N * N * N * sizeof(cufftComplex));
        cudaMalloc((void**)&d_cud, N * N * N * sizeof(cufftComplex));
        convertFftwToCufftComplexOnDevice(h_a, d_cua, N);
        convertFftwToCufftComplexOnDevice(h_b, d_cub, N);
        convertFftwToCufftComplexOnDevice(h_c, d_cuc, N);
        convertFftwToCufftComplexOnDevice(h_d, d_cud, N);

        // All opertions within the iterations loop
        //complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);
        //deviceTestKernel(N, d_cua, d_cub, d_cuc);

        // Create FFT-Plan for reuse
        cufftHandle plan;
        cufftResult result = cufftPlan3d(&plan, N, N, N, CUFFT_C2C);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "[ERROR] error while creating FFT-Plan: " << result << std::endl;
        }

        // Calculate FFT
        //cufftForward(d_cuc, d_cuc, plan);
    //calculateLaplacianCufftComplex(N, d_cua, d_cua);
    complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc, 1e-6);
    complexElementwiseMatDivNaiveCufftComplex(N, d_cua, d_cub, d_cud);

       // cufftInverse(d_cuc,d_cuc, plan, N);
        //gradZCufftComplex(N, d_cuc, d_cuc);
    //complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);
        convertCufftToFftwComplexOnHost(h_c, d_cuc, N);
    convertCufftToFftwComplexOnHost(h_d, d_cud, N);

        //checkUniformity(h_c, N);
        printFirstElem(h_c);
        printFirstElem(h_d);

        // Free memory
        cufftDestroy(plan);
        cudaFree(d_cuc);
        free(h_c);


    return 0;
}
