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

#include <cstring> // Für memset



int main() {

    //printDeviceProperties();

    // NxNxN matrix
    const int N = 100;
    // Memory size of matix


    // Host-Arrays initialization
    fftw_complex *h_a = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));;
    fftw_complex *h_b = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));;
    fftw_complex *h_c = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));;
    fftw_complex *h_d = (fftw_complex*) fftw_malloc(2*N * 2*N * 2*N * sizeof(fftw_complex));;
    // Creates a matrix filled with 2+i0
    createFftwUniformMat(N,N,N, h_a);
    createFftwUniformMat(N,N,N, h_b);
    createFftwSphereMat(N,N,N, h_c);
    // Init values in matrix
    checkUniformity(N,N,N,h_a);
    checkUniformity(N,N,N,h_b);
    checkUniformity(N,N,N,h_c);
    //printSpecificElem(55050, h_c);

    padFftwMat(N,N,N, 200, 200, 200, h_c, h_d);
   // displayHeatmap(400, 200, 200, h_d);
    /*
    // Fourier Shift Test
        fftw_complex *h_c2 = (fftw_complex*) fftw_malloc(N * N * N * sizeof(fftw_complex));;
        std::memcpy(h_c2, h_c,  N * N * N * sizeof(fftw_complex));
        // Device-Pointer
        cufftComplex *d_c;

        // Allocate memory on GPU
        cudaMalloc((void**)&d_c, N * N * N * sizeof(cufftComplex));

        convertFftwToCufftComplexOnDevice(N,N, N,h_c, d_c);

        printSpecificElem(55050, h_c);
        printSpecificElem(55050, h_c2);

        //octantFourierShiftCufftComplex(N,N,N, d_c);
        //octantFourierShiftCufftComplex(N,N,N, d_c);
        complexElementwiseMatMulCufftComplex(N, N, N, d_c, d_c, d_c);

        convertCufftToFftwComplexOnHost(N,N,N, h_c, d_c);

        checkOctantFourierShift(N,N,N, h_c2, h_c);
        printSpecificElem(55050, h_c);
        printSpecificElem(55050, h_c2);

        //octantFourierShiftFftwComplex(N,N,N, h_c);
        displayHeatmap(N,N,N, h_c);

*/
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
    printFirstElem(h_c);

    //printSpecificElem(h_c, 68644);
        cufftComplex *d_cua, *d_cub, *d_cuc, *d_cud;
       // cudaMalloc((void**)&d_cua, N * N * N * sizeof(cufftComplex));
       // cudaMalloc((void**)&d_cub, N * N * N * sizeof(cufftComplex));
        cudaMalloc((void**)&d_cuc, N * N * N * sizeof(cufftComplex));
        cudaMalloc((void**)&d_cud, 2*N * 2*N * 2*N * sizeof(cufftComplex));
      //  convertFftwToCufftComplexOnDevice(N,N,N,h_a, d_cua);
       // convertFftwToCufftComplexOnDevice(N,N,N,h_b, d_cub);
        convertFftwToCufftComplexOnDevice(N,N,N,h_c, d_cuc);
        //convertFftwToCufftComplexOnDevice(N,N,N,h_d, d_cud);
padCufftMat(N,N,N,2*N,2*N,2*N, d_cuc, d_cud);
        // All opertions within the iterations loop
        //complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);
        //deviceTestKernel(N, d_cua, d_cub, d_cuc);

        // Create FFT-Plan for reuse
      //  cufftHandle plan;
      //  cufftResult result = cufftPlan3d(&plan, N, N, N, CUFFT_C2C);
        //if (result != CUFFT_SUCCESS) {
//std::cerr << "[ERROR] error while creating FFT-Plan: " << result << std::endl;
      //  }

        // Calculate FFT
       // cufftForward(d_cuc, d_cuc, plan);
    //calculateLaplacianCufftComplex(N, d_cua, d_cua);
    //complexElementwiseMatDivCufftComplex(N,N,N, d_cua, d_cub, d_cuc, 1e-6);
    //complexElementwiseMatDivNaiveCufftComplex(N,N,N, d_cua, d_cub, d_cud);

        //cufftInverse(N,N,N,d_cuc,d_cuc, plan);
       // gradZCufftComplex(N,N,N, d_cuc, d_cuc);
    //complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);
        convertCufftToFftwComplexOnHost(2*N,2*N,2*N,h_d, d_cud);
    //convertCufftToFftwComplexOnHost(N,N,N,h_d, d_cud);

        //checkUniformity(h_c, N);
        printFirstElem(h_c);
        printFirstElem(h_d);
        displayHeatmap(2*N, 2*N, 2*N, h_d);
        // Free memory
        //cufftDestroy(plan);
        cudaFree(d_cuc);
        free(h_c);


    return 0;
}
