#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <utl.h>
#include <cufftw.h>
#include <operations.h>
#include <cstring>
#include <cufft.h>
#include <kernels.h>
#include <omp.h>

//CUDA Boost Engine - CUBE

#include <cstring> // Für memset

using namespace CUBE_MAT;
using namespace CUBE_REG;
using namespace CUBE_FTT;
using namespace CUBE_TILED;
using namespace CUBE_DEVICE_KERNEL;
using namespace CUBE_UTL_COPY;
using namespace CUBE_UTL_CONVERT;
using namespace CUBE_UTL_INFO;
using namespace CUBE_UTL_CHECK;
using namespace CUBE_UTL_INIT_MAT;

int main() {

    //printDeviceProperties();

    // NxNxN matrix
    const int N = 500;
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

   // padFftwMat(N,N,N, 200, 200, 200, h_c, h_d);
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
        cudaError_t err = complexElementwiseMatMulCufftComplex(N, N, N, d_c, d_c, d_c);
        if (err != cudaSuccess) {
            std::cerr << "[ERROR] complexElementwiseMatMulCufftComplex failed: " << cudaGetErrorString(err) << std::endl;
        }

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
    */





/*
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
cudaError_t err = padCufftMat(N,N,N,2*N,2*N,2*N, d_cuc, d_cud);
if (err != cudaSuccess) {
    std::cerr << "[ERROR] padCufftMat failed: " << cudaGetErrorString(err) << std::endl;
}
        // All opertions within the iterations loop
        //complexElementwiseMatDivCufftComplex(N, d_cua, d_cub, d_cuc);
        //deviceTestKernel(N, d_cua, d_cub, d_cuc);

        // Create FFT-Plan for reuse
        //cufftHandle plan;
      //  cufftResult result = cufftPlan3d(&plan, N, N, N, CUFFT_C2C);
        //if (result != CUFFT_SUCCESS) {
//std::cerr << "[ERROR] error while creating FFT-Plan: " << result << std::endl;
      //  }

        // Calculate FFT
       // cufftForward(d_cuc, d_cuc, plan);
    //calculateLaplacianCufftComplex(N, d_cua, d_cua);
    //complexElementwiseMatDivCufftComplex(N,N,N, d_cua, d_cub, d_cuc, 1e-6);
    //complexElementwiseMatDivNaiveCufftComplex(N,N,N, d_cua, d_cub, d_cud);

        //cudaError_t err = cufftInverse(N,N,N,d_cuc,d_cuc, plan);
        //if (err != cudaSuccess) {
        //    std::cerr << "[ERROR] cufftInverse failed: " << cudaGetErrorString(err) << std::endl;
        //}
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

*/
    const int Nx = 500, Ny = 500, Nz = 500;

    // Host-Arrays
    fftw_complex* h_input = (fftw_complex*)malloc(sizeof(fftw_complex) * Nx * Ny * Nz);
    fftw_complex* h_output = (fftw_complex*)malloc(sizeof(fftw_complex) * Nx * Ny * Nz);

    // GPU-Arrays
    fftw_complex *d_input, *d_output;
    cudaMalloc((void**)&d_input, sizeof(fftw_complex) * Nx * Ny * Nz);
    cudaMalloc((void**)&d_output, sizeof(fftw_complex) * Nx * Ny * Nz);

    // Beispielinitialisierung der Eingabedaten
    for (int i = 0; i < Nx * Ny * Nz; i++) {
        h_input[i][0] = static_cast<float>(i);  // Realteil
        h_input[i][1] = 0.0f;                   // Imaginärteil
    }

    // Kopieren der Eingabedaten auf die GPU
    cudaMemcpy(d_input, h_input, sizeof(fftw_complex) * Nx * Ny * Nz, cudaMemcpyHostToDevice);
    double start = omp_get_wtime();

    // FFTW-Plan erstellen
    fftw_plan plan = fftw_plan_dft_3d(Nx, Ny, Nz, d_input, d_output, FFTW_FORWARD, FFTW_MEASURE);

    // FFT ausführen
    fftw_execute(plan);
    double end = omp_get_wtime();
    std::cout << "[TIME][" << (end - start) * 1000 << " Device ms]" << std::endl;;
    // Ergebnisse zurück auf den Host kopieren
    cudaMemcpy(h_output, d_output, sizeof(fftw_complex) * Nx * Ny * Nz, cudaMemcpyDeviceToHost);
    double start2 = omp_get_wtime();

    // FFTW-Plan erstellen
    fftw_plan plan2 = fftw_plan_dft_3d(Nx, Ny, Nz, h_input, h_output, FFTW_FORWARD, FFTW_MEASURE);

    // FFT ausführen
    fftw_execute(plan2);
    double end2 = omp_get_wtime();
    std::cout << "[TIME][" << (end2 - start2) * 1000 << " Host ms]" << std::endl;;
    // Ergebnisse anzeigen (z.B. für die ersten 10 Werte)
    for (int i = 0; i < 10; i++) {
        std::cout << "Output[" << i << "]: (" << h_output[i][0] << ", " << h_output[i][1] << ")\n";
    }

    // Ressourcen freigeben


    // Create FFT-Plan for reuse
    cufftComplex *d;
     cudaMalloc((void**)&d, N * N * N * sizeof(cufftComplex));
    convertFftwToCufftComplexOnDevice(N,N,N, h_input, d);
    double start3 = omp_get_wtime();

    cufftHandle plan3;
      cufftResult result = cufftPlan3d(&plan3, N, N, N, CUFFT_C2C);
    if (result != CUFFT_SUCCESS) {
    std::cerr << "[ERROR] error while creating FFT-Plan: " << result << std::endl;
      }
    cudaError_t err;
    // Calculate FFT
    //  cudaError_t err = cufftForward(d, d, plan3);
    //  if (err != cudaSuccess) {
    //      std::cerr << "[ERROR] cufftForward failed: " << cudaGetErrorString(err) << std::endl;
    //  }
    // double end3 = omp_get_wtime();
    // std::cout << "[TIME][" << (end3 - start3) * 1000 << " cuFFT ms]" << std::endl;;
    // Device-Pointer
    fftw_complex *d_a, *d_b, *d_c;
    // Allocate memory on GPU
    cudaMalloc((void**)&d_a, sizeof(fftw_complex)*N*N*N);;
    cudaMalloc((void**)&d_b, sizeof(fftw_complex)*N*N*N);
    cudaMalloc((void**)&d_c, sizeof(fftw_complex)*N*N*N);

    convertFftwToCufftComplexOnDevice(N,N,N, h_input, d);
    // Copy Data from CPU to GPU
    copyDataFromHostToDevice(N,N,N,d_a, h_a);
    copyDataFromHostToDevice(N,N,N,d_b, h_b);

    err = complexMatMulFftwComplex(N,N,N, d_a, d_b, d_a, "cuda");
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] complexMatMulFftwComplex failed: " << cudaGetErrorString(err) << std::endl;
    }
    // Copy Data back from GPU to CPU
    copyDataFromDeviceToHost(N,N,N,h_c, d_c);
    err = complexElementwiseMatMulCufftComplex(N,N,N, d, d, d);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] complexElementwiseMatMulCufftComplex failed: " << cudaGetErrorString(err) << std::endl;
    }

    cudaFree(d_output);
    return 0;
}
