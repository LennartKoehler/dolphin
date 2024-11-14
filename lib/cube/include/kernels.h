#pragma once
#include <cuComplex.h>
#include <cufft.h>
#include <fftw3.h>

// Conversions
__global__ void fftwToCuComplexKernelGlobal(cuComplex* cuArr, fftw_complex* fftwArr, int N);
__global__ void cuToFftwComplexKernelGlobal(fftw_complex* fftwArr, cuComplex* cuArr, int N);
__global__ void fftwToCufftComplexKernelGlobal(cufftComplex* cufftArr, fftw_complex* fftwArr, int N);
__global__ void cufftToFftwComplexKernelGlobal(fftw_complex* fftwArr, cufftComplex* cufftArr, int N);

// Mat operations
__global__ void complexMatMulFftwComplexGlobal(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
__global__ void complexMatMulCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatMulCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatDivCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__global__ void complexElementwiseMatMulCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatMulConjugateCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatDivCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__global__ void complexElementwiseMatDivNaiveCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__global__ void complexElementwiseMatDivStabilizedCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);


// Regularization
__global__ void calculateLaplacianCufftComplexGlobal(int N, cufftComplex* Afft, cufftComplex* laplacianfft);
__global__ void gradientXCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradX);
__global__ void gradientYCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradX);
__global__ void gradientZCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradX);
__global__ void computeTVCufftComplexGlobal(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
__global__ void normalizeTVCufftComplexGlobal(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);

// Tiled
__global__ void calculateLaplacianCufftComplexTiledGlobal(int N, cufftComplex* Afft, cufftComplex* laplacianfft);

// Fourier Shift
__global__ void normalizeComplexData(cufftComplex* d_data, int N);
__global__ void octantFourierShiftCufftComplexGlobal(int N, cufftComplex* data);

// Device Kernels
__global__ void deviceTestKernelGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexMatMulCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatMulCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatDivCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C);
__device__ void complexElementwiseMatMulCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatMulConjugateCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatDivCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__device__ void complexElementwiseMatDivNaiveCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
__device__ void complexElementwiseMatDivStabilizedCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
__device__ void calculateLaplacianCufftComplexDevice(int N, cufftComplex* Afft, cufftComplex* laplacianfft);
__device__ void gradientXCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradX);
__device__ void gradientYCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradY);
__device__ void gradientZCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradZ);
__device__ void computeTVCufftComplexDevice(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
__device__ void normalizeTVCufftComplexDevice(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);
__device__ void calculateLaplacianCufftComplexTiledDevice(int N, cufftComplex* Afft, cufftComplex* laplacianfft);