#pragma once
#include <cuComplex.h>
#include <fftw3.h>
#include <iostream>
#include <omp.h>
#include <cufft.h>

// Runs on naive CPP (CPU), CPP with OMP (CPU) or CUDA (GPU)
void complexMatMultCpp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulCppOmp(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulCudaFftwComplex(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulFftwComplex(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type);

// Runs always on CUDA (GPU)
void complexMatMulCudaCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatMulCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatDivCuComplex(int N, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatMulCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatMulConjugateCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatDivCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
void complexElementwiseMatDivNaiveCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatDivStabilizedCufftComplex(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);

// Regularization
void calculateLaplacianCufftComplex(int N, cufftComplex* psf, cufftComplex* laplacian_fft);
void gradXCufftComplex(int N, cufftComplex* image, cufftComplex* gradX);
void gradYCufftComplex(int N, cufftComplex* image, cufftComplex* gradY);
void gradZCufftComplex(int N, cufftComplex* image, cufftComplex* gradZ);
void computeTVCufftComplex(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
void normalizeTVCufftComplex(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);

// Tiled
void calculateLaplacianCufftComplexTiled(int N, cufftComplex* Afft, cufftComplex* laplacianfft);

// FFT
void cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan);
void cufftInverse(cufftComplex* input, cufftComplex* output, cufftHandle plan, int N);

// Fourier Shift (fftw on CPU and cufft on GPU)
void octantFourierShiftFftwComplex(int N, fftw_complex* data);
void octantFourierShiftCufftComplex(int N, cufftComplex* data);

// Testing __device__ kernels
void deviceTestKernel(int N, cuComplex* A, cuComplex* B, cuComplex* C);



