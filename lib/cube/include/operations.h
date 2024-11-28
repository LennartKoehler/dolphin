#pragma once
#include <cuComplex.h>
#include <cufftw.h>
#include <iostream>
#include <omp.h>
#include <cufft.h>

// Runs on naive CPP (CPU), CPP with OMP (CPU) or CUDA (GPU)
void complexMatMulCpp(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulCppOmp(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulCudaFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type);

// Runs always on CUDA (GPU)
void complexMatMulCudaCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatMulCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatDivCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
void complexElementwiseMatMulCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatMulConjugateCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatDivCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
void complexElementwiseMatDivNaiveCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
void complexElementwiseMatDivStabilizedCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);

// New Fftw functions
void complexElementwiseMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexElementwiseMatMulConjugateFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
void complexElementwiseMatDivStabilizedFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
void octantFourierShiftFftwComplex(int Nx, int Ny, int Nz, fftw_complex* data);
void normalizeFftwComplexData(int Nx, int Ny, int Nz, fftw_complex* d_data);


// Regularization
void calculateLaplacianCufftComplex(int Nx, int Ny, int Nz, cufftComplex* psf, cufftComplex* laplacian_fft);
void gradXCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
void gradYCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY);
void gradZCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ);
void computeTVCufftComplex(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
void normalizeTVCufftComplex(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);

// Tiled
void calculateLaplacianCufftComplexTiled(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);

// FFT
void cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan);
void cufftInverse(int Nx, int Ny, int Nz, cufftComplex* input, cufftComplex* output, cufftHandle plan);

// Fourier Shift (fftw on CPU and cufft on GPU)
void octantFourierShiftFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* data);
void octantFourierShiftCufftComplex(int Nx, int Ny, int Nz, cufftComplex* data);
void padFftwMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, fftw_complex* oldMat, fftw_complex* newMat);
void padCufftMat(int oldNx, int oldNy, int oldNz,int newNx, int newNy, int newNz, cufftComplex* d_oldMat, cufftComplex* d_newMat);

// Testing __device__ kernels
void deviceTestKernel(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);



