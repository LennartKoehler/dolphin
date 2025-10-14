#pragma once
#include <cuComplex.h>
#include <cufftw.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

namespace CUBE_MAT {
    // Normal Matrix Multiplication
    cudaError_t complexMatMulFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    cudaError_t complexMatMulFftwComplexOmpCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    cudaError_t complexMatMulFftwComplexCUDA(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    cudaError_t complexMatMulCuComplexCUDA(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    cudaError_t complexMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type);

    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMulCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    cudaError_t complexElementwiseMatMulCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    cudaError_t complexElementwiseMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    cudaError_t complexElementwiseMatMulConjugateCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    cudaError_t complexElementwiseMatMulConjugateFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    cudaError_t complexElementwiseMatDivCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    cudaError_t complexElementwiseMatDivNaiveCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    cudaError_t complexElementwiseMatDivCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
    cudaError_t complexElementwiseMatDivFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
    cudaError_t complexElementwiseMatDivStabilizedCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
    cudaError_t complexElementwiseMatDivStabilizedFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacianCufftComplex(int Nx, int Ny, int Nz, cufftComplex* psf, cufftComplex* laplacian_fft);
    cudaError_t gradXCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
    cudaError_t gradYCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY);
    cudaError_t gradZCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ);
    cudaError_t computeTVCufftComplex(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
    cudaError_t normalizeTVCufftComplex(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);
    cudaError_t calculateLaplacianFftwComplex(int Nx, int Ny, int Nz, fftw_complex* psf, fftw_complex* laplacian_fft);
    cudaError_t gradXFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX);
    cudaError_t gradYFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradY);
    cudaError_t gradZFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradZ);
    cudaError_t computeTVFftwComplex(int Nx, int Ny, int Nz, double lambda, fftw_complex *gx, fftw_complex *gy, fftw_complex *gz, fftw_complex *tv);
    cudaError_t normalizeTVFftwComplex(int Nx, int Ny, int Nz, fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, double epsilon);
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianCufftComplexTiled(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);
}

namespace CUBE_FTT {
    // FFT
    cudaError_t cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan);
    cudaError_t cufftInverse(int Nx, int Ny, int Nz, cufftComplex* input, cufftComplex* output, cufftHandle plan);

    // Fourier Shift, Padding and Normalization
    cudaError_t octantFourierShiftFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* data);
    cudaError_t octantFourierShiftFftwComplex(int Nx, int Ny, int Nz, fftw_complex* data);
    cudaError_t octantFourierShiftCufftComplex(int Nx, int Ny, int Nz, cufftComplex* data);
    cudaError_t padFftwMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, fftw_complex* oldMat, fftw_complex* newMat);
    cudaError_t padCufftMat(int oldNx, int oldNy, int oldNz,int newNx, int newNy, int newNz, cufftComplex* d_oldMat, cufftComplex* d_newMat);
    cudaError_t normalizeFftwComplexData(int Nx, int Ny, int Nz, fftw_complex* d_data);
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
}



