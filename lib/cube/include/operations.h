#pragma once
#include <cuComplex.h>
#include <cufftw.h>
#include <cufft.h>

namespace CUBE_MAT {
    // Normal Matrix Multiplication
    void complexMatMulFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    void complexMatMulFftwComplexOmpCPU(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    void complexMatMulFftwComplexCUDA(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    void complexMatMulCuComplexCUDA(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    void complexMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, const char* type);

    // Elementwise Matrix Multiplication/Division (always GPU)
    void complexElementwiseMatMulCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    void complexElementwiseMatMulCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    void complexElementwiseMatMulFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    void complexElementwiseMatMulConjugateCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    void complexElementwiseMatMulConjugateFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C);
    void complexElementwiseMatDivCuComplex(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
    void complexElementwiseMatDivNaiveCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C);
    void complexElementwiseMatDivCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
    void complexElementwiseMatDivFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
    void complexElementwiseMatDivStabilizedCufftComplex(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon);
    void complexElementwiseMatDivStabilizedFftwComplex(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon);
}

namespace CUBE_REG {
    // Regularization
    void calculateLaplacianCufftComplex(int Nx, int Ny, int Nz, cufftComplex* psf, cufftComplex* laplacian_fft);
    void gradXCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX);
    void gradYCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY);
    void gradZCufftComplex(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ);
    void computeTVCufftComplex(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv);
    void normalizeTVCufftComplex(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon);
    void calculateLaplacianFftwComplex(int Nx, int Ny, int Nz, fftw_complex* psf, fftw_complex* laplacian_fft);
    void gradXFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradX);
    void gradYFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradY);
    void gradZFftwComplex(int Nx, int Ny, int Nz, fftw_complex* image, fftw_complex* gradZ);
    void computeTVFftwComplex(int Nx, int Ny, int Nz, double lambda, fftw_complex *gx, fftw_complex *gy, fftw_complex *gz, fftw_complex *tv);
    void normalizeTVFftwComplex(int Nx, int Ny, int Nz, fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, double epsilon);
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    void calculateLaplacianCufftComplexTiled(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft);
}

namespace CUBE_FTT {
    // FFT
    void cufftForward(cufftComplex* input, cufftComplex* output, cufftHandle plan);
    void cufftInverse(int Nx, int Ny, int Nz, cufftComplex* input, cufftComplex* output, cufftHandle plan);

    // Fourier Shift, Padding and Normalization
    void octantFourierShiftFftwComplexCPU(int Nx, int Ny, int Nz, fftw_complex* data);
    void octantFourierShiftFftwComplex(int Nx, int Ny, int Nz, fftw_complex* data);
    void octantFourierShiftCufftComplex(int Nx, int Ny, int Nz, cufftComplex* data);
    void padFftwMat(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, fftw_complex* oldMat, fftw_complex* newMat);
    void padCufftMat(int oldNx, int oldNy, int oldNz,int newNx, int newNy, int newNz, cufftComplex* d_oldMat, cufftComplex* d_newMat);
    void normalizeFftwComplexData(int Nx, int Ny, int Nz, fftw_complex* d_data);
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    void deviceTestKernel(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C);
}



