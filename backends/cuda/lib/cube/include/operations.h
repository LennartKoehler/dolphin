#pragma once
#include "CUBE.h"


namespace CUBE_MAT {

    cudaError_t complexMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatMulConjugate(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream = 0);
    cudaError_t complexElementwiseMatDivStabilized(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream = 0);
    cudaError_t complexScalarMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t scalarReal, real_t scalarImag, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexAddition(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream = 0);
    cudaError_t complexAddition(complex_t** data, complex_t* sums, size_t nImages, size_t imageVolume, cudaStream_t stream = 0);

    cudaError_t elementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon, cudaStream_t stream = 0);
    cudaError_t scalarMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t scalar , real_t* C, cudaStream_t stream = 0);
    cudaError_t elementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B , real_t* C, cudaStream_t stream = 0);

    cudaError_t sum(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result, cudaStream_t stream = 0);
    cudaError_t meanSquareError(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result, cudaStream_t stream = 0);
    cudaError_t sumToOne(real_t** data, size_t nImages, size_t imageVolume, cudaStream_t stream = 0);
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(size_t Nx, size_t Ny, size_t Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream = 0);
    cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradX, cudaStream_t stream = 0);
    cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY, cudaStream_t stream = 0);
    cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream = 0);
    cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv, cudaStream_t stream = 0);
    cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t beta, cudaStream_t stream = 0);
    // Gradient functions for real-valued data
    cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, cudaStream_t stream = 0);
    cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY, cudaStream_t stream = 0);
    cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ, cudaStream_t stream = 0);
    // Combined gradient (computes all three gradients in a single pass)
    cudaError_t grad(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ, cudaStream_t stream = 0);
    // Divergence (backward differences — adjoint of forward gradient)
    cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result, cudaStream_t stream = 0);
    cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result, cudaStream_t stream = 0);
    // computeTV for real-valued divergence
    cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv, cudaStream_t stream = 0);
    cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t beta, cudaStream_t stream = 0);
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft);
}

namespace CUBE_FTT {
    // Fourier Shift, Padding and Normalization
    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, complex_t* data, cudaStream_t stream = 0);
    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, size_t stride, real_t* data, cudaStream_t stream = 0);
    // cudaError_t padMat(size_t oldNx, size_t oldNy, size_t oldNz, size_t newNx, size_t newNy, size_t newNz, complex_t* oldMat, complex_t* newMat);
    cudaError_t normalizeData(size_t Nx, size_t Ny, size_t Nz, complex_t* d_data, cudaStream_t stream = 0);
}

namespace CUBE_DEVICE_KERNEL {
    // Testing __device__ kernels
    cudaError_t deviceTestKernel(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C);
}


