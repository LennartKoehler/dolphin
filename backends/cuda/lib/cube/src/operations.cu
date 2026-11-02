/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/
#include <cuda_runtime_api.h>
#include "operations.h"
#include "kernels.h"
#include "launch.h"



// Global CUDA kernel configuration
// Global thread block configuration
const dim3 GLOBAL_THREADS_PER_BLOCK(4, 8, 8);
const int GLOBAL_THREADS_PER_BLOCK_1D = 256;

// Helper function to compute blocks per grid
inline dim3 computeBlocksPerGrid(size_t Nx, size_t Ny, size_t Nz) {
    return dim3(
        static_cast<unsigned int>((Nx + GLOBAL_THREADS_PER_BLOCK.x - 1) / GLOBAL_THREADS_PER_BLOCK.x),
        static_cast<unsigned int>((Ny + GLOBAL_THREADS_PER_BLOCK.y - 1) / GLOBAL_THREADS_PER_BLOCK.y),
        static_cast<unsigned int>((Nz + GLOBAL_THREADS_PER_BLOCK.z - 1) / GLOBAL_THREADS_PER_BLOCK.z)
    );
}


namespace CUBE_MAT {


cudaError_t elementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { elementwiseMatDivGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideB, strideC, A, B, C, epsilon); }, stream);
    }

cudaError_t scalarMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t B, real_t* C, cudaStream_t stream) {
        if (!A || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { scalarMulGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideC, A, B, C); }, stream);
    }

cudaError_t elementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { elementwiseMatMulGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideB, strideC, A, B, C); }, stream);
    }

    cudaError_t complexMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexMatMulGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C); }, stream);
    }

    cudaError_t complexScalarMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t scalarReal, real_t scalarImag, complex_t* C, cudaStream_t stream){
        if (!A || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexScalarMulGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, scalarReal, scalarImag, C); }, stream);
    }



    cudaError_t sum(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result, cudaStream_t stream){
        if (!data || !result) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { sumGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, data, result); }, stream);
    }

    cudaError_t meanSquareError(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result, cudaStream_t stream){
        if (!a || !b || !result) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { meanSquareErrorGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, a, b, result); }, stream);
    }

    cudaError_t sumToOne(real_t** A, size_t nImages, size_t imageVolume, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = (imageVolume + GLOBAL_THREADS_PER_BLOCK_1D - 1) / GLOBAL_THREADS_PER_BLOCK_1D;
        return launchSync([=] { sumToOneGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK_1D, 0, stream>>>(A, nImages, imageVolume); }, stream);
    }

    cudaError_t complexAddition(complex_t** A, complex_t* sums, size_t nImages, size_t imageVolume, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = (imageVolume + GLOBAL_THREADS_PER_BLOCK_1D - 1) / GLOBAL_THREADS_PER_BLOCK_1D;
        return launchSync([=] { complexAdditionGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK_1D, 0, stream>>>(A, sums, nImages, imageVolume); }, stream);
    }

    cudaError_t complexAddition(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexAdditionGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C); }, stream);
    }


    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexElementwiseMatMulGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C); }, stream);
    }

    cudaError_t complexElementwiseMatMulConjugate(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexElementwiseMatMulConjugateGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C); }, stream);
    }

    cudaError_t complexElementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexElementwiseMatDivGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon); }, stream);
    }

    cudaError_t complexElementwiseMatDivStabilized(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { complexElementwiseMatDivStabilizedGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon); }, stream);
    }
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(size_t Nx, size_t Ny, size_t Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream) {
        if (!psf || !laplacian_fft) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { calculateLaplacianGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, psf, laplacian_fft); }, stream);
    }

    cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientXGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradX); }, stream);
    }

    cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientYGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradY); }, stream);
    }

    cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientZGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradZ); }, stream);
    }

cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv, cudaStream_t stream) {
        if (!div || !tv) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { computeTVGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, lambda, div, tv); }, stream);
    }

cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { normalizeTVGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon); }, stream);
    }

    // Gradient functions for real-valued data
cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientXGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradX); }, stream);
    }

cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientYGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradY); }, stream);
    }

cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientZGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradZ); }, stream);
    }

// Combined gradient (computes all three gradients in a single pass)
cudaError_t grad(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ, cudaStream_t stream) {
        if (!image || !gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { gradientGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradX, gradY, gradZ); }, stream);
    }

// Divergence (backward differences — adjoint of forward gradient)
cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result, cudaStream_t stream) {
        if (!gx || !gy || !gz || !result) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { divergenceGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideGx, strideGy, strideGz, strideOut, gx, gy, gz, result); }, stream);
    }

cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result, cudaStream_t stream) {
        if (!gx || !gy || !gz || !result) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { divergenceGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gx, gy, gz, result); }, stream);
    }

cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv, cudaStream_t stream) {
        if (!div || !tv) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { computeTVGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideDiv, strideTv, lambda, div, tv); }, stream);
    }

cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t beta, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { normalizeTVGlobalReal<<<blocks, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideGradX, strideGradY, strideGradZ, gradX, gradY, gradZ, beta); }, stream);
    }
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft) {
        if (!Afft || !laplacianfft) {
            return cudaErrorInvalidValue;
        }
        dim3 blocks = computeBlocksPerGrid(Nx, Ny, Nz);
        return launchSync([=] { calculateLaplacianTiledGlobal<<<blocks, GLOBAL_THREADS_PER_BLOCK>>>(Nx, Ny, Nz, Afft, laplacianfft); }, cudaStreamLegacy);
    }
}

namespace CUBE_FTT {


    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, size_t stride, real_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }
        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
        return launchSync([=] { octantFourierShiftGlobal<<<blocks, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, stride, data); }, stream);
    }

    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, complex_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }
        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocks((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                    (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);
        return launchSync([=] { octantFourierShiftGlobal<<<blocks, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, data); }, stream);
    }

    cudaError_t normalizeData(size_t Nx, size_t Ny, size_t Nz, complex_t* d_data, cudaStream_t stream) {
        if (!d_data) {
            return cudaErrorInvalidValue;
        }
        size_t num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 1024;
        size_t num_blocks = (num_elements + block_size - 1) / block_size;
        return launchSync([=] { normalizeDataGlobal<<<num_blocks, block_size, 0, stream>>>(Nx, Ny, Nz, d_data); }, stream);
    }
}
