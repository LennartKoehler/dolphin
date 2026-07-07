#include <cuda_runtime_api.h>
#include "operations.h"
#include "kernels.h"



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

// Macro to handle CUDA kernel launch with event synchronization and error checking
// Usage: CUDA_CHECK_KERNEL(kernel_launch, stream);
// The kernel_launch should be the full kernel expression including <<<...>>>
#define CUDA_CHECK_KERNEL(kernel_launch, stream) \
    do { \
        cudaEvent_t _event; \
        cudaError_t _err = cudaEventCreate(&_event); \
        if (_err != cudaSuccess) { \
            return _err; \
        } \
        kernel_launch; \
        _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        _err = cudaEventRecord(_event, stream); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        _err = cudaEventSynchronize(_event); \
        if (_err != cudaSuccess) { \
            cudaEventDestroy(_event); \
            return _err; \
        } \
        cudaEventDestroy(_event); \
    } while(0)




namespace CUBE_MAT {


cudaError_t elementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (elementwiseMatDivGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideB, strideC, A, B, C, epsilon)),
                stream);
        return cudaSuccess;
    }

cudaError_t scalarMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t B, real_t* C, cudaStream_t stream) {
        if (!A || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (scalarMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideC, A, B, C)),
                stream);
        return cudaSuccess;
    }

cudaError_t elementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (elementwiseMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideA, strideB, strideC, A, B, C)),
                stream);
        return cudaSuccess;
    }

    cudaError_t complexMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexScalarMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t scalarReal, real_t scalarImag, complex_t* C, cudaStream_t stream){
        if (!A || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexScalarMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, scalarReal, scalarImag, C);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }



    cudaError_t sum(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result, cudaStream_t stream){
        if (!data || !result) {
            return cudaErrorInvalidValue;
        }

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (sumGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, data, result)),
                stream);
        return cudaSuccess;
    }

    cudaError_t meanSquareError(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result, cudaStream_t stream){
        if (!a || !b || !result) {
            return cudaErrorInvalidValue;
        }

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (meanSquareErrorGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, a, b, result)),
                stream);
        return cudaSuccess;
    }

    cudaError_t sumToOne(real_t** A, size_t nImages, size_t imageVolume, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration

        size_t blocksPerGrid = (imageVolume + GLOBAL_THREADS_PER_BLOCK_1D - 1) / GLOBAL_THREADS_PER_BLOCK_1D;
        sumToOneGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK_1D, 0, stream>>>(A, nImages, imageVolume);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexAddition(complex_t** A, complex_t* sums, size_t nImages, size_t imageVolume, cudaStream_t stream){
        if (!A ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        size_t blocksPerGrid = (imageVolume + GLOBAL_THREADS_PER_BLOCK_1D - 1) / GLOBAL_THREADS_PER_BLOCK_1D;

        complexAdditionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK_1D, 0, stream>>>(A, sums, nImages, imageVolume);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexAddition(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream){
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexAdditionGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }


    // Elementwise Matrix Multiplication/Division (always GPU)
    cudaError_t complexElementwiseMatMul(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatMulGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexElementwiseMatMulConjugate(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatMulConjugateGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexElementwiseMatDiv(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatDivGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t complexElementwiseMatDivStabilized(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon, cudaStream_t stream) {
        if (!A || !B || !C) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        complexElementwiseMatDivStabilizedGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, A, B, C, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }
}

namespace CUBE_REG {
    // Regularization
    cudaError_t calculateLaplacian(size_t Nx, size_t Ny, size_t Nz, complex_t* psf, complex_t* laplacian_fft, cudaStream_t stream) {
        if (!psf || !laplacian_fft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        calculateLaplacianGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, psf, laplacian_fft);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientXGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradX);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientYGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradY);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientZGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, image, gradZ);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv, cudaStream_t stream) {
        if (!div || !tv) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        computeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, lambda, div, tv);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        normalizeTVGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gradX, gradY, gradZ, epsilon);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    // Gradient functions for real-valued data
cudaError_t gradX(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, cudaStream_t stream) {
        if (!image || !gradX) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientXGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradX);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t gradY(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY, cudaStream_t stream) {
        if (!image || !gradY) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientYGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradY);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t gradZ(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ, cudaStream_t stream) {
        if (!image || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        gradientZGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradZ);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

// Combined gradient (computes all three gradients in a single pass)
cudaError_t grad(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ, cudaStream_t stream) {
        if (!image || !gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);
        CUDA_CHECK_KERNEL(
                (gradientGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideIn, strideOut, image, gradX, gradY, gradZ)),
                stream);
        return cudaSuccess;
    }

// Divergence (backward differences — adjoint of forward gradient)
cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result, cudaStream_t stream) {
        if (!gx || !gy || !gz || !result) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        divergenceGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideGx, strideGy, strideGz, strideOut, gx, gy, gz, result);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t divergence(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result, cudaStream_t stream) {
        if (!gx || !gy || !gz || !result) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        divergenceGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, gx, gy, gz, result);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t computeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv, cudaStream_t stream) {
        if (!div || !tv) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        computeTVGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideDiv, strideTv, lambda, div, tv);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

cudaError_t normalizeTV(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t beta, cudaStream_t stream) {
        if (!gradX || !gradY || !gradZ) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        normalizeTVGlobalReal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK, 0, stream>>>(Nx, Ny, Nz, strideGradX, strideGradY, strideGradZ, gradX, gradY, gradZ, beta);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }
}

namespace CUBE_TILED {
    // Tiled Memory in GPU
    cudaError_t calculateLaplacianTiled(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft) {
        if (!Afft || !laplacianfft) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Use global kernel configuration
        dim3 blocksPerGrid = computeBlocksPerGrid(Nx, Ny, Nz);

        calculateLaplacianTiledGlobal<<<blocksPerGrid, GLOBAL_THREADS_PER_BLOCK>>>(Nx, Ny, Nz, Afft, laplacianfft);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaEventDestroy(event);
            return err;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }
}

namespace CUBE_FTT {


    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, size_t stride, real_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        octantFourierShiftGlobal<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, stride, data);
        cudaError_t errp = cudaPeekAtLastError();
        if (errp != cudaSuccess) {
            cudaEventDestroy(event);
            return errp;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    cudaError_t octantFourierShift(size_t Nx, size_t Ny, size_t Nz, complex_t* data, cudaStream_t stream) {
        if (!data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        // Kernel dimension 3D, because 3D matrix stored in 1D array, index in kernel operation depend on structure
        dim3 threadsPerBlock(2, 2, 2); //=6 //TODO with more threads artefacts visible
        dim3 blocksPerGrid((Nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (Ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                           (Nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

        octantFourierShiftGlobal<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(Nx, Ny, Nz, data);
        cudaError_t errp = cudaPeekAtLastError();
        if (errp != cudaSuccess) {
            cudaEventDestroy(event);
            return errp;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }

    // cudaError_t padMat(size_t oldNx, size_t oldNy, size_t oldNz, size_t newNx, size_t newNy, size_t newNz, complex_t* oldMat, complex_t* newMat)
    // {
    //     if (!oldMat || !newMat) {
    //         return cudaErrorInvalidValue;
    //     }
    //
    //     auto start = std::chrono::high_resolution_clock::now();
    //     // Sicherheitsprüfung: Neue Dimensionen müssen größer oder gleich den alten sein
    //     if (newNx < oldNx || newNy < oldNy || newNz < oldNz) {
    //         return cudaErrorInvalidValue;
    //     }
    //
    //     // Offset für Padding (Startkoordinaten der alten Matrix in der neuen Matrix)
    //     size_t offsetX = (newNx - oldNx) / 2;
    //     size_t offsetY = (newNy - oldNy) / 2;
    //     size_t offsetZ = (newNz - oldNz) / 2;
    //
    //     // Initialisiere die neue Matrix mit Nullen
    //     for (size_t i = 0; i < newNx * newNy * newNz; ++i) {
    //         newMat[i][0] = 0.0; // Realteil
    //         newMat[i][1] = 0.0; // Imaginärteil
    //     }
    //
    //     // Kopiere die Werte der alten Matrix in die Mitte der neuen Matrix
    //     for (size_t z = 0; z < oldNz; ++z) {
    //         for (size_t y = 0; y < oldNy; ++y) {
    //             for (size_t x = 0; x < oldNx; ++x) {
    //                 // Index in der alten Matrix
    //                 size_t oldIndex = z * oldNy * oldNx + y * oldNx + x;
    //
    //                 // Index in der neuen Matrix
    //                 size_t newIndex =
    //                     (z + offsetZ) * newNy * newNx +
    //                     (y + offsetY) * newNx +
    //                     (x + offsetX);
    //
    //                 // Kopiere den Wert
    //                 newMat[newIndex][0] = oldMat[oldIndex][0]; // Realteil
    //                 newMat[newIndex][1] = oldMat[oldIndex][1]; // Imaginärteil
    //             }
    //         }
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    //     float time = duration.count();
    //
    //
    //
    //     return cudaSuccess;
    // }

    cudaError_t normalizeData(size_t Nx, size_t Ny, size_t Nz, complex_t* d_data, cudaStream_t stream) {
        if (!d_data) {
            return cudaErrorInvalidValue;
        }

        cudaEvent_t event;
        cudaEventCreate(&event);

        size_t num_elements = Nx * Ny * Nz;  // Beispiel: Gesamtzahl der Elemente
        int block_size = 1024;
        size_t num_blocks = (num_elements + block_size - 1) / block_size;

        normalizeDataGlobal<<<num_blocks, block_size, 0, stream>>>(Nx, Ny, Nz, d_data);
        cudaError_t errp = cudaPeekAtLastError();
        if (errp != cudaSuccess) {
            cudaEventDestroy(event);
            return errp;
        }

        cudaEventRecord(event);
        cudaError_t syncErr = cudaEventSynchronize(event);
        if (syncErr != cudaSuccess) {
            cudaEventDestroy(event);
            return syncErr;
        }

        cudaEventDestroy(event);
        return cudaSuccess;
    }
}
