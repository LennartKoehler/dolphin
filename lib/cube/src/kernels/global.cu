#include "kernels.h"
#include <cuComplex.h>
#include <cufft.h>
#include <cufftw.h>
#include <iostream>


// Conversions
__global__
void fftwToCuComplexKernelGlobal(int Nx, int Ny, int Nz, cuComplex* cuArr, fftw_complex* fftwArr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * (Nx * Ny) + y * Nx + x;
        cuArr[idx] = make_cuComplex(fftwArr[idx][0], fftwArr[idx][1]);
    }
}
__global__
void fftwToCufftComplexKernelGlobal(int Nx, int Ny, int Nz, cufftComplex* cufftArr, fftw_complex* fftwArr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * (Nx * Ny) + y * Nx + x;
        cufftArr[idx].x = fftwArr[idx][0];  // Realteil
        cufftArr[idx].y = fftwArr[idx][1];  // Imaginärteil
    }

}
__global__
void cuToFftwComplexKernelGlobal(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cuComplex* cuArr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * (Nx * Ny) + y * Nx + x;
        fftwArr[idx][0] = cuCrealf(cuArr[idx]);  // Realteil
        fftwArr[idx][1] = cuCimagf(cuArr[idx]);  // Imaginärteil
    }
}
__global__
void cufftToFftwComplexKernelGlobal(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cufftArr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int idx = z * (Nx * Ny) + y * Nx + x;
        fftwArr[idx][0] = cufftArr[idx].x;  // Realteil
        fftwArr[idx][1] = cufftArr[idx].y;  // Imaginärteil
    }
}


// Mat operations
__global__
void complexMatMulFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        float realA = A[index][0];
        float imagA = A[index][1];
        float realB = B[index][0];
        float imagB = B[index][1];

        // Perform element-wise complex multiplication
        C[index][0] = realA * realB - imagA * imagB; // Realteil
        C[index][1] = realA * imagB + imagA * realB; // Imaginärteil
    }
}
__global__
void complexMatMulCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        cuComplex sum = make_cuComplex(0.0f, 0.0f);

        for (int k = 0; k < Nz; ++k) {
            int indexA = z * (Nx * Ny) + y * Nx + k;
            int indexB = k * (Nx * Ny) + y * Nx + x;
            sum = cuCaddf(sum, cuCmulf(A[indexA], B[indexB]));
        }

        C[index] = sum;
    }
}
__global__
void complexElementwiseMatMulCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatDivCuComplexGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        // Elementweise Division von A und B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatMulCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}

__global__
void complexElementwiseMatMulFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        float realA = A[index][0];
        float imagA = A[index][1];
        float realB = B[index][0];
        float imagB = B[index][1];

        // Perform element-wise complex multiplication
        C[index][0] = realA * realB - imagA * imagB; // Realteil
        C[index][1] = realA * imagB + imagA * realB; // Imaginärteil
    }
}

__global__
void complexElementwiseMatMulConjugateCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and conjugated B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = -cuCimagf(B[index]);  // Conjugate the imaginary part

        // Perform the complex multiplication with conjugation
        float real_c = real_a * real_b - imag_a * imag_b;
        float imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result in the output array
        C[index].x = real_c;
        C[index].y = imag_c;
    }
}

__global__
void complexElementwiseMatMulConjugateFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and conjugated B
        float real_a = A[index][0];
        float imag_a = A[index][1];
        float real_b = B[index][0];
        float imag_b = -B[index][1];

        // Perform the complex multiplication with conjugation
        float real_c = real_a * real_b - imag_a * imag_b;
        float imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result in the output array
        C[index][0]= real_c;
        C[index][1] = imag_c;
    }
}
__global__
void complexElementwiseMatDivCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = cuCimagf(B[index]);

        // Calculate the denominator (magnitude squared of B)
        double denominator = real_b * real_b + imag_b * imag_b;

        // Apply stabilization: if denominator is smaller than epsilon, set to zero
        if (denominator < epsilon) {
            C[index].x = 0.0f;  // Real part of C
            C[index].y = 0.0f;  // Imaginary part of C
        } else {
            // Perform the complex division
            C[index].x = (real_a * real_b + imag_a * imag_b) / denominator; // Real part
            C[index].y = (imag_a * real_b - real_a * imag_b) / denominator; // Imaginary part
        }
    }
}
__global__
void complexElementwiseMatDivNaiveCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Perform element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatDivStabilizedCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    // Compute the 3D coordinates of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread's coordinates are within bounds of the 3D matrix
    if (x < Nx && y < Ny && z < Nz) {
        // Calculate the linear index for the current thread's position
        int index = z * (Nx * Ny) + y * Nx + x;

        // Extract the real and imaginary components of A and B
        float real_a = A[index].x;
        float imag_a = A[index].y;
        float real_b = B[index].x;
        float imag_b = B[index].y;

        // Compute the magnitude squared of B with stabilization to avoid division by zero
        float mag = fmaxf(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform the stabilized element-wise complex division
        C[index].x = (real_a * real_b + imag_a * imag_b) / mag; // Real part of the result
        C[index].y = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of the result
    }
}

__global__
void complexElementwiseMatDivStabilizedFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* A, fftw_complex* B, fftw_complex* C, double epsilon) {
    // Compute the 3D coordinates of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread's coordinates are within bounds of the 3D matrix
    if (x < Nx && y < Ny && z < Nz) {
        // Calculate the linear index for the current thread's position
        int index = z * (Nx * Ny) + y * Nx + x;

        // Extract the real and imaginary components of A and B
        float real_a = A[index][0];
        float imag_a = A[index][1];
        float real_b = B[index][0];
        float imag_b = -B[index][1];


        // Compute the magnitude squared of B with stabilization to avoid division by zero
        float mag = fmaxf(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform the stabilized element-wise complex division
        C[index][0] = (real_a * real_b + imag_a * imag_b) / mag; // Real part of the result
        C[index][1] = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of the result
    }
}


// Regularization
__global__
void calculateLaplacianCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int index = (z * height + y) * width + x;

        // Berechne die Frequenzkomponenten
        float wx = 2 * M_PI * x / width;
        float wy = 2 * M_PI * y / height;
        float wz = 2 * M_PI * z / depth;

        // Laplace-Wert im Frequenzraum berechnen
        float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Elementweise Multiplikation im Frequenzraum
        laplacianfft[index].x = Afft[index].x * laplacian_value;  // Realteil
        laplacianfft[index].y = Afft[index].y * laplacian_value;  // Imaginärteil
    }
}
__global__
void gradientXCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width - 1 && y < height && z < depth) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + 1;

        // Compute gradient in the x-direction
        gradX[index].x = image[index].x - image[nextIndex].x; // Real part
        gradX[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradX[lastIndex].x = 0.0f;
        gradX[lastIndex].y = 0.0f;
    }
}
__global__
void gradientYCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (y < height - 1 && x < width && z < depth) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + width;

        // Compute gradient in the y-direction
        gradY[index].x = image[index].x - image[nextIndex].x; // Real part
        gradY[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradY[lastIndex].x = 0.0f;
        gradY[lastIndex].y = 0.0f;
    }
}
__global__
void gradientZCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < depth - 1 && y < height && x < width) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + height * width;

        // Compute gradient in the z-direction
        gradZ[index].x = image[index].x - image[nextIndex].x; // Real part
        gradZ[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        int lastIndex = z * height * width + y * width + x;
        gradZ[lastIndex].x = 0.0f;
        gradZ[lastIndex].y = 0.0f;
    }
}
__global__
void computeTVCufftComplexGlobal(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Retrieve the gradient components
        double dx = gx[index].x; // Assuming gradient data is in the real part
        double dy = gy[index].x;
        double dz = gz[index].x;

        // Compute the total variation (TV) value
        tv[index].x = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
        tv[index].y = 0.0f; // Assuming the output is real-valued, set the imaginary part to zero
    }
}
__global__
void normalizeTVCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Compute the norm of the vector
        double norm = sqrt(
            gradX[index].x * gradX[index].x + gradX[index].y * gradX[index].y +
            gradY[index].x * gradY[index].x + gradY[index].y * gradY[index].y +
            gradZ[index].x * gradZ[index].x + gradZ[index].y * gradZ[index].y
        );

        // Avoid division by very small values by setting a minimum threshold
        norm = fmax(norm, epsilon);

        // Normalize the components
        gradX[index].x /= norm;
        gradX[index].y /= norm;
        gradY[index].x /= norm;
        gradY[index].y /= norm;
        gradZ[index].x /= norm;
        gradZ[index].y /= norm;
    }
}


// Tiled
__global__
void calculateLaplacianCufftComplexTiledGlobal(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    // Tile dimensions, including a halo
    const int TILE_DIM = 8;
    __shared__ cufftComplex tile[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

    // Calculate global index
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int z = blockIdx.z * TILE_DIM + threadIdx.z;

    // Shared memory index (with a halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    // Check if the index is within bounds
    if (x < width && y < height && z < depth) {
        // Load the center of the tile
        int index = z * width * height + y * width + x;
        tile[tx][ty][tz] = Afft[index];

        // Load neighboring elements into shared memory (halo region)
        if (threadIdx.x == 0 && x > 0) tile[0][ty][tz] = Afft[index - 1]; // left
        if (threadIdx.x == TILE_DIM - 1 && x < width - 1) tile[tx + 1][ty][tz] = Afft[index + 1]; // right
        if (threadIdx.y == 0 && y > 0) tile[tx][0][tz] = Afft[index - width]; // down
        if (threadIdx.y == TILE_DIM - 1 && y < height - 1) tile[tx][ty + 1][tz] = Afft[index + width]; // up
        if (threadIdx.z == 0 && z > 0) tile[tx][ty][0] = Afft[index - width * height]; // back
        if (threadIdx.z == TILE_DIM - 1 && z < depth - 1) tile[tx][ty][tz + 1] = Afft[index + width * height]; // front

        __syncthreads();

        // Compute Laplacian in the frequency domain
        float wx = 2 * M_PI * x / width;
        float wy = 2 * M_PI * y / height;
        float wz = 2 * M_PI * z / depth;
        float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Apply Laplacian in the frequency domain
        laplacianfft[index].x = tile[tx][ty][tz].x * laplacian_value;
        laplacianfft[index].y = tile[tx][ty][tz].y * laplacian_value;
    }
}


// Fourier Shift
__global__
void octantFourierShiftCufftComplexGlobal(int Nx, int Ny, int Nz, cufftComplex* data) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    // Calculate the indices for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure that the thread is within bounds, just iterate over the first 4 octants
    if (z < halfDepth) {
        // Calculate the linear indices for the current element and its counterpart in the other octant
        int idx1 = z * height * width + y * width + x;
        int idx2 = ((z + halfDepth) % depth) * height * width +
                   ((y + halfHeight) % height) * width +
                   ((x + halfWidth) % width);

        // Check if the indices are different to avoid duplicate swapping
        if (idx1 != idx2) {
            // Swap the real and imaginary parts
            cufftComplex temp = data[idx1];
            data[idx1] = data[idx2];
            data[idx2] = temp;
        }
    }
}
__global__
void octantFourierShiftFftwComplexGlobal(int Nx, int Ny, int Nz, fftw_complex* data) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    // Calculate the indices for the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Ensure that the thread is within bounds, just iterate over the first 4 octants
    if (z < halfDepth) {
        // Calculate the linear indices for the current element and its counterpart in the other octant
        int idx1 = z * height * width + y * width + x;
        int idx2 = ((z + halfDepth) % depth) * height * width +
                   ((y + halfHeight) % height) * width +
                   ((x + halfWidth) % width);

        // Check if the indices are different to avoid duplicate swapping
        if (idx1 != idx2) {
            // Manually swap the real and imaginary parts of fftw_complex values
            double real1 = data[idx1][0];
            double imag1 = data[idx1][1];
            double real2 = data[idx2][0];
            double imag2 = data[idx2][1];

            // Swap in global memory
            data[idx1][0] = real2;
            data[idx1][1] = imag2;
            data[idx2][0] = real1;
            data[idx2][1] = imag1;

            //TODO
            //atomicExch((unsigned long long*)(&data[idx1]), *(unsigned long long*)&data[idx2]);
        }
    }
}
__global__
void normalizeComplexData(int Nx, int Ny, int Nz, cufftComplex* d_data) {
    // Calculate the 1D index for the 3D data array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread is within bounds of the data
    if (idx < Nx * Ny * Nz) {
        // Normalize the real part by dividing by the total number of elements
        d_data[idx].x /= (Nx * Ny * Nz);

        // Set the imaginary part to 0 as specified
        d_data[idx].y = 0;
    }
}
__global__
void normalizeFftwComplexDataGlobal(int Nx, int Ny, int Nz, fftw_complex* d_data) {
    // Calculate the 1D index for the 3D data array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread is within bounds of the data
    if (idx < Nx * Ny * Nz) {
        // Normalize the real part by dividing by the total number of elements
        d_data[idx][0] /= (Nx * Ny * Nz);

        // Set the imaginary part to 0 as specified
        d_data[idx][1] /= (Nx * Ny * Nz);
    }
}
__global__
void padCufftMatGlobal(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, cufftComplex* oldMat, cufftComplex* newMat, int offsetX, int offsetY, int offsetZ)
{
    // 3D-Index des Threads im Grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Neue Matrixgröße als Grenze
    if (x >= newNx || y >= newNy || z >= newNz) return;

    // Index in der neuen Matrix
    int newIndex = z * newNy * newNx + y * newNx + x;

    // Initialisiere die neue Matrix mit Null
    newMat[newIndex].x = 0.0f; // Realteil
    newMat[newIndex].y = 0.0f; // Imaginärteil

    // Berechnung der Position der alten Matrix
    if (x >= offsetX && x < offsetX + oldNx &&
        y >= offsetY && y < offsetY + oldNy &&
        z >= offsetZ && z < offsetZ + oldNz)
    {
        // Index in der alten Matrix
        int oldX = x - offsetX;
        int oldY = y - offsetY;
        int oldZ = z - offsetZ;
        int oldIndex = oldZ * oldNy * oldNx + oldY * oldNx + oldX;

        // Kopiere den Wert von der alten in die neue Matrix
        newMat[newIndex] = oldMat[oldIndex];
    }
}


// Device Kernels (TODO __device__ in decvice.cu)
__global__
void deviceTestKernelGlobal(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    //complexMatMulCuComplexDevice( N, A, B, C);
    complexElementwiseMatMulCuComplexDevice( Nx,Ny, Nz, A, B, C);
    //complexElementwiseMatDivCuComplexDevice( N, A, B, C);
}
__device__
void complexMatMulCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        cuComplex sum = make_cuComplex(0.0f, 0.0f);

        for (int k = 0; k < Nz; ++k) {
            int indexA = z * (Nx * Ny) + y * Nx + k;
            int indexB = k * (Nx * Ny) + y * Nx + x;
            sum = cuCaddf(sum, cuCmulf(A[indexA], B[indexB]));
        }

        C[index] = sum;
    }
}
__device__
void complexElementwiseMatMulCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatDivCuComplexDevice(int Nx, int Ny, int Nz, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        // Elementweise Division von A und B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatMulCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatMulConjugateCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and conjugated B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = -cuCimagf(B[index]);  // Conjugate the imaginary part

        // Perform the complex multiplication with conjugation
        float real_c = real_a * real_b - imag_a * imag_b;
        float imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result in the output array
        C[index].x = real_c;
        C[index].y = imag_c;
    }
}
__device__
void complexElementwiseMatDivCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = cuCimagf(B[index]);

        // Calculate the denominator (magnitude squared of B)
        double denominator = real_b * real_b + imag_b * imag_b;

        // Apply stabilization: if denominator is smaller than epsilon, set to zero
        if (denominator < epsilon) {
            C[index].x = 0.0f;  // Real part of C
            C[index].y = 0.0f;  // Imaginary part of C
        } else {
            // Perform the complex division
            C[index].x = (real_a * real_b + imag_a * imag_b) / denominator; // Real part
            C[index].y = (imag_a * real_b - real_a * imag_b) / denominator; // Imaginary part
        }
    }
}
__device__
void complexElementwiseMatDivNaiveCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Perform element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatDivStabilizedCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    // Compute the 3D coordinates of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread's coordinates are within bounds of the 3D matrix
    if (x < Nx && y < Ny && z < Nz) {
        // Calculate the linear index for the current thread's position
        int index = z * (Nx * Ny) + y * Nx + x;

        // Extract the real and imaginary components of A and B
        float real_a = A[index].x;
        float imag_a = A[index].y;
        float real_b = B[index].x;
        float imag_b = B[index].y;

        // Compute the magnitude squared of B with stabilization to avoid division by zero
        float mag = fmaxf(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform the stabilized element-wise complex division
        C[index].x = (real_a * real_b + imag_a * imag_b) / mag; // Real part of the result
        C[index].y = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of the result
    }
}
__device__
void calculateLaplacianCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int index = (z * height + y) * width + x;

        // Berechne die Frequenzkomponenten
        float wx = 2 * M_PI * x / width;
        float wy = 2 * M_PI * y / height;
        float wz = 2 * M_PI * z / depth;

        // Laplace-Wert im Frequenzraum berechnen
        float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Elementweise Multiplikation im Frequenzraum
        laplacianfft[index].x = Afft[index].x * laplacian_value;  // Realteil
        laplacianfft[index].y = Afft[index].y * laplacian_value;  // Imaginärteil
    }
}
__device__
void gradientXCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradX) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width - 1 && y < height && z < depth) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + 1;

        // Compute gradient in the x-direction
        gradX[index].x = image[index].x - image[nextIndex].x; // Real part
        gradX[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradX[lastIndex].x = 0.0f;
        gradX[lastIndex].y = 0.0f;
    }
}
__device__
void gradientYCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradY) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (y < height - 1 && x < width && z < depth) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + width;

        // Compute gradient in the y-direction
        gradY[index].x = image[index].x - image[nextIndex].x; // Real part
        gradY[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradY[lastIndex].x = 0.0f;
        gradY[lastIndex].y = 0.0f;
    }
}
__device__
void gradientZCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* image, cufftComplex* gradZ) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (z < depth - 1 && y < height && x < width) {
        int index = z * height * width + y * width + x;
        int nextIndex = index + height * width;

        // Compute gradient in the z-direction
        gradZ[index].x = image[index].x - image[nextIndex].x; // Real part
        gradZ[index].y = image[index].y - image[nextIndex].y; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        int lastIndex = z * height * width + y * width + x;
        gradZ[lastIndex].x = 0.0f;
        gradZ[lastIndex].y = 0.0f;
    }
}
__device__
void computeTVCufftComplexDevice(int Nx, int Ny, int Nz, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Retrieve the gradient components
        double dx = gx[index].x; // Assuming gradient data is in the real part
        double dy = gy[index].x;
        double dz = gz[index].x;

        // Compute the total variation (TV) value
        tv[index].x = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
        tv[index].y = 0.0f; // Assuming the output is real-valued, set the imaginary part to zero
    }
}
__device__
void normalizeTVCufftComplexDevice(int Nx, int Ny, int Nz, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Compute the norm of the vector
        double norm = sqrt(
            gradX[index].x * gradX[index].x + gradX[index].y * gradX[index].y +
            gradY[index].x * gradY[index].x + gradY[index].y * gradY[index].y +
            gradZ[index].x * gradZ[index].x + gradZ[index].y * gradZ[index].y
        );

        // Avoid division by very small values by setting a minimum threshold
        norm = fmax(norm, epsilon);

        // Normalize the components
        gradX[index].x /= norm;
        gradX[index].y /= norm;
        gradY[index].x /= norm;
        gradY[index].y /= norm;
        gradZ[index].x /= norm;
        gradZ[index].y /= norm;
    }
}
__global__
void padCufftMatDevice(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, cufftComplex* oldMat, cufftComplex* newMat, int offsetX, int offsetY, int offsetZ)
{
    // 3D-Index des Threads im Grid
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Neue Matrixgröße als Grenze
    if (x >= newNx || y >= newNy || z >= newNz) return;

    // Index in der neuen Matrix
    int newIndex = z * newNy * newNx + y * newNx + x;

    // Initialisiere die neue Matrix mit Null
    newMat[newIndex].x = 0.0f; // Realteil
    newMat[newIndex].y = 0.0f; // Imaginärteil

    // Berechnung der Position der alten Matrix
    if (x >= offsetX && x < offsetX + oldNx &&
        y >= offsetY && y < offsetY + oldNy &&
        z >= offsetZ && z < offsetZ + oldNz)
    {
        // Index in der alten Matrix
        int oldX = x - offsetX;
        int oldY = y - offsetY;
        int oldZ = z - offsetZ;
        int oldIndex = oldZ * oldNy * oldNx + oldY * oldNx + oldX;

        // Kopiere den Wert von der alten in die neue Matrix
        newMat[newIndex] = oldMat[oldIndex];
    }
}
__device__
void calculateLaplacianCufftComplexTiledDevice(int Nx, int Ny, int Nz, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    // Tile dimensions, including a halo
    const int TILE_DIM = 8;
    __shared__ cufftComplex tile[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

    // Calculate global index
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int z = blockIdx.z * TILE_DIM + threadIdx.z;

    // Shared memory index (with a halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    // Check if the index is within bounds
    if (x < width && y < height && z < depth) {
        // Load the center of the tile
        int index = z * width * height + y * width + x;
        tile[tx][ty][tz] = Afft[index];

        // Load neighboring elements into shared memory (halo region)
        if (threadIdx.x == 0 && x > 0) tile[0][ty][tz] = Afft[index - 1]; // left
        if (threadIdx.x == TILE_DIM - 1 && x < width - 1) tile[tx + 1][ty][tz] = Afft[index + 1]; // right
        if (threadIdx.y == 0 && y > 0) tile[tx][0][tz] = Afft[index - width]; // down
        if (threadIdx.y == TILE_DIM - 1 && y < height - 1) tile[tx][ty + 1][tz] = Afft[index + width]; // up
        if (threadIdx.z == 0 && z > 0) tile[tx][ty][0] = Afft[index - width * height]; // back
        if (threadIdx.z == TILE_DIM - 1 && z < depth - 1) tile[tx][ty][tz + 1] = Afft[index + width * height]; // front

        __syncthreads();

        // Compute Laplacian in the frequency domain
        float wx = 2 * M_PI * x / width;
        float wy = 2 * M_PI * y / height;
        float wz = 2 * M_PI * z / depth;
        float laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Apply Laplacian in the frequency domain
        laplacianfft[index].x = tile[tx][ty][tz].x * laplacian_value;
        laplacianfft[index].y = tile[tx][ty][tz].y * laplacian_value;
    }
}

