#include "kernels.h"
#include <cuComplex.h>
#include <cufft.h>
#include <fftw3.h>
#include <iostream>

// Conversion
__global__
void fftwToCuComplexKernelGlobal(cuComplex* cuArr, fftw_complex* fftwArr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N * N) {
        cuArr[idx] = make_cuComplex(fftwArr[idx][0], fftwArr[idx][1]);
        //printf("cuArr[%d]: %f\n", idx, cuCrealf(cuArr[idx]));
    }
}
__global__
void fftwToCufftComplexKernelGlobal(cufftComplex* cufftArr, fftw_complex* fftwArr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N * N) {
        cufftArr[idx].x = fftwArr[idx][0];  // Realteil
        cufftArr[idx].y = fftwArr[idx][1];  // Imaginärteil
    }
}
__global__
void cuToFftwComplexKernelGlobal(fftw_complex* fftwArr, cuComplex* cuArr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N * N) {
        //printf("cuArr[%d]: %f\n", idx, cuCrealf(cuArr[idx]));
        fftwArr[idx][0] = cuCrealf(cuArr[idx]);  // Realteil
        fftwArr[idx][1] = cuCimagf(cuArr[idx]);  // Imaginärteil
    }
}
__global__
void cufftToFftwComplexKernelGlobal(fftw_complex* fftwArr, cufftComplex* cufftArr, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N * N * N) {
        fftwArr[idx][0] = cufftArr[idx].x;  // Realteil
        fftwArr[idx][1] = cufftArr[idx].y;  // Imaginärteil
    }
}


// Mat operations
__global__
void complexMatMulFftwComplexGlobal(int N, fftw_complex* A, fftw_complex* B, fftw_complex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;
        fftw_complex sum = {0.0, 0.0};

        for (int k = 0; k < N; ++k) {
            int indexA = x * N * N + y * N + k;
            int indexB = k * N * N + y * N + z;

            float realA = A[indexA][0];
            float imagA = A[indexA][1];
            float realB = B[indexB][0];
            float imagB = B[indexB][1];
            sum[0] += realA * realB - imagA * imagB;  // Realteil
            sum[1] += realA * imagB + imagA * realB;  // Imaginärteil
        }
        C[index][0] = sum[0];
        C[index][1] = sum[1];
    }
}
__global__
void complexMatMulCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;
        cuComplex sum = make_cuComplex(0.0f, 0.0f);

        for (int k = 0; k < N; ++k) {
            int indexA = x * N * N + y * N + k;
            int indexB = k * N * N + y * N + z;
            sum = cuCaddf(sum, cuCmulf(A[indexA], B[indexB]));
        }

        C[index] = sum;
    }
}
__global__
void complexElementwiseMatMulCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatDivCuComplexGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatMulCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatMulConjugateCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Get real and imaginary components of A and conjugated B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = -cuCimagf(B[index]);  // Conjugate the imaginary part

        // Perform the complex multiplication with conjugation
        float real_c = real_a * real_b - imag_a * imag_b;
        float imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result directly into the real and imaginary parts of C
        C[index].x = real_c;
        C[index].y = imag_c;
    }
}
__global__
void complexElementwiseMatDivCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

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
            C[index].x = (real_a * real_b + imag_a * imag_b) / denominator;
            C[index].y = (imag_a * real_b - real_a * imag_b) / denominator;
        }
    }
}
__global__
void complexElementwiseMatDivNaiveCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__global__
void complexElementwiseMatDivStabilizedCufftComplexGlobal(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Retrieve real and imaginary parts of A and B
        float real_a = A[index].x;
        float imag_a = A[index].y;
        float real_b = B[index].x;
        float imag_b = B[index].y;

        // Compute magnitude of B with stabilization to avoid division by zero
        float mag = fmaxf(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform stabilized complex division
        C[index].x = (real_a * real_b + imag_a * imag_b) / mag; // Real part of result
        C[index].y = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of result
    }
}

// Regularization
__global__
void calculateLaplacianCufftComplexGlobal(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientXCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradX) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientYCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradY) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientZCufftComplexGlobal(int N, cufftComplex* image, cufftComplex* gradZ) {
    int width = N;
    int height = N;
    int depth = N;

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
void computeTVCufftComplexGlobal(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
    int width = N;
    int height = N;
    int depth = N;

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
void normalizeTVCufftComplexGlobal(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
    int width = N;
    int height = N;
    int depth = N;

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
void calculateLaplacianCufftComplexTiledGlobal(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = N;
    int height = N;
    int depth = N;

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
void octantFourierShiftCufftComplexGlobal(int N, cufftComplex* data) {
    int width = N, height = N, depth = N;
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
void normalizeComplexData(cufftComplex* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Divide the real part by the number of elements
        d_data[idx].x /= N;

        // Set the imaginary part to 0
        d_data[idx].y = 0;
    }
}


// Device Kernels (TODO __device__ in decvice.cu)
__global__
void deviceTestKernelGlobal(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    //complexMatMulCuComplexDevice( N, A, B, C);
    complexElementwiseMatMulCuComplexDevice( N, A, B, C);
    //complexElementwiseMatDivCuComplexDevice( N, A, B, C);
}
__device__
void complexMatMulCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;
        cuComplex sum = make_cuComplex(0.0f, 0.0f);

        for (int k = 0; k < N; ++k) {
            int indexA = x * N * N + y * N + k;
            int indexB = k * N * N + y * N + z;
            sum = cuCaddf(sum, cuCmulf(A[indexA], B[indexB]));
        }

        C[index] = sum;
    }
}
__device__
void complexElementwiseMatMulCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatDivCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatMulCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise multiplication of A and B
        C[index] = cuCmulf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatMulConjugateCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Get real and imaginary components of A and conjugated B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = -cuCimagf(B[index]);  // Conjugate the imaginary part

        // Perform the complex multiplication with conjugation
        float real_c = real_a * real_b - imag_a * imag_b;
        float imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result directly into the real and imaginary parts of C
        C[index].x = real_c;
        C[index].y = imag_c;
    }
}
__device__
void complexElementwiseMatDivCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Get real and imaginary components of A and B
        float real_a = cuCrealf(A[index]);
        float imag_a = cuCimagf(A[index]);
        float real_b = cuCrealf(B[index]);
        float imag_b = cuCimagf(B[index]);

        // Calculate the denominator (magnitude squared of B)
        float denominator = real_b * real_b + imag_b * imag_b;

        // Apply stabilization: if denominator is smaller than epsilon, set to zero
        if ((double)denominator < epsilon) {
            C[index].x = 0.0f;  // Real part of C
            C[index].y = 0.0f;  // Imaginary part of C
        } else {
            // Perform the complex division
            C[index].x = (real_a * real_b + imag_a * imag_b) / denominator;
            C[index].y = (imag_a * real_b - real_a * imag_b) / denominator;
        }
    }
}
__device__
void complexElementwiseMatDivNaiveCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Element-wise division of A and B
        C[index] = cuCdivf(A[index], B[index]);
    }
}
__device__
void complexElementwiseMatDivStabilizedCufftComplexDevice(int N, cufftComplex* A, cufftComplex* B, cufftComplex* C, double epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < N && y < N && z < N) {
        int index = x * N * N + y * N + z;

        // Retrieve real and imaginary parts of A and B
        float real_a = A[index].x;
        float imag_a = A[index].y;
        float real_b = B[index].x;
        float imag_b = B[index].y;

        // Compute magnitude of B with stabilization to avoid division by zero
        float mag = fmaxf(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform stabilized complex division
        C[index].x = (real_a * real_b + imag_a * imag_b) / mag; // Real part of result
        C[index].y = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of result
    }
}
__device__
void calculateLaplacianCufftComplexDevice(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientXCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradX) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientYCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradY) {
    int width = N;
    int height = N;
    int depth = N;

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
void gradientZCufftComplexDevice(int N, cufftComplex* image, cufftComplex* gradZ) {
    int width = N;
    int height = N;
    int depth = N;

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
void computeTVCufftComplexDevice(int N, double lambda, cufftComplex* gx, cufftComplex* gy, cufftComplex* gz, cufftComplex* tv) {
    int width = N;
    int height = N;
    int depth = N;

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
void normalizeTVCufftComplexDevice(int N, cufftComplex* gradX, cufftComplex* gradY, cufftComplex* gradZ, double epsilon) {
    int width = N;
    int height = N;
    int depth = N;

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
__device__
void calculateLaplacianCufftComplexTiledDevice(int N, cufftComplex* Afft, cufftComplex* laplacianfft) {
    int width = N;
    int height = N;
    int depth = N;

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

