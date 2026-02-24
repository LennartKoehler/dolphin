#include "kernels.h"
#include <iostream>

// Conversions (removed - no longer needed with single complex_t type)

// Mat operations
__global__
void complexMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        real_t realA = A[index][0];
        real_t imagA = A[index][1];
        real_t realB = B[index][0];
        real_t imagB = B[index][1];

        // Perform element-wise complex_t multiplication
        C[index][0] = realA * realB - imagA * imagB; // Realteil
        C[index][1] = realA * imagB + imagA * realB; // Imaginärteil
    }
}

__global__
void complexScalarMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    real_t realB = B[0];
    real_t imagB = B[1];
    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        real_t realA = A[index][0];
        real_t imagA = A[index][1];
        C[index][0] = realA * realB - imagA * imagB; // Realteil
        C[index][1] = realA * imagB + imagA * realB; // Imaginärteil
    }
}



__global__
void complexAdditionGlobal(complex_t** data, complex_t* sums, int nImages, int imageVolume) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;

    complex_t sum{0,0};

    if (position < imageVolume) {


        for (int i = 0; i < nImages; ++i){
            sum[0] += data[i][position][0];
            sum[1] += data[i][position][1];
        }

        sums[position][0] = sum[0];
        sums[position][1] = sum[1];
    }
    
}

__global__
void complexAdditionGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        C[index][0] = B[index][0] + C[index][0];
        C[index][1] = B[index][1] + C[index][1];
    }
}
__global__
void sumToOneRealGlobal(complex_t** data, int nImages, int imageVolume) {
    int position = blockIdx.x * blockDim.x + threadIdx.x;

    complex_t sum{0,0};

    if (position < imageVolume) {


        for (int i = 0; i < nImages; ++i){
            sum[0] += data[i][position][0];
            sum[1] += data[i][position][1];
        }

        for (int i = 0; i < nImages; ++i){
            data[i][position][0] /= sum[0];
            data[i][position][1] /= sum[1];
        }
    }
    
}

__global__
void complexElementwiseMatMulGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;

        real_t realA = A[index][0];
        real_t imagA = A[index][1];
        real_t realB = B[index][0];
        real_t imagB = B[index][1];

        // Perform element-wise complex_t multiplication
        C[index][0] = realA * realB - imagA * imagB; // Realteil
        C[index][1] = realA * imagB + imagA * realB; // Imaginärteil
    }
}

__global__
void complexElementwiseMatMulConjugateGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and conjugated B
        real_t real_a = A[index][0];
        real_t imag_a = A[index][1];
        real_t real_b = B[index][0];
        real_t imag_b = -B[index][1];

        // Perform the complex_t multiplication with conjugation
        real_t real_c = real_a * real_b - imag_a * imag_b;
        real_t imag_c = real_a * imag_b + imag_a * real_b;

        // Store the result in the output array
        C[index][0] = real_c;
        C[index][1] = imag_c;
    }
}

__global__
void complexElementwiseMatDivGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and B
        real_t real_a = A[index][0];
        real_t imag_a = A[index][1];
        real_t real_b = B[index][0];
        real_t imag_b = B[index][1];

        // Calculate the denominator (magnitude squared of B)
        real_t denominator = real_b * real_b + imag_b * imag_b;

        // Apply stabilization: if denominator^2 is smaller than epsilon, set to zero
        if (denominator < epsilon) {
            C[index][0] = 0.0;  // Real part of C
            C[index][1] = 0.0;  // Imaginary part of C
        } else {
            // Perform the complex_t division
            C[index][0] = (real_a * real_b + imag_a * imag_b) / denominator; // Real part
            C[index][1] = (imag_a * real_b - real_a * imag_b) / denominator; // Imaginary part
        }
    }
}

__global__
void complexElementwiseMatDivStabilizedGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    // Compute the 3D coordinates of the current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread's coordinates are within bounds of the 3D matrix
    if (x < Nx && y < Ny && z < Nz) {
        // Calculate the linear index for the current thread's position
        int index = z * (Nx * Ny) + y * Nx + x;

        // Extract the real and imaginary components of A and B
        real_t real_a = A[index][0];
        real_t imag_a = A[index][1];
        real_t real_b = B[index][0];
        real_t imag_b = -B[index][1];


        // Compute the magnitude squared of B with stabilization to avoid division by zero
        real_t mag = fmax(epsilon, real_b * real_b + imag_b * imag_b);

        // Perform the stabilized element-wise complex_t division
        C[index][0] = (real_a * real_b + imag_a * imag_b) / mag; // Real part of the result
        C[index][1] = (imag_a * real_b - real_a * imag_b) / mag; // Imaginary part of the result
    }
}

// Regularization
__global__
void calculateLaplacianGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int index = (z * height + y) * width + x;

        // Berechne die Frequenzkomponenten
        real_t wx = 2 * M_PI * x / width;
        real_t wy = 2 * M_PI * y / height;
        real_t wz = 2 * M_PI * z / depth;

        // Laplace-Wert im Frequenzraum berechnen
        real_t laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Elementweise Multiplikation im Frequenzraum
        laplacianfft[index][0] = Afft[index][0] * laplacian_value;  // Realteil
        laplacianfft[index][1] = Afft[index][1] * laplacian_value;  // Imaginärteil
    }
}

__global__
void gradientXGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX) {
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
        gradX[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradX[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradX[lastIndex][0] = 0.0;
        gradX[lastIndex][1] = 0.0;
    }
}

__global__
void gradientYGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY) {
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
        gradY[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradY[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradY[lastIndex][0] = 0.0;
        gradY[lastIndex][1] = 0.0;
    }
}

__global__
void gradientZGlobal(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ) {
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
        gradZ[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradZ[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        int lastIndex = z * height * width + y * width + x;
        gradZ[lastIndex][0] = 0.0;
        gradZ[lastIndex][1] = 0.0;
    }
}

__global__
void computeTVGlobal(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Retrieve the gradient components
        real_t dx = gx[index][0]; // Assuming gradient data is in the real part
        real_t dy = gy[index][0];
        real_t dz = gz[index][0];

        // Compute the total variation (TV) value
        tv[index][0] = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
        tv[index][1] = 0.0; // Assuming the output is real-valued, set the imaginary part to zero
    }
}

__global__
void normalizeTVGlobal(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Compute the norm of the vector
        real_t norm = sqrt(
            gradX[index][0] * gradX[index][0] + gradX[index][1] * gradX[index][1] +
            gradY[index][0] * gradY[index][0] + gradY[index][1] * gradY[index][1] +
            gradZ[index][0] * gradZ[index][0] + gradZ[index][1] * gradZ[index][1]
        );

        // Avoid division by very small values by setting a minimum threshold
        norm = fmax(norm, epsilon);

        // Normalize the components
        gradX[index][0] /= norm;
        gradX[index][1] /= norm;
        gradY[index][0] /= norm;
        gradY[index][1] /= norm;
        gradZ[index][0] /= norm;
        gradZ[index][1] /= norm;
    }
}

// Tiled
__global__
void calculateLaplacianTiledGlobal(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    // Tile dimensions, including a halo
    const int TILE_DIM = 8;
    __shared__ complex_t tile[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

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
        // Copy components explicitly because `complex_t` is an array type and cannot be assigned directly
        tile[tx][ty][tz][0] = Afft[index][0];
        tile[tx][ty][tz][1] = Afft[index][1];

        // Load neighboring elements into shared memory (halo region)
        if (threadIdx.x == 0 && x > 0) {
            int idx = index - 1;
            tile[0][ty][tz][0] = Afft[idx][0]; // left
            tile[0][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.x == TILE_DIM - 1 && x < width - 1) {
            int idx = index + 1;
            tile[tx + 1][ty][tz][0] = Afft[idx][0]; // right
            tile[tx + 1][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == 0 && y > 0) {
            int idx = index - width;
            tile[tx][0][tz][0] = Afft[idx][0]; // down
            tile[tx][0][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == TILE_DIM - 1 && y < height - 1) {
            int idx = index + width;
            tile[tx][ty + 1][tz][0] = Afft[idx][0]; // up
            tile[tx][ty + 1][tz][1] = Afft[idx][1];
        }
        if (threadIdx.z == 0 && z > 0) {
            int idx = index - width * height;
            tile[tx][ty][0][0] = Afft[idx][0]; // back
            tile[tx][ty][0][1] = Afft[idx][1];
        }
        if (threadIdx.z == TILE_DIM - 1 && z < depth - 1) {
            int idx = index + width * height;
            tile[tx][ty][tz + 1][0] = Afft[idx][0]; // front
            tile[tx][ty][tz + 1][1] = Afft[idx][1];
        }

        __syncthreads();

        // Compute Laplacian in the frequency domain
        real_t wx = 2 * M_PI * x / width;
        real_t wy = 2 * M_PI * y / height;
        real_t wz = 2 * M_PI * z / depth;
        real_t laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Apply Laplacian in the frequency domain
        laplacianfft[index][0] = tile[tx][ty][tz][0] * laplacian_value;
        laplacianfft[index][1] = tile[tx][ty][tz][1] * laplacian_value;
    }
}

// Fourier Shift
__global__
void normalizeDataGlobal(int Nx, int Ny, int Nz, complex_t* d_data) {
    // Calculate the 1D index for the 3D data array
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread is within bounds of the data
    if (idx < Nx * Ny * Nz) {
        // Normalize the real part by dividing by the total number of elements
        d_data[idx][0] /= (Nx * Ny * Nz);

        // Normalize the img part by dividing by the total number of elements
        d_data[idx][1] /= (Nx * Ny * Nz);
    }
}

__global__
void octantFourierShiftGlobal(int Nx, int Ny, int Nz, complex_t* data) {
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
            // Manually swap the real and imaginary parts of complex_t values
            real_t real1 = data[idx1][0];
            real_t imag1 = data[idx1][1];
            real_t real2 = data[idx2][0];
            real_t imag2 = data[idx2][1];

            // Swap in global memory
            data[idx1][0] = real2;
            data[idx1][1] = imag2;
            data[idx2][0] = real1;
            data[idx2][1] = imag1;
        }
    }
}

__global__
void padMatGlobal(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat, int offsetX, int offsetY, int offsetZ)
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
    newMat[newIndex][0] = 0.0; // Realteil
    newMat[newIndex][1] = 0.0; // Imaginärteil

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
        newMat[newIndex][0] = oldMat[oldIndex][0];
        newMat[newIndex][1] = oldMat[oldIndex][1];
    }
}

// Device Kernels
__global__
void deviceTestKernelGlobal(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    //complexMatMulDevice( N, A, B, C);
    complexElementwiseMatMulDevice( Nx,Ny, Nz, A, B, C);
    //complexElementwiseMatDivDevice( N, A, B, C);
}

__device__
void complexMatMulDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        complex_t sum = {0.0, 0.0};

        for (int k = 0; k < Nz; ++k) {
            int indexA = z * (Nx * Ny) + y * Nx + k;
            int indexB = k * (Nx * Ny) + y * Nx + x;
            // Manual complex_t multiplication
            real_t realA = A[indexA][0], imagA = A[indexA][1];
            real_t realB = B[indexB][0], imagB = B[indexB][1];
            sum[0] += realA * realB - imagA * imagB;
            sum[1] += realA * imagB + imagA * realB;
        }

        C[index][0] = sum[0];
        C[index][1] = sum[1];
    }
}

__device__
void complexElementwiseMatMulDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        int index = z * (Nx * Ny) + y * Nx + x;
        real_t realA = A[index][0], imagA = A[index][1];
        real_t realB = B[index][0], imagB = B[index][1];
        C[index][0] = realA * realB - imagA * imagB;
        C[index][1] = realA * imagB + imagA * realB;
    }
}

__device__
void complexElementwiseMatDivDevice(int Nx, int Ny, int Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        int index = z * (Nx * Ny) + y * Nx + x;

        // Get real and imaginary components of A and B
        real_t real_a = A[index][0];
        real_t imag_a = A[index][1];
        real_t real_b = B[index][0];
        real_t imag_b = B[index][1];

        // Calculate the denominator (magnitude squared of B)
        real_t denominator = real_b * real_b + imag_b * imag_b;

        // Apply stabilization: if denominator is smaller than epsilon, set to zero
        if (denominator < epsilon) {
            C[index][0] = 0.0;  // Real part of C
            C[index][1] = 0.0;  // Imaginary part of C
        } else {
            // Perform the complex_t division
            C[index][0] = (real_a * real_b + imag_a * imag_b) / denominator; // Real part
            C[index][1] = (imag_a * real_b - real_a * imag_b) / denominator; // Imaginary part
        }
    }
}

__device__
void calculateLaplacianDevice(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        int index = (z * height + y) * width + x;

        // Berechne die Frequenzkomponenten
        real_t wx = 2 * M_PI * x / width;
        real_t wy = 2 * M_PI * y / height;
        real_t wz = 2 * M_PI * z / depth;

        // Laplace-Wert im Frequenzraum berechnen
        real_t laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Elementweise Multiplikation im Frequenzraum
        laplacianfft[index][0] = Afft[index][0] * laplacian_value;  // Realteil
        laplacianfft[index][1] = Afft[index][1] * laplacian_value;  // Imaginärteil
    }
}

__device__
void gradientXDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradX) {
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
        gradX[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradX[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradX[lastIndex][0] = 0.0;
        gradX[lastIndex][1] = 0.0;
    }
}

__device__
void gradientYDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradY) {
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
        gradY[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradY[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        int lastIndex = z * height * width + y * width + x;
        gradY[lastIndex][0] = 0.0;
        gradY[lastIndex][1] = 0.0;
    }
}

__device__
void gradientZDevice(int Nx, int Ny, int Nz, complex_t* image, complex_t* gradZ) {
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
        gradZ[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradZ[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        int lastIndex = z * height * width + y * width + x;
        gradZ[lastIndex][0] = 0.0;
        gradZ[lastIndex][1] = 0.0;
    }
}

__device__
void computeTVDevice(int Nx, int Ny, int Nz, real_t lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Retrieve the gradient components
        real_t dx = gx[index][0]; // Assuming gradient data is in the real part
        real_t dy = gy[index][0];
        real_t dz = gz[index][0];

        // Compute the total variation (TV) value
        tv[index][0] = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
        tv[index][1] = 0.0; // Assuming the output is real-valued, set the imaginary part to zero
    }
}

__device__
void normalizeTVDevice(int Nx, int Ny, int Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Compute the norm of the vector
        real_t norm = sqrt(
            gradX[index][0] * gradX[index][0] + gradX[index][1] * gradX[index][1] +
            gradY[index][0] * gradY[index][0] + gradY[index][1] * gradY[index][1] +
            gradZ[index][0] * gradZ[index][0] + gradZ[index][1] * gradZ[index][1]
        );

        // Avoid division by very small values by setting a minimum threshold
        norm = fmax(norm, epsilon);

        // Normalize the components
        gradX[index][0] /= norm;
        gradX[index][1] /= norm;
        gradY[index][0] /= norm;
        gradY[index][1] /= norm;
        gradZ[index][0] /= norm;
        gradZ[index][1] /= norm;
    }
}

__global__
void padMatDevice(int oldNx, int oldNy, int oldNz, int newNx, int newNy, int newNz, complex_t* oldMat, complex_t* newMat, int offsetX, int offsetY, int offsetZ)
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
    newMat[newIndex][0] = 0.0; // Realteil
    newMat[newIndex][1] = 0.0; // Imaginärteil

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
        newMat[newIndex][0] = oldMat[oldIndex][0];
        newMat[newIndex][1] = oldMat[oldIndex][1];
    }
}

__device__
void calculateLaplacianTiledDevice(int Nx, int Ny, int Nz, complex_t* Afft, complex_t* laplacianfft) {
    int width = Nx;
    int height = Ny;
    int depth = Nz;

    // Tile dimensions, including a halo
    const int TILE_DIM = 8;
    __shared__ complex_t tile[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

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
        // Copy components explicitly because `complex_t` is an array type and cannot be assigned directly
        tile[tx][ty][tz][0] = Afft[index][0];
        tile[tx][ty][tz][1] = Afft[index][1];

        // Load neighboring elements into shared memory (halo region)
        if (threadIdx.x == 0 && x > 0) {
            int idx = index - 1;
            tile[0][ty][tz][0] = Afft[idx][0]; // left
            tile[0][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.x == TILE_DIM - 1 && x < width - 1) {
            int idx = index + 1;
            tile[tx + 1][ty][tz][0] = Afft[idx][0]; // right
            tile[tx + 1][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == 0 && y > 0) {
            int idx = index - width;
            tile[tx][0][tz][0] = Afft[idx][0]; // down
            tile[tx][0][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == TILE_DIM - 1 && y < height - 1) {
            int idx = index + width;
            tile[tx][ty + 1][tz][0] = Afft[idx][0]; // up
            tile[tx][ty + 1][tz][1] = Afft[idx][1];
        }
        if (threadIdx.z == 0 && z > 0) {
            int idx = index - width * height;
            tile[tx][ty][0][0] = Afft[idx][0]; // back
            tile[tx][ty][0][1] = Afft[idx][1];
        }
        if (threadIdx.z == TILE_DIM - 1 && z < depth - 1) {
            int idx = index + width * height;
            tile[tx][ty][tz + 1][0] = Afft[idx][0]; // front
            tile[tx][ty][tz + 1][1] = Afft[idx][1];
        }

        __syncthreads();

        // Compute Laplacian in the frequency domain
        real_t wx = 2 * M_PI * x / width;
        real_t wy = 2 * M_PI * y / height;
        real_t wz = 2 * M_PI * z / depth;
        real_t laplacian_value = -2 * (cos(wx) + cos(wy) + cos(wz) - 3);

        // Apply Laplacian in the frequency domain
        laplacianfft[index][0] = tile[tx][ty][tz][0] * laplacian_value;
        laplacianfft[index][1] = tile[tx][ty][tz][1] * laplacian_value;
    }
}