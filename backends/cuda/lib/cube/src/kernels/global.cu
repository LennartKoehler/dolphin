#include "kernels.h"

// Conversions (removed - no longer needed with single complex_t type)
//
//


// Mat operations
// this is not matmul lol
__global__
void complexMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;

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
void elementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C){
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t indexA = z * (strideA * Ny) + y * strideA + x;
        size_t indexB = z * (strideB * Ny) + y * strideB + x;
        size_t indexC = z * (strideC * Ny) + y * strideC + x;
        C[indexC] = B[indexB] * A[indexA];
    }
}

__global__
void scalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t b, real_t* C){
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t indexA = z * (strideA * Ny) + y * strideA + x;
        size_t indexC = z * (strideC * Ny) + y * strideC + x;
        C[indexC] = A[indexA] * b;
    }
}

__global__
void elementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon){
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t indexA = z * (strideA * Ny) + y * strideA + x;
        size_t indexB = z * (strideB * Ny) + y * strideB + x;
        size_t indexC = z * (strideC * Ny) + y * strideC + x;
        real_t denominator = B[indexB];
        if (denominator < epsilon) C[indexC] = 0;
        else C[indexC] = A[indexA] / B[indexB];
    }
}

__global__
void complexScalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t realB, real_t imagB, complex_t* C) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;
        real_t realA = A[index][0];
        real_t imagA = A[index][1];
        C[index][0] = realA * realB - imagA * imagB;
        C[index][1] = realA * imagB + imagA * realB;
    }
}



__global__
void complexAdditionGlobal(complex_t** data, complex_t* sums, size_t nImages, size_t imageVolume) {
    size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    complex_t sum{0,0};

    if (position < imageVolume) {


        for (size_t i = 0; i < nImages; ++i){
            sum[0] += data[i][position][0];
            sum[1] += data[i][position][1];
        }

        sums[position][0] = sum[0];
        sums[position][1] = sum[1];
    }

}

__global__
void complexAdditionGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;
        C[index][0] = B[index][0] + A[index][0];
        C[index][1] = B[index][1] + A[index][1];
    }
}
__global__
void sumGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;
        atomicAdd(&result[0][0], data[index][0]);
        atomicAdd(&result[0][1], data[index][1]);
    }
}

__global__
void meanSquareErrorGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;
        real_t dr = a[index][0] - b[index][0];
        real_t di = a[index][1] - b[index][1];
        atomicAdd(result, dr * dr + di * di);
    }
}

__global__
void sumToOneGlobal(real_t** data, size_t nImages, size_t imageVolume) {
    size_t position = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    real_t sum{0};

    if (position < imageVolume) {


        for (size_t i = 0; i < nImages; ++i){
            sum += data[i][position];
        }

        for (size_t i = 0; i < nImages; ++i){
            data[i][position] /= sum;
        }
    }

}

__global__
void complexElementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < Nx && y < Ny && z < Nz) {
        size_t index = z * (Nx * Ny) + y * Nx + x;

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
void complexElementwiseMatMulConjugateGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        size_t index = z * (Nx * Ny) + y * Nx + x;

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
void complexElementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Check if the thread is within the valid bounds of the 3D grid
    if (x < Nx && y < Ny && z < Nz) {
        // Compute the 1D index from the 3D coordinates
        size_t index = z * (Nx * Ny) + y * Nx + x;

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
void complexElementwiseMatDivStabilizedGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    // Compute the 3D coordinates of the current thread
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Check if the thread's coordinates are within bounds of the 3D matrix
    if (x < Nx && y < Ny && z < Nz) {
        // Calculate the linear index for the current thread's position
        size_t index = z * (Nx * Ny) + y * Nx + x;

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
void calculateLaplacianGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t index = (z * height + y) * width + x;

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

// Gradient kernels for real-valued data
__global__
void gradientXGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width - 1 && y < height && z < depth) {
        size_t indexIn = z * (strideIn * height) + y * strideIn + x;
        size_t nextIndexIn = indexIn + 1;
        size_t indexOut = z * (strideOut * height) + y * strideOut + x;

        // Compute gradient in the x-direction
        gradX[indexOut] = image[indexIn] - image[nextIndexIn];
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        size_t lastIndexOut = z * (strideOut * height) + y * strideOut + x;
        gradX[lastIndexOut] = 0.0;
    }
}

__global__
void gradientYGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (y < height - 1 && x < width && z < depth) {
        size_t indexIn = z * (strideIn * height) + y * strideIn + x;
        size_t nextIndexIn = indexIn + strideIn;
        size_t indexOut = z * (strideOut * height) + y * strideOut + x;

        // Compute gradient in the y-direction
        gradY[indexOut] = image[indexIn] - image[nextIndexIn];
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        size_t lastIndexOut = z * (strideOut * height) + y * strideOut + x;
        gradY[lastIndexOut] = 0.0;
    }
}

__global__
void gradientZGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (z < depth - 1 && y < height && x < width) {
        size_t indexIn = z * (strideIn * height) + y * strideIn + x;
        size_t nextIndexIn = indexIn + strideIn * height;
        size_t indexOut = z * (strideOut * height) + y * strideOut + x;

        // Compute gradient in the z-direction
        gradZ[indexOut] = image[indexIn] - image[nextIndexIn];
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        size_t lastIndexOut = z * (strideOut * height) + y * strideOut + x;
        gradZ[lastIndexOut] = 0.0;
    }
}

// Combined gradient kernel (computes all three gradients in a single pass)
__global__
void gradientGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t indexIn = z * (strideIn * height) + y * strideIn + x;
        size_t indexOut = z * (strideOut * height) + y * strideOut + x;

        // Gradient in x-direction: forward difference
        if (x < width - 1) {
            gradX[indexOut] = image[indexIn] - image[indexIn + 1];
        } else {
            gradX[indexOut] = 0.0;
        }

        // Gradient in y-direction: forward difference
        if (y < height - 1) {
            gradY[indexOut] = image[indexIn] - image[indexIn + strideIn];
        } else {
            gradY[indexOut] = 0.0;
        }

        // Gradient in z-direction: forward difference
        if (z < depth - 1) {
            gradZ[indexOut] = image[indexIn] - image[indexIn + strideIn * height];
        } else {
            gradZ[indexOut] = 0.0;
        }
    }
}

__global__
void computeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t indexDiv = z * (strideDiv * height) + y * strideDiv + x;
        size_t indexTv = z * (strideTv * height) + y * strideTv + x;

        // TV damping factor: tv = 1 / (1 + lambda * div)
        // The denominator is always >= 1 for lambda > 0
        real_t d = div[indexDiv];
        real_t denom = 1.0 - lambda * d;
        denom = fmax(denom, (real_t)1e-8);

        tv[indexTv] = static_cast<real_t>(1.0 / denom);
    }
}

__global__
void normalizeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t epsilon) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t indexGradX = z * (strideGradX * height) + y * strideGradX + x;
        size_t indexGradY = z * (strideGradY * height) + y * strideGradY + x;
        size_t indexGradZ = z * (strideGradZ * height) + y * strideGradZ + x;

        // Smoothed TV subgradient: gx / sqrt(|nabla f|² + beta²)
        // beta prevents noise amplification in flat regions
        real_t normSq =
            gradX[indexGradX] * gradX[indexGradX] +
            gradY[indexGradY] * gradY[indexGradY] +
            gradZ[indexGradZ] * gradZ[indexGradZ];
        real_t norm = sqrt(normSq + epsilon);

        gradX[indexGradX] /= norm;
        gradY[indexGradY] /= norm;
        gradZ[indexGradZ] /= norm;
    }
}

__global__
void gradientXGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradX) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width - 1 && y < height && z < depth) {
        size_t index = z * height * width + y * width + x;
        size_t nextIndex = index + 1;

        // Compute gradient in the x-direction
        gradX[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradX[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == width - 1 && y < height && z < depth) {
        size_t lastIndex = z * height * width + y * width + x;
        gradX[lastIndex][0] = 0.0;
        gradX[lastIndex][1] = 0.0;
    }
}

__global__
void gradientYGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (y < height - 1 && x < width && z < depth) {
        size_t index = z * height * width + y * width + x;
        size_t nextIndex = index + width;

        // Compute gradient in the y-direction
        gradY[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradY[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == height - 1 && x < width && z < depth) {
        size_t lastIndex = z * height * width + y * width + x;
        gradY[lastIndex][0] = 0.0;
        gradY[lastIndex][1] = 0.0;
    }
}

__global__
void gradientZGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (z < depth - 1 && y < height && x < width) {
        size_t index = z * height * width + y * width + x;
        size_t nextIndex = index + height * width;

        // Compute gradient in the z-direction
        gradZ[index][0] = image[index][0] - image[nextIndex][0]; // Real part
        gradZ[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == depth - 1 && y < height && x < width) {
        size_t lastIndex = z * height * width + y * width + x;
        gradZ[lastIndex][0] = 0.0;
        gradZ[lastIndex][1] = 0.0;
    }
}

// Divergence kernels (backward differences — adjoint of forward gradient)
__global__
void divergenceGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t indexGx = z * (strideGx * height) + y * strideGx + x;
        size_t indexGy = z * (strideGy * height) + y * strideGy + x;
        size_t indexGz = z * (strideGz * height) + y * strideGz + x;
        size_t indexOut = z * (strideOut * height) + y * strideOut + x;

        // Backward difference in x: gx[x] - gx[x-1], with 0 at x=0
        real_t divX = gx[indexGx] - (x > 0 ? gx[indexGx - 1] : real_t(0));
        // Backward difference in y: gy[y] - gy[y-1], with 0 at y=0
        real_t divY = gy[indexGy] - (y > 0 ? gy[indexGy - strideGy] : real_t(0));
        // Backward difference in z: gz[z] - gz[z-1], with 0 at z=0
        real_t divZ = gz[indexGz] - (z > 0 ? gz[indexGz - strideGz * height] : real_t(0));

        result[indexOut] = divX + divY + divZ;
    }
}

__global__
void divergenceGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x < width && y < height && z < depth) {
        size_t index = z * height * width + y * width + x;

        // Backward difference in x (real part only for TV divergence)
        real_t divX_r = gx[index][0] - (x > 0 ? gx[index - 1][0] : real_t(0));
        real_t divX_i = gx[index][1] - (x > 0 ? gx[index - 1][1] : real_t(0));
        // Backward difference in y
        real_t divY_r = gy[index][0] - (y > 0 ? gy[index - width][0] : real_t(0));
        real_t divY_i = gy[index][1] - (y > 0 ? gy[index - width][1] : real_t(0));
        // Backward difference in z
        real_t divZ_r = gz[index][0] - (z > 0 ? gz[index - height * width][0] : real_t(0));
        real_t divZ_i = gz[index][1] - (z > 0 ? gz[index - height * width][1] : real_t(0));

        result[index][0] = divX_r + divY_r + divZ_r;
        result[index][1] = divX_i + divY_i + divZ_i;
    }
}

__global__
void computeTVGlobal(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    size_t index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // TV damping factor: tv = 1 / (1 + lambda * div)
        // The denominator is always >= 1 for lambda > 0
        real_t d = div[index][0]; // real part only
        real_t denom = 1.0 - lambda * d;
        denom = fmax(denom, (real_t)1e-8);

        tv[index][0] = static_cast<real_t>(1.0 / denom);
        tv[index][1] = 0.0;
    }
}

__global__
void normalizeTVGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    size_t index = z * height * width + y * width + x;

    if (x < width && y < height && z < depth) {
        // Smoothed TV subgradient: gx / sqrt(|nabla f|² + beta²)
        // beta prevents noise amplification in flat regions
        real_t normSq =
            gradX[index][0] * gradX[index][0] + gradX[index][1] * gradX[index][1] +
            gradY[index][0] * gradY[index][0] + gradY[index][1] * gradY[index][1] +
            gradZ[index][0] * gradZ[index][0] + gradZ[index][1] * gradZ[index][1];
        real_t norm = sqrt(normSq + epsilon);

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
void calculateLaplacianTiledGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    // Tile dimensions, including a halo
    const int TILE_DIM = 8;
    __shared__ complex_t tile[TILE_DIM + 2][TILE_DIM + 2][TILE_DIM + 2];

    // Calculate global index
    size_t x = static_cast<size_t>(blockIdx.x) * TILE_DIM + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * TILE_DIM + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * TILE_DIM + threadIdx.z;

    // Shared memory index (with a halo)
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    int tz = threadIdx.z + 1;

    // Check if the index is within bounds
    if (x < width && y < height && z < depth) {
        // Load the center of the tile
        size_t index = z * width * height + y * width + x;
        // Copy components explicitly because `complex_t` is an array type and cannot be assigned directly
        tile[tx][ty][tz][0] = Afft[index][0];
        tile[tx][ty][tz][1] = Afft[index][1];

        // Load neighboring elements into shared memory (halo region)
        if (threadIdx.x == 0 && x > 0) {
            size_t idx = index - 1;
            tile[0][ty][tz][0] = Afft[idx][0]; // left
            tile[0][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.x == TILE_DIM - 1 && x < width - 1) {
            size_t idx = index + 1;
            tile[tx + 1][ty][tz][0] = Afft[idx][0]; // right
            tile[tx + 1][ty][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == 0 && y > 0) {
            size_t idx = index - width;
            tile[tx][0][tz][0] = Afft[idx][0]; // down
            tile[tx][0][tz][1] = Afft[idx][1];
        }
        if (threadIdx.y == TILE_DIM - 1 && y < height - 1) {
            size_t idx = index + width;
            tile[tx][ty + 1][tz][0] = Afft[idx][0]; // up
            tile[tx][ty + 1][tz][1] = Afft[idx][1];
        }
        if (threadIdx.z == 0 && z > 0) {
            size_t idx = index - width * height;
            tile[tx][ty][0][0] = Afft[idx][0]; // back
            tile[tx][ty][0][1] = Afft[idx][1];
        }
        if (threadIdx.z == TILE_DIM - 1 && z < depth - 1) {
            size_t idx = index + width * height;
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
void normalizeDataGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* d_data) {
    // Calculate the 1D index for the 3D data array
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Ensure that the thread is within bounds of the data
    if (idx < Nx * Ny * Nz) {
        // Normalize the real part by dividing by the total number of elements
        d_data[idx][0] /= (Nx * Ny * Nz);

        // Normalize the img part by dividing by the total number of elements
        d_data[idx][1] /= (Nx * Ny * Nz);
    }
}

__global__
void octantFourierShiftGlobal(size_t Nx, size_t Ny, size_t Nz, size_t stride, real_t* data) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;
    size_t halfWidth = width / 2;
    size_t halfHeight = height / 2;
    size_t halfDepth = depth / 2;

    // Calculate the indices for the current thread
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Ensure that the thread is within bounds, just iterate over the first half of the depth
    if (x < width && y < height && z < halfDepth) {
        // Calculate the linear indices using stride for addressing
        // but modular arithmetic on logical dimensions for swap positions
        size_t idx1 = z * (stride * height) + y * stride + x;
        size_t idx2 = ((z + halfDepth) % depth) * (stride * height) +
                   ((y + halfHeight) % height) * stride +
                   ((x + halfWidth) % width);

        // Check if the indices are different to avoid duplicate swapping
        if (idx1 != idx2) {
            // Swap real values in global memory
            real_t val1 = data[idx1];
            real_t val2 = data[idx2];
            data[idx1] = val2;
            data[idx2] = val1;
        }
    }
}

__global__
void octantFourierShiftGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* data) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;
    size_t halfWidth = width / 2;
    size_t halfHeight = height / 2;
    size_t halfDepth = depth / 2;

    // Calculate the indices for the current thread
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Ensure that the thread is within bounds, just iterate over the first 4 octants
    if (z < halfDepth) {
        // Calculate the linear indices for the current element and its counterpart in the other octant
        size_t idx1 = z * height * width + y * width + x;
        size_t idx2 = ((z + halfDepth) % depth) * height * width +
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
void padMatGlobal(size_t oldNx, size_t oldNy, size_t oldNz, size_t newNx, size_t newNy, size_t newNz, complex_t* oldMat, complex_t* newMat, size_t offsetX, size_t offsetY, size_t offsetZ)
{
    // 3D-Index des Threads im Grid
    size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    // Neue Matrixgröße als Grenze
    if (x >= newNx || y >= newNy || z >= newNz) return;

    // Index in der neuen Matrix
    size_t newIndex = z * newNy * newNx + y * newNx + x;

    // Initialisiere die neue Matrix mit Null
    newMat[newIndex][0] = 0.0; // Realteil
    newMat[newIndex][1] = 0.0; // Imaginärteil

    // Berechnung der Position der alten Matrix
    if (x >= offsetX && x < offsetX + oldNx &&
        y >= offsetY && y < offsetY + oldNy &&
        z >= offsetZ && z < offsetZ + oldNz)
    {
        // Index in der alten Matrix
        size_t oldX = x - offsetX;
        size_t oldY = y - offsetY;
        size_t oldZ = z - offsetZ;
        size_t oldIndex = oldZ * oldNy * oldNx + oldY * oldNx + oldX;

        // Kopiere den Wert von der alten in die neue Matrix
        newMat[newIndex][0] = oldMat[oldIndex][0];
        newMat[newIndex][1] = oldMat[oldIndex][1];
    }
}
