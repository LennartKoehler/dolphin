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
#include "kernels.h"

// Device-side grid, index and arithmetic helpers shared by the kernels in
// this translation unit.
namespace detail {

__device__ inline size_t tx() { return static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; }
__device__ inline size_t ty() { return static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y; }
__device__ inline size_t tz() { return static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z; }

__device__ inline bool inBounds(size_t x, size_t y, size_t z, size_t Nx, size_t Ny, size_t Nz) {
    return x < Nx && y < Ny && z < Nz;
}

// Compact (stride = dimension) 3D index
__device__ inline size_t index3D(size_t x, size_t y, size_t z, size_t Nx, size_t Ny) {
    return z * (Nx * Ny) + y * Nx + x;
}

// Strided 3D index for padded rows
__device__ inline size_t indexStrided(size_t x, size_t y, size_t z, size_t stride, size_t Ny) {
    return z * (stride * Ny) + y * stride + x;
}

__device__ inline void cMul(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) {
    cr = ar * br - ai * bi;
    ci = ar * bi + ai * br;
}

// Elementwise binary operations on complex values, applied per grid index.
struct ComplexMulOp {
    __device__ void operator()(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) const {
        cMul(ar, ai, br, bi, cr, ci);
    }
};

struct ComplexAddOp {
    __device__ void operator()(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) const {
        cr = ar + br;
        ci = ai + bi;
    }
};

struct ComplexMulConjugateOp {
    __device__ void operator()(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) const {
        cMul(ar, ai, br, -bi, cr, ci);
    }
};

struct ComplexDivOp {
    real_t epsilon;
    __device__ void operator()(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) const {
        real_t denom = br * br + bi * bi;
        if (denom < epsilon) {
            cr = 0.0;
            ci = 0.0;
        } else {
            cr = (ar * br + ai * bi) / denom;
            ci = (ai * br - ar * bi) / denom;
        }
    }
};

struct ComplexDivStabilizedOp {
    real_t epsilon;
    __device__ void operator()(real_t ar, real_t ai, real_t br, real_t bi, real_t& cr, real_t& ci) const {
        real_t brj = br;
        real_t bij = -bi;
        real_t mag = fmax(epsilon, brj * brj + bij * bij);
        cr = (ar * brj + ai * bij) / mag;
        ci = (ai * brj - ar * bij) / mag;
    }
};

template <typename Op>
__device__ inline void complexElementwiseBody(size_t x, size_t y, size_t z, size_t Nx, size_t Ny, size_t Nz,
                                              const complex_t* A, const complex_t* B, complex_t* C, Op op) {
    if (!inBounds(x, y, z, Nx, Ny, Nz)) return;
    size_t index = index3D(x, y, z, Nx, Ny);
    real_t ar = A[index][0];
    real_t ai = A[index][1];
    real_t br = B[index][0];
    real_t bi = B[index][1];
    real_t cr, ci;
    op(ar, ai, br, bi, cr, ci);
    C[index][0] = cr;
    C[index][1] = ci;
}

} // namespace detail

// Mat operations
// this is not matmul lol
__global__
void complexMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexMulOp{});
}


__global__
void elementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C){
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexA = detail::indexStrided(x, y, z, strideA, Ny);
        size_t indexB = detail::indexStrided(x, y, z, strideB, Ny);
        size_t indexC = detail::indexStrided(x, y, z, strideC, Ny);
        C[indexC] = B[indexB] * A[indexA];
    }
}

__global__
void scalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideC, real_t* A, real_t b, real_t* C){
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexA = detail::indexStrided(x, y, z, strideA, Ny);
        size_t indexC = detail::indexStrided(x, y, z, strideC, Ny);
        C[indexC] = A[indexA] * b;
    }
}

__global__
void elementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, size_t strideA, size_t strideB, size_t strideC, real_t* A, real_t* B, real_t* C, real_t epsilon){
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexA = detail::indexStrided(x, y, z, strideA, Ny);
        size_t indexB = detail::indexStrided(x, y, z, strideB, Ny);
        size_t indexC = detail::indexStrided(x, y, z, strideC, Ny);
        real_t denominator = B[indexB];
        if (denominator < epsilon) C[indexC] = 0;
        else C[indexC] = A[indexA] / B[indexB];
    }
}

__global__
void complexScalarMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, real_t realB, real_t imagB, complex_t* C) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        real_t realA = A[index][0];
        real_t imagA = A[index][1];
        real_t realC, imagC;
        detail::cMul(realA, imagA, realB, imagB, realC, imagC);
        C[index][0] = realC;
        C[index][1] = imagC;
    }
}



__global__
void complexAdditionGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexAddOp{});
}

__global__
void sumGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* data, complex_t* result) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        atomicAdd(&result[0][0], data[index][0]);
        atomicAdd(&result[0][1], data[index][1]);
    }
}

__global__
void meanSquareErrorGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* a, complex_t* b, real_t* result) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        real_t dr = a[index][0] - b[index][0];
        real_t di = a[index][1] - b[index][1];
        atomicAdd(result, dr * dr + di * di);
    }
}

__global__
void sumToOneGlobal(size_t Nx, size_t Ny, size_t Nz, size_t stride, size_t sliceStride, real_t** data, size_t nImages) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (x < Nx && y < Ny && z < Nz) {
        size_t offset = z * sliceStride + y * stride + x;

        real_t sum{0};
        for (size_t i = 0; i < nImages; ++i) {
            sum += data[i][offset];
        }

        for (size_t i = 0; i < nImages; ++i) {
            if (sum == 0.0f) {
                data[i][offset] = 0.0f;
            } else {
                data[i][offset] /= sum;
            }
        }
    }
}

__global__
void complexElementwiseMatMulGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexMulOp{});
}

__global__
void complexElementwiseMatMulConjugateGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexMulConjugateOp{});
}

__global__
void complexElementwiseMatDivGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexDivOp{epsilon});
}

__global__
void complexElementwiseMatDivStabilizedGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* A, complex_t* B, complex_t* C, real_t epsilon) {
    detail::complexElementwiseBody(detail::tx(), detail::ty(), detail::tz(), Nx, Ny, Nz, A, B, C, detail::ComplexDivStabilizedOp{epsilon});
}

// Regularization
__global__
void calculateLaplacianGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* Afft, complex_t* laplacianfft) {
    size_t width = Nx;
    size_t height = Ny;
    size_t depth = Nz;

    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, width, height, depth)) {
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
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (x < Nx - 1 && y < Ny && z < Nz) {
        size_t indexIn = detail::indexStrided(x, y, z, strideIn, Ny);
        // Compute gradient in the x-direction
        gradX[detail::indexStrided(x, y, z, strideOut, Ny)] = image[indexIn] - image[indexIn + 1];
    }

    // Handle boundary condition at the last x position
    if (x == Nx - 1 && y < Ny && z < Nz) {
        gradX[detail::indexStrided(x, y, z, strideOut, Ny)] = 0.0;
    }
}

__global__
void gradientYGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradY) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (y < Ny - 1 && x < Nx && z < Nz) {
        size_t indexIn = detail::indexStrided(x, y, z, strideIn, Ny);
        // Compute gradient in the y-direction
        gradY[detail::indexStrided(x, y, z, strideOut, Ny)] = image[indexIn] - image[indexIn + strideIn];
    }

    // Handle boundary condition at the last y position
    if (y == Ny - 1 && x < Nx && z < Nz) {
        gradY[detail::indexStrided(x, y, z, strideOut, Ny)] = 0.0;
    }
}

__global__
void gradientZGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradZ) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (z < Nz - 1 && y < Ny && x < Nx) {
        size_t indexIn = detail::indexStrided(x, y, z, strideIn, Ny);
        // Compute gradient in the z-direction
        gradZ[detail::indexStrided(x, y, z, strideOut, Ny)] = image[indexIn] - image[indexIn + strideIn * Ny];
    }

    // Handle boundary condition at the last z position
    if (z == Nz - 1 && y < Ny && x < Nx) {
        gradZ[detail::indexStrided(x, y, z, strideOut, Ny)] = 0.0;
    }
}

// Combined gradient kernel (computes all three gradients in a single pass)
__global__
void gradientGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideIn, size_t strideOut, real_t* image, real_t* gradX, real_t* gradY, real_t* gradZ) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexIn = detail::indexStrided(x, y, z, strideIn, Ny);
        size_t indexOut = detail::indexStrided(x, y, z, strideOut, Ny);

        // Gradient in x-direction: forward difference
        if (x < Nx - 1) {
            gradX[indexOut] = image[indexIn + 1] - image[indexIn];
        } else {
            gradX[indexOut] = 0.0;
        }

        // Gradient in y-direction: forward difference
        if (y < Ny - 1) {
            gradY[indexOut] = image[indexIn + strideIn] - image[indexIn];
        } else {
            gradY[indexOut] = 0.0;
        }

        // Gradient in z-direction: forward difference
        if (z < Nz - 1) {
            gradZ[indexOut] = image[indexIn + strideIn * Ny] - image[indexIn];
        } else {
            gradZ[indexOut] = 0.0;
        }
    }
}

__global__
void computeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideDiv, size_t strideTv, real_t lambda, real_t* div, real_t* tv) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexDiv = detail::indexStrided(x, y, z, strideDiv, Ny);
        size_t indexTv = detail::indexStrided(x, y, z, strideTv, Ny);

        // TV damping factor: tv = 1 / (1 - lambda * div)
        // div is negative at edges/spikes (standard forward gradient + backward divergence),
        // so the denominator > 1 there, producing damping.
        real_t d = div[indexDiv];
        real_t denom = 1.0 - lambda * d;
        denom = fmax(denom, (real_t)1e-8);

        tv[indexTv] = static_cast<real_t>(1.0 / denom);
    }
}

__global__
void normalizeTVGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGradX, size_t strideGradY, size_t strideGradZ, real_t* gradX, real_t* gradY, real_t* gradZ, real_t epsilon) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexGradX = detail::indexStrided(x, y, z, strideGradX, Ny);
        size_t indexGradY = detail::indexStrided(x, y, z, strideGradY, Ny);
        size_t indexGradZ = detail::indexStrided(x, y, z, strideGradZ, Ny);

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
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (x < Nx - 1 && y < Ny && z < Nz) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);

        // Compute gradient in the x-direction
        gradX[index][0] = image[index + 1][0] - image[index][0]; // Real part
        gradX[index][1] = image[index + 1][1] - image[index][1]; // Imaginary part
    }

    // Handle boundary condition at the last x position
    if (x == Nx - 1 && y < Ny && z < Nz) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        gradX[index][0] = 0.0;
        gradX[index][1] = 0.0;
    }
}

__global__
void gradientYGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradY) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (y < Ny - 1 && x < Nx && z < Nz) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);

        // Compute gradient in the y-direction
        gradY[index][0] = image[index + Nx][0] - image[index][0]; // Real part
        gradY[index][1] = image[index + Nx][1] - image[index][1]; // Imaginary part
    }

    // Handle boundary condition at the last y position
    if (y == Ny - 1 && x < Nx && z < Nz) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        gradY[index][0] = 0.0;
        gradY[index][1] = 0.0;
    }
}

__global__
void gradientZGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* image, complex_t* gradZ) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (z < Nz - 1 && y < Ny && x < Nx) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);

        // Compute gradient in the z-direction
        gradZ[index][0] = image[index + Ny * Nx][0] - image[index][0]; // Real part
        gradZ[index][1] = image[index + Ny * Nx][1] - image[index][1]; // Imaginary part
    }

    // Handle boundary condition at the last z position
    if (z == Nz - 1 && y < Ny && x < Nx) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);
        gradZ[index][0] = 0.0;
        gradZ[index][1] = 0.0;
    }
}

// Divergence kernels (backward differences — adjoint of forward gradient)
__global__
void divergenceGlobalReal(size_t Nx, size_t Ny, size_t Nz, size_t strideGx, size_t strideGy, size_t strideGz, size_t strideOut, real_t* gx, real_t* gy, real_t* gz, real_t* result) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t indexGx = detail::indexStrided(x, y, z, strideGx, Ny);
        size_t indexGy = detail::indexStrided(x, y, z, strideGy, Ny);
        size_t indexGz = detail::indexStrided(x, y, z, strideGz, Ny);
        size_t indexOut = detail::indexStrided(x, y, z, strideOut, Ny);

        // Backward difference in x: gx[x] - gx[x-1], with 0 at x=0
        real_t divX = gx[indexGx] - (x > 0 ? gx[indexGx - 1] : real_t(0));
        // Backward difference in y: gy[y] - gy[y-1], with 0 at y=0
        real_t divY = gy[indexGy] - (y > 0 ? gy[indexGy - strideGy] : real_t(0));
        // Backward difference in z: gz[z] - gz[z-1], with 0 at z=0
        real_t divZ = gz[indexGz] - (z > 0 ? gz[indexGz - strideGz * Ny] : real_t(0));

        result[indexOut] = divX + divY + divZ;
    }
}

__global__
void divergenceGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* result) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        size_t index = detail::index3D(x, y, z, Nx, Ny);

        // Backward difference in x (real part only for TV divergence)
        real_t divX_r = gx[index][0] - (x > 0 ? gx[index - 1][0] : real_t(0));
        real_t divX_i = gx[index][1] - (x > 0 ? gx[index - 1][1] : real_t(0));
        // Backward difference in y
        real_t divY_r = gy[index][0] - (y > 0 ? gy[index - Nx][0] : real_t(0));
        real_t divY_i = gy[index][1] - (y > 0 ? gy[index - Nx][1] : real_t(0));
        // Backward difference in z
        real_t divZ_r = gz[index][0] - (z > 0 ? gz[index - Ny * Nx][0] : real_t(0));
        real_t divZ_i = gz[index][1] - (z > 0 ? gz[index - Ny * Nx][1] : real_t(0));

        result[index][0] = divX_r + divY_r + divZ_r;
        result[index][1] = divX_i + divY_i + divZ_i;
    }
}

__global__
void computeTVGlobal(size_t Nx, size_t Ny, size_t Nz, real_t lambda, complex_t* div, complex_t* tv) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();
    size_t index = detail::index3D(x, y, z, Nx, Ny);

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
        // TV damping factor: tv = 1 / (1 - lambda * div)
        // div is negative at edges/spikes (standard forward gradient + backward divergence),
        // so the denominator > 1 there, producing damping.
        real_t d = div[index][0]; // real part only
        real_t denom = 1.0 - lambda * d;
        denom = fmax(denom, (real_t)1e-8);

        tv[index][0] = static_cast<real_t>(1.0 / denom);
        tv[index][1] = 0.0;
    }
}

__global__
void normalizeTVGlobal(size_t Nx, size_t Ny, size_t Nz, complex_t* gradX, complex_t* gradY, complex_t* gradZ, real_t epsilon) {
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();
    size_t index = detail::index3D(x, y, z, Nx, Ny);

    if (detail::inBounds(x, y, z, Nx, Ny, Nz)) {
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
    size_t halfWidth = Nx / 2;
    size_t halfHeight = Ny / 2;
    size_t halfDepth = Nz / 2;

    // Calculate the indices for the current thread
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    // Ensure that the thread is within bounds, just iterate over the first half of the depth
    if (x < Nx && y < Ny && z < halfDepth) {
        // Calculate the linear indices using stride for addressing
        // but modular arithmetic on logical dimensions for swap positions
        size_t idx1 = z * (stride * Ny) + y * stride + x;
        size_t idx2 = ((z + halfDepth) % Nz) * (stride * Ny) +
                    ((y + halfHeight) % Ny) * stride +
                    ((x + halfWidth) % Nx);

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
    size_t halfWidth = Nx / 2;
    size_t halfHeight = Ny / 2;
    size_t halfDepth = Nz / 2;

    // Calculate the indices for the current thread
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

    // Ensure that the thread is within bounds, just iterate over the first 4 octants
    if (z < halfDepth) {
        // Calculate the linear indices for the current element and its counterpart in the other octant
        size_t idx1 = z * Ny * Nx + y * Nx + x;
        size_t idx2 = ((z + halfDepth) % Nz) * Ny * Nx +
                    ((y + halfHeight) % Ny) * Nx +
                    ((x + halfWidth) % Nx);

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
    size_t x = detail::tx();
    size_t y = detail::ty();
    size_t z = detail::tz();

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
