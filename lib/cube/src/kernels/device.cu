#include "kernels.h"
#include <cuComplex.h>

/*
 //TODO
__device__ void complexMatMulCuComplexDevice(int N, cuComplex* A, cuComplex* B, cuComplex* C) {
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
}*/

