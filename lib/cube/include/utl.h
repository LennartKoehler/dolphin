#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>

// Print information
void printDeviceProperties();
void printFirstElem(fftw_complex* mat);
void printSpecificElem(fftw_complex* mat, int index);
void printRandomElem(fftw_complex* mat, int N);

// Check Mat
void checkUniformity(fftw_complex* mat, int N);
void displayHeatmap(const fftw_complex* data, int N);
bool checkOctantFourierShift(int N, fftw_complex* original, fftw_complex* shifted);

// Mat initialization
void createFftwUniformMat(int N, fftw_complex* mat);
void createFftwRandomMat(int N, fftw_complex* mat);
void createFftwSphereMat(int N, fftw_complex* mat);

// Copying fftw_complex datatype to GPU
void copyDataFromHostToDevice(fftw_complex* dest, fftw_complex* src, int matrixSize);
void copyDataFromDeviceToHost(fftw_complex* dest, fftw_complex* src, int matrixSize);

// Conversions
void convertFftwToCuComplexOnDevice(fftw_complex* fftwArr, cuComplex* cuArr, int N);
void convertCuToFftwComplexOnHost(fftw_complex* fftwArr, cuComplex* cuArr, int N);
void convertFftwToCufftComplexOnDevice(fftw_complex* fftwArr, cufftComplex* cuArr, int N);
void convertCufftToFftwComplexOnHost(fftw_complex* fftwArr, cufftComplex* cuArr, int N);

