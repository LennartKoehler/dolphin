#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fftw3.h>

// Print information
void printDeviceProperties();
void printFirstElem(fftw_complex* mat);
void printSpecificElem(int index, fftw_complex* mat);
void printRandomElem(int Nx, int Ny, int Nz, fftw_complex* mat);
void printFftwComplexValueFromDevice(int idx, fftw_complex* fftwArr);
void printCufftComplexValueFromDevice(int idx, cufftComplex* cuArr);

// Check Mat
void checkUniformity(int Nx, int Ny, int Nz,fftw_complex* mat);
void displayHeatmap(int Nx, int Ny, int Nz,const fftw_complex* data);
bool checkOctantFourierShift(int Nx, int Ny, int Nz, fftw_complex* original, fftw_complex* shifted);

// Mat initialization
void createFftwUniformMat(int Nx, int Ny, int Nz, fftw_complex* mat);
void createFftwRandomMat(int Nx, int Ny, int Nz, fftw_complex* mat);
void createFftwSphereMat(int Nx, int Ny, int Nz, fftw_complex* mat);

// Copying fftw_complex datatype to GPU
void copyDataFromHostToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src);
void copyDataFromDeviceToHost(int Nx, int Ny, int Nz, fftw_complex* dest, fftw_complex* src);

// Conversions
void convertFftwToCuComplexOnDevice(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr);
void convertCuToFftwComplexOnHost(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr);
void convertFftwToCufftComplexOnDevice(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cuArr);
void convertCufftToFftwComplexOnHost(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cuArr);

