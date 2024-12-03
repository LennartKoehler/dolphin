#pragma once
#include <cuComplex.h>
#include <cufft.h>
#include <cufftw.h>

namespace CUBE_UTL_INFO{
    // Print information
    void printDeviceProperties();
    void printFirstElem(fftw_complex* mat);
    void printSpecificElem(int index, fftw_complex* mat);
    void printRandomElem(int Nx, int Ny, int Nz, fftw_complex* mat);

}

namespace CUBE_UTL_CHECK {
    // Check Mat
    void checkUniformity(int Nx, int Ny, int Nz,fftw_complex* mat);
    void displayHeatmap(int Nx, int Ny, int Nz,const fftw_complex* data);
    bool checkOctantFourierShift(int Nx, int Ny, int Nz, fftw_complex* original, fftw_complex* shifted);
    void printFftwComplexValueFromDevice(int idx, fftw_complex* fftwArr);
    void printCufftComplexValueFromDevice(int idx, cufftComplex* cuArr);
}

namespace CUBE_UTL_INIT_MAT {
    // Mat initialization
    void createFftwUniformMat(int Nx, int Ny, int Nz, fftw_complex* mat);
    void createFftwRandomMat(int Nx, int Ny, int Nz, fftw_complex* mat);
    void createFftwSphereMat(int Nx, int Ny, int Nz, fftw_complex* mat);
}

namespace CUBE_UTL_COPY {
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src);
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz, fftw_complex* dest, fftw_complex* src);
}

namespace CUBE_UTL_CONVERT {
    void convertFftwToCuComplexOnDevice(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr);
    void convertCuToFftwComplexOnHost(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr);
    void convertFftwToCufftComplexOnDevice(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cuArr);
    void convertCufftToFftwComplexOnHost(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cuArr);
}
