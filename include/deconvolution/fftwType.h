#ifndef FFTW_TYPES_H
#define FFTW_TYPES_H

#ifdef CUDA_AVAILABLE
    // Use cuFFTW types when CUDA is available
    #include <cufftw.h>
    // fftw_complex is already defined by cufftw.h
#else
    // Use regular FFTW types when CUDA is not available
    #include <fftw3.h>
    // fftw_complex is already defined by fftw3.h
#endif

#endif // FFTW_TYPES_H