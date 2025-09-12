#pragma once

class fftw_complex;

class DeconvolutionAlgorithm{
public:
    virtual void configure(DeconvolutionConfig config) = 0;
    virtual void deconvolve(fftw_complex* H, fftw_complex* g, fftw_complex* f) = 0;
};