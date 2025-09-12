#pragma once
#include "../DeconvolutionConfig.h"
#include "../DeconvolutionProcessor.h" // to include rectangleshape
class fftw_complex;

class DeconvolutionAlgorithm{
public:
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void deconvolve(fftw_complex* H, fftw_complex* g, fftw_complex* f, const RectangleShape& cubeShape) = 0;

protected:
    double complexDivisionEpsilon;
};