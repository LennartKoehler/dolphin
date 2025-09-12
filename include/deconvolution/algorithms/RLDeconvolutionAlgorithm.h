#pragma once

#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "deconvolution/backend/IDeconvolutionBackend.h"
#include <memory>
#include <iostream>

class RLDeconvolutionAlgorithm : public DeconvolutionAlgorithm {
public:
    // Constructor that takes a backend parameter
    explicit RLDeconvolutionAlgorithm(std::shared_ptr<IDeconvolutionBackend> backend);
    
    void deconvolve(fftw_complex* H, fftw_complex* g, fftw_complex* f, const RectangleShape& cubeShape) override;
    void configure(const DeconvolutionConfig& config) override;

private:
    int iterations;
    std::shared_ptr<IDeconvolutionBackend> backend;  // Backend pointer for backend-agnostic operations
    
    // Helper methods for Richardson-Lucy algorithm using backend

    bool copyComplexArray(const fftw_complex* source, fftw_complex* destination, int size);
    bool validateComplexArray(fftw_complex* array, int size, const std::string& context);
    bool allocateCPUArray(fftw_complex*& array, int size);
    void deallocateCPUArray(fftw_complex* array);

};
