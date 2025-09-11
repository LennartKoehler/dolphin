#pragma once
#include <string>
#include "../DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"

#ifdef CUDA_AVAILABLE
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

class BaseDeconvolutionAlgorithm{
public:
    Hyperstack run(Hyperstack& data, std::vector<PSF>& psfs);

    virtual ~BaseDeconvolutionAlgorithm(){cleanup();}
    virtual void configure(DeconvolutionConfig config);
    virtual void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) = 0;

    Hyperstack deconvolve(Hyperstack& data, std::vector<PSF>& psfs);
    bool preprocess(Channel& channel, std::vector<PSF>& psfs);
    bool postprocess(double epsilon, ImageMetaData& meta);
    void cleanup();

protected:
    // Configuration
    DeconvolutionConfig config;
    
    // Common algorithm parameters (should be moved to derived base class)
    double epsilon = 1e-6;      // Minimum threshold for values
    bool time = false;            // Whether to measure and display timing
    bool saveSubimages = false;   // Whether to save subimage results
    bool grid = false;            // Whether to use grid processing
    std::string gpu = "";        // GPU API selection ("cuda", "", "opencl")

    // Image handling and fftw
    std::vector<cv::Mat> mergedVolume;
    std::vector<std::vector<cv::Mat>> gridImages;
    fftw_plan forwardPlan  = nullptr;
    fftw_plan backwardPlan = nullptr;

    fftw_complex *fftwPlanMem = nullptr;
    std::vector<fftw_complex*> paddedHs;
#ifdef CUDA_AVAILABLE
    fftw_complex *d_paddedH = nullptr;
    fftw_complex *d_paddedH_2 = nullptr;
    std::vector<fftw_complex*> d_paddedHs;
#endif

    // Image info
    int originalImageWidth, originalImageHeight, originalImageDepth;

    // Grid/cube arrangement
    int totalGridNum = 1;
    int cubesPerX, cubesPerY, cubesPerZ, cubesPerLayer = 0;
    int cubeVolume, cubeWidth, cubeHeight, cubeDepth, cubePadding;
    std::vector<int> secondpsflayers, secondpsfcubes;
    std::vector<std::vector<int>> cubePSFMap;
    std::vector<std::vector<int>> layerPSFMap;


    std::string gpu = "";
};
