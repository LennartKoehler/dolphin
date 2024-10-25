#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "PSF.h"
#include <fftw3.h>

class BaseDeconvolutionAlgorithm {
public:
    virtual ~BaseDeconvolutionAlgorithm(){cleanup();}
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void algorithm(Hyperstack& data, int channel_num) = 0;

    Hyperstack deconvolve(Hyperstack& data, std::vector<PSF>& psfs);
    bool preprocess(Channel& channel, std::vector<PSF>& psfs);
    bool postprocess(double epsilon);
    void cleanup();

protected:
    // Configuration
    double epsilon;
    bool grid;
    int borderType;
    int psfSafetyBorder;
    int cubeSize;

    // Image handling and fftw
    std::vector<cv::Mat> mergedVolume;
    std::vector<std::vector<cv::Mat>> gridImages;
    fftw_plan forwardPlan, backwardPlan = nullptr;
    fftw_complex *paddedH, *paddedH_2, *fftwPlanMem = nullptr;

    // Image info
    int originalImageWidth, originalImageHeight, originalImageDepth;

    // Grid/cube arrangement
    int totalGridNum = 1;
    int cubesPerX, cubesPerY, cubesPerZ, cubesPerLayer = 0;
    int cubeVolume, cubeWidth, cubeHeight, cubeDepth, cubePadding;
    std::vector<int> secondpsflayers, secondpsfcubes;
};
