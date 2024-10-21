#pragma once

#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "PSF.h"
#include <fftw3.h>




class BaseDeconvolutionAlgorithm {
public:
    virtual ~BaseDeconvolutionAlgorithm(){cleanup();}
    virtual Hyperstack deconvolve(Hyperstack& data, std::vector<PSF>& psfs) = 0;
    virtual void configure(const DeconvolutionConfig& config) = 0;

    bool preprocess(Channel& channel, std::vector<PSF>& psfs);
    bool postprocess(Hyperstack& data, double epsilon);
    void cleanup();


protected:
    std::vector<std::vector<cv::Mat>> gridImages;
    std::vector<cv::Mat> mergedVolume;
    fftw_complex *paddedH = nullptr;
    fftw_complex *paddedH_2 = nullptr;
    fftw_complex *fftwPlanMem = nullptr;
    fftw_plan forwardPlan = nullptr;
    fftw_plan backwardPlan = nullptr;
    int cubeVolume, cubeWidth, cubeHeight, cubeDepth, cubePadding;
    bool grid;
    int borderType;
    int psfSafetyBorder;
    int cubeSize;
    std::vector<int> secondpsflayers;
    std::vector<int> secondpsfcubes;
    int originalImageWidth;
    int originalImageHeight;
    int originalImageDepth;

};
