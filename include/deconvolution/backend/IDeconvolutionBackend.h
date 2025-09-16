#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>

#include <fftw3.h>

struct RectangleShape{
    int width;
    int height;
    int depth;
    int volume;
};
struct FFTWData{
    fftw_complex* data;
    RectangleShape size;
};
struct InputData{
    FFTWData H;
    FFTWData g;
    FFTWData f;
};


// should split into memory management and fftw backend?
class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};

    virtual void preprocess() = 0;
    virtual void postprocess() = 0;

    // data management
    virtual void allocateMemoryOnDevice(FFTWData& data) = 0;
    virtual void initializeFFTPlans(const RectangleShape& cube) = 0;
    virtual bool isOnDevice(void* data) = 0;
    virtual FFTWData moveDataToDevice(const FFTWData& srcdata) = 0;
    virtual FFTWData moveDataFromDevice(const FFTWData& srcdata) = 0;
    virtual FFTWData copyData(const FFTWData& srcdata) = 0;
    virtual FFTWData allocateMemoryOnDevice(const RectangleShape& shape) = 0;
    virtual void freeMemoryOnDevice(FFTWData& data) = 0;

    //
    virtual void reorderLayers(FFTWData& data) = 0;
    virtual void visualizeFFT(const FFTWData& data) = 0;

    // FFT functions
    virtual void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, FFTWData& output) = 0;
    virtual void convertFFTWComplexToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) = 0;

    virtual void convertFFTWComplexRealToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) = 0;
    virtual void convertFFTWComplexImgToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) = 0;

    virtual void padPSF(const FFTWData& psf, FFTWData& padded_psf, const RectangleShape& target_size) = 0;

    virtual void forwardFFT(const FFTWData& in, FFTWData& out) = 0;
    virtual void backwardFFT(const FFTWData& in, FFTWData& out) = 0;

    virtual void octantFourierShift(FFTWData& data) = 0;
    virtual void inverseQuadrantShift(FFTWData& data) = 0;
    virtual void quadrantShiftMat(cv::Mat& magI) = 0;

    virtual void complexMultiplication(const FFTWData& a, const FFTWData& b, FFTWData& result) = 0;
    virtual void complexDivision(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) = 0;
    virtual void complexAddition(const FFTWData& a, const FFTWData& b, FFTWData& result) = 0;
    virtual void scalarMultiplication(const FFTWData& a, double scalar, FFTWData& result) = 0;
    virtual void complexMultiplicationWithConjugate(const FFTWData& a, const FFTWData& b, FFTWData& result) = 0;
    virtual void complexDivisionStabilized(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) = 0;

    virtual void calculateLaplacianOfPSF(const FFTWData& psf, FFTWData& laplacian) = 0;

    virtual void normalizeImage(FFTWData& resultImage, double epsilon) = 0;
    virtual void rescaledInverse(FFTWData& data, double cubeVolume) = 0;
    virtual void saveInterimImages(const FFTWData& resultImage, int gridNum, int channel_z, int i) = 0;

    virtual void gradientX(const FFTWData& image, FFTWData& gradX) = 0;
    virtual void gradientY(const FFTWData& image, FFTWData& gradY) = 0;
    virtual void gradientZ(const FFTWData& image, FFTWData& gradZ) = 0;
    virtual void computeTV(double lambda, const FFTWData& gx, const FFTWData& gy, const FFTWData& gz, FFTWData& tv) = 0;
    virtual void normalizeTV(FFTWData& gradX, FFTWData& gradY, FFTWData& gradZ, double epsilon) = 0;

protected:
    fftw_plan forwardPlan;
    fftw_plan backwardPlan;
    fftw_complex* planMemory;
    std::vector<fftw_complex*> allocatedCPUMemory;
    bool plansInitialized = false;
};