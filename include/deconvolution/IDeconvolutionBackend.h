#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>
#include "complexType.h"


struct RectangleShape{
    int width;
    int height;
    int depth;
    int volume;
};
struct ComplexData{
    complex* data;
    RectangleShape size;
};
struct InputData{
    ComplexData H;
    ComplexData g;
    ComplexData f;
};


// should split into memory management and fftw backend?
class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};

    virtual void init(const RectangleShape& shape) = 0;
    virtual void postprocess() = 0;
    virtual std::shared_ptr<IDeconvolutionBackend> clone() const = 0;

    
    // data management
    virtual void allocateMemoryOnDevice(ComplexData& data) = 0;
    virtual void initializeFFTPlans(const RectangleShape& cube) = 0;
    virtual bool isOnDevice(void* data) = 0;
    virtual ComplexData moveDataToDevice(const ComplexData& srcdata) = 0;
    virtual ComplexData moveDataFromDevice(const ComplexData& srcdata) = 0;
    virtual void memCopy(ComplexData& srcData, ComplexData& destdata) = 0;
    virtual ComplexData copyData(const ComplexData& srcdata) = 0;
    virtual ComplexData allocateMemoryOnDevice(const RectangleShape& shape) = 0;
    virtual void freeMemoryOnDevice(ComplexData& data) = 0;

    //
    virtual void reorderLayers(ComplexData& data) = 0;
    // virtual void visualizeFFT(const ComplexData& data) = 0;

    // FFT functions
    // virtual void convertCVMatVectorToComplex(const std::vector<cv::Mat>& input, const RectangleShape& shape);
    // virtual void convertFFTWComplexToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output);

    // virtual void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output);
    // virtual void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output);

    // virtual void padPSF(const ComplexData& psf, ComplexData& padded_psf) = 0;

    virtual void forwardFFT(const ComplexData& in, ComplexData& out) = 0;
    virtual void backwardFFT(const ComplexData& in, ComplexData& out) = 0;

    virtual void octantFourierShift(ComplexData& data) = 0;
    virtual void inverseQuadrantShift(ComplexData& data) = 0;
    virtual void quadrantShiftMat(cv::Mat& magI) = 0;

    virtual void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) = 0;
    virtual void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) = 0;
    virtual void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) = 0;
    virtual void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) = 0;
    virtual void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) = 0;
    virtual void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) = 0;

    virtual void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) = 0;

    virtual void normalizeImage(ComplexData& resultImage, double epsilon) = 0;
    virtual void rescaledInverse(ComplexData& data, double cubeVolume) = 0;
    // virtual void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) = 0;

    virtual void gradientX(const ComplexData& image, ComplexData& gradX) = 0;
    virtual void gradientY(const ComplexData& image, ComplexData& gradY) = 0;
    virtual void gradientZ(const ComplexData& image, ComplexData& gradZ) = 0;
    virtual void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) = 0;
    virtual void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) = 0;

    virtual bool isInitialized(){ return plansInitialized; }

protected:
    bool plansInitialized = false;
};

