#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>

#include <fftw3.h>


class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};

    virtual void preprocess() = 0;
    virtual void postprocess() = 0;




    virtual void reorderLayers(fftw_complex* data, int width, int height, int depth) = 0;
    virtual void visualizeFFT(fftw_complex* data, int width, int height, int depth) = 0;

    // FFT functions
    virtual void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth) = 0;
    virtual void convertFFTWComplexToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) = 0;

    virtual void convertFFTWComplexRealToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) = 0;
    virtual void convertFFTWComplexImgToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) = 0;


    virtual void padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth) = 0;

    virtual void forwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth) = 0;
    virtual void backwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth) = 0;

    virtual void octantFourierShift(fftw_complex* data, int width, int height, int depth) = 0;
    virtual void inverseQuadrantShift(fftw_complex* data, int width, int height, int depth) = 0;
    virtual void quadrantShiftMat(cv::Mat& magI) = 0;

    virtual void complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) = 0;
    virtual void complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) = 0;
    virtual void complexAddition(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) = 0;
    virtual void scalarMultiplication(fftw_complex* a, double scalar, fftw_complex* result, int size) = 0;
    virtual void complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) = 0;
    virtual void complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) = 0;

    virtual void calculateLaplacianOfPSF(fftw_complex* psf, fftw_complex* laplacian, int width, int height, int depth) = 0;

    virtual void normalizeImage(fftw_complex* resultImage, int size, double epsilon) = 0;
    virtual void rescaledInverse(fftw_complex* data, double cubeVolume) = 0;
    virtual void saveInterimImages(fftw_complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i) = 0;

    virtual void gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth) = 0;
    virtual void gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth) = 0;
    virtual void gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth) = 0;
    virtual void computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth) = 0;
    virtual void normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon) = 0;
};