#pragma once
#include "IDeconvolutionBackend.h"

class CPUBackend : public IDeconvolutionBackend{
    virtual void preprocess() override;
    virtual void postprocess() override;



    virtual void reorderLayers(fftw_complex* data, int width, int height, int depth) override;
    virtual void visualizeFFT(fftw_complex* data, int width, int height, int depth) override;

    // FFT functions
    virtual void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth) override;
    virtual void convertFFTWComplexToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) override;

    virtual void convertFFTWComplexRealToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) override;
    virtual void convertFFTWComplexImgToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth) override;


    virtual void padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth) override;

    virtual void forwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth) override;
    virtual void backwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth) override;

    virtual void octantFourierShift(fftw_complex* data, int width, int height, int depth) override;
    virtual void inverseQuadrantShift(fftw_complex* data, int width, int height, int depth) override;
    virtual void quadrantShiftMat(cv::Mat& magI) override;

    virtual void complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    virtual void complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) override;
    virtual void complexAddition(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    virtual void scalarMultiplication(fftw_complex* a, double scalar, fftw_complex* result, int size) override;
    virtual void complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    virtual void complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) override;

    virtual void calculateLaplacianOfPSF(fftw_complex* psf, fftw_complex* laplacian, int width, int height, int depth) override;

    virtual void normalizeImage(fftw_complex* resultImage, int size, double epsilon) override;
    virtual void rescaledInverse(fftw_complex* data, double cubeVolume) override;
    virtual void saveInterimImages(fftw_complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i) override;

    virtual void gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth) override;
    virtual void gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth) override;
    virtual void gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth) override;
    virtual void computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth) override;
    virtual void normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon) override;
};