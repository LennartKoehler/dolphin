#pragma once
#include "IDeconvolutionBackend.h"
#include <unordered_map>
#include "CUBE.h"

class GPUBackend : public IDeconvolutionBackend{
public:
    GPUBackend();
    ~GPUBackend();

    // Core processing functions
    void preprocess() override;
    void postprocess() override;

    // Layer and visualization functions
    void reorderLayers(fftw_complex* data, int width, int height, int depth) override;
    void visualizeFFT(fftw_complex* data, int width, int height, int depth) override;

    // Conversion functions
    void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth) override;
    void convertFFTWComplexToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) override;
    void convertFFTWComplexRealToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) override;
    void convertFFTWComplexImgToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) override;

    // PSF and FFT functions
    void padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth) override;
    void forwardFFT(fftw_complex* in, fftw_complex* out, int imageDepth, int imageHeight, int imageWidth) override;
    void backwardFFT(fftw_complex* in, fftw_complex* out, int imageDepth, int imageHeight, int imageWidth) override;

    // Shift functions
    void octantFourierShift(fftw_complex* data, int width, int height, int depth) override;
    void inverseQuadrantShift(fftw_complex* data, int width, int height, int depth) override;
    void quadrantShiftMat(cv::Mat& magI) override;

    // Complex arithmetic functions
    void complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    void complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) override;
    void complexAddition(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    void scalarMultiplication(fftw_complex* a, double scalar, fftw_complex* result, int size) override;
    void complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) override;
    void complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) override;

    // Specialized functions
    void calculateLaplacianOfPSF(fftw_complex* psf, fftw_complex* laplacian, int width, int height, int depth) override;
    void normalizeImage(fftw_complex* resultImage, int size, double epsilon) override;
    void rescaledInverse(fftw_complex* data, double cubeVolume) override;
    void saveInterimImages(fftw_complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i) override;

    // Gradient and TV functions
    void gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth) override;
    void gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth) override;
    void gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth) override;
    void computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth) override;
    void normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon) override;

private:
    cufftHandle forwardPlan;
    cufftHandle backwardPlan;
    bool plansInitialized;
    
    // Helper functions
    void initializeFFTPlans(int width, int height, int depth);
    void destroyFFTPlans();
    std::unordered_map<PSFIndex, PSFfftw*>& movePSFstoGPU(std::unordered_map<PSFIndex, PSFfftw*>& psfMap);
};