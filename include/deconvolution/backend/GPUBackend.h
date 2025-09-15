#pragma once
#include "IDeconvolutionBackend.h"
#include <unordered_map>
#include "CUBE.h"

typedef size_t PSFIndex;

class GPUBackend : public IDeconvolutionBackend{
public:
    GPUBackend();
    ~GPUBackend();

    // Core processing functions
    void preprocess() override;
    void postprocess() override;

    // Data management
    std::unordered_map<PSFIndex, FFTWData>& movePSFstoGPU(std::unordered_map<PSFIndex, FFTWData>& psfMap); // Updated to use FFTWData
    void allocateMemoryOnDevice(FFTWData& data) override;
    FFTWData allocateMemoryOnDevice(const RectangleShape& shape) override;
    bool isOnDevice(void* data) override;
    FFTWData copyData(const FFTWData& srcdata) override;
    FFTWData moveDataToDevice(const FFTWData& srcdata) override; // for gpu these are copy operations
    FFTWData moveDataFromDevice(const FFTWData& srcdata) override; // for gpu these are copy operations
    void freeMemoryOnDevice(FFTWData& data) override;

    // Layer and visualization functions
    void reorderLayers(FFTWData& data) override;
    void visualizeFFT(const FFTWData& data) override;

    // Conversion functions
    void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, FFTWData& output) override;
    void convertFFTWComplexToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) override;
    void convertFFTWComplexRealToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) override;
    void convertFFTWComplexImgToCVMatVector(const FFTWData& input, std::vector<cv::Mat>& output) override;

    // PSF and FFT functions
    void padPSF(const FFTWData& psf, FFTWData& padded_psf, const RectangleShape& target_size) override;
    void forwardFFT(const FFTWData& in, FFTWData& out) override;
    void backwardFFT(const FFTWData& in, FFTWData& out) override;

    // Shift functions
    void octantFourierShift(FFTWData& data) override;
    void inverseQuadrantShift(FFTWData& data) override;
    void quadrantShiftMat(cv::Mat& magI) override;

    // Complex arithmetic functions
    void complexMultiplication(const FFTWData& a, const FFTWData& b, FFTWData& result) override;
    void complexDivision(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) override;
    void complexAddition(const FFTWData& a, const FFTWData& b, FFTWData& result) override;
    void scalarMultiplication(const FFTWData& a, double scalar, FFTWData& result) override;
    void complexMultiplicationWithConjugate(const FFTWData& a, const FFTWData& b, FFTWData& result) override;
    void complexDivisionStabilized(const FFTWData& a, const FFTWData& b, FFTWData& result, double epsilon) override;

    // Specialized functions
    void calculateLaplacianOfPSF(const FFTWData& psf, FFTWData& laplacian) override;
    void normalizeImage(FFTWData& resultImage, double epsilon) override;
    void rescaledInverse(FFTWData& data, double cubeVolume) override;
    void saveInterimImages(const FFTWData& resultImage, int gridNum, int channel_z, int i) override;

    // Gradient and TV functions
    void gradientX(const FFTWData& image, FFTWData& gradX) override;
    void gradientY(const FFTWData& image, FFTWData& gradY) override;
    void gradientZ(const FFTWData& image, FFTWData& gradZ) override;
    void computeTV(double lambda, const FFTWData& gx, const FFTWData& gy, const FFTWData& gz, FFTWData& tv) override;
    void normalizeTV(FFTWData& gradX, FFTWData& gradY, FFTWData& gradZ, double epsilon) override;

private:
    void initializeFFTPlans(const RectangleShape& cube) override;
    // Helper functions
    void destroyFFTPlans();
};