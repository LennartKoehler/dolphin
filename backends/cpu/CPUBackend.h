#pragma once
#include "deconvolution/IDeconvolutionBackend.h"
#include <fftw3.h>


class CPUBackend : public IDeconvolutionBackend{
public:
    CPUBackend();
    ~CPUBackend() override;

    // Core processing functions
    void init(const RectangleShape& shape) override;
    void postprocess() override;
    virtual std::shared_ptr<IDeconvolutionBackend> clone() const override ;

    // Data management
    void allocateMemoryOnDevice(ComplexData& data) override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) override;
    bool isOnDevice(void* data) override;
    void memCopy(const ComplexData& src, ComplexData& dest) override;
    ComplexData copyData(const ComplexData& srcdata) override;
    ComplexData moveDataToDevice(const ComplexData& srcdata) override;
    ComplexData moveDataFromDevice(const ComplexData& srcdata) override;
    void freeMemoryOnDevice(ComplexData& data) override;

    void hasNAN(const ComplexData& data) override;
    // Layer and visualization functions
    void reorderLayers(ComplexData& data) override;
    // void visualizeFFT(const ComplexData& data) override;

    // Conversion functions
    // void readCVMat(const std::vector<cv::Mat>& input, ComplexData& output) override;
    // void convertFFTWComplexToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;

    // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) override;
    void backwardFFT(const ComplexData& in, ComplexData& out) override;

    // Shift functions
    void octantFourierShift(ComplexData& data) override;
    void inverseQuadrantShift(ComplexData& data) override;
    void quadrantShiftMat(cv::Mat& magI) override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) override;
    void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) override;

    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) override;
    void normalizeImage(ComplexData& resultImage, double epsilon) override;
    void rescaledInverse(ComplexData& data, double cubeVolume) override;
    // void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) override;
    void gradientY(const ComplexData& image, ComplexData& gradY) override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) override;
    void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) override;

    // Memory usage function
    size_t getWorkSize() const override;
    RectangleShape getWorkShape() const override;
    size_t getAvailableMemory() override;


private:
    void initializeFFTPlans(const RectangleShape& cube) override;
    void destroyFFTPlans();

    fftw_plan forwardPlan;
    fftw_plan backwardPlan;
    size_t workSize;
    RectangleShape workShape;
};