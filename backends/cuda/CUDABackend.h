#pragma once
#include "backend/IDeconvolutionBackend.h"
#include "backend/IBackendMemoryManager.h"
#include <cufftw.h>
#include <CUBE.h>
#include <cuda_runtime.h>

class CUDABackendMemoryManager : public IBackendMemoryManager{
public:
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override; // for gpu these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override; // for gpu these are copy operations
    void freeMemoryOnDevice(ComplexData& data) const override;
    size_t getAvailableMemory() const override;

};

class CUDADeconvolutionBackend : public IDeconvolutionBackend{
public:
    CUDADeconvolutionBackend();
    ~CUDADeconvolutionBackend() override;

    // Core processing functions
    void init(const RectangleShape& shape) override;
    void cleanup() override;

    // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) const override;
    void backwardFFT(const ComplexData& in, ComplexData& out) const override;

    // Shift functions
    void octantFourierShift(ComplexData& data) const override;
    void inverseQuadrantShift(ComplexData& data) const override;

    // Complex arithmetic functions
    void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const override;
    void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) const override;
    void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const override;
    void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const override;

    // Specialized functions
    void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const override;
    void normalizeImage(ComplexData& resultImage, double epsilon) const override;
    void rescaledInverse(ComplexData& data, double cubeVolume) const override;
    // void saveInterimImages(const ComplexData& resultImage, int gridNum, int channel_z, int i) const override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) const override;
    void gradientY(const ComplexData& image, ComplexData& gradY) const override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) const override;
    void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const override;

    // Layer and visualization functions
    // void reorderLayers(ComplexData& data) override;
    // void visualizeFFT(const ComplexData& data) override;

    // Conversion functions
    // void readCVMat(const std::vector<cv::Mat>& input, ComplexData& output) override;
    // void convertFFTWComplexToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexRealToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
    // void convertFFTWComplexImgToCVMatVector(const ComplexData& input, std::vector<cv::Mat>& output) override;
   void hasNAN(const ComplexData& data) const override;



private:
    void initializeFFTPlans(const RectangleShape& cube);
    // Helper functions
    void destroyFFTPlans();
    cufftHandle forwardPlan;
    cufftHandle backwardPlan;
    int numThreads_ = 1; // For CPU fallback operations if needed
};