#pragma once
#include "backend/IDeconvolutionBackend.h"
#include "backend/IBackendMemoryManager.h"
#include <fftw3.h>

class CPUBackendMemoryManager : public IBackendMemoryManager{
public:
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) override;
    void allocateMemoryOnDevice(ComplexData& data) override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) override;
    bool isOnDevice(void* data) override;
    ComplexData copyData(const ComplexData& srcdata) override;
    ComplexData moveDataToDevice(const ComplexData& srcdata) override; // for cpu these are copy operations
    ComplexData moveDataFromDevice(const ComplexData& srcdata) override; // for cpu these are copy operations
    void freeMemoryOnDevice(ComplexData& data) override;
    size_t getAvailableMemory() override; 

};

class CPUDeconvolutionBackend : public IDeconvolutionBackend{
public:
    CPUDeconvolutionBackend();
    ~CPUDeconvolutionBackend() override;

    // Core processing functions
    void init(const RectangleShape& shape) override;
    void cleanup() override;

    // FFT functions
    void forwardFFT(const ComplexData& in, ComplexData& out) override;
    void backwardFFT(const ComplexData& in, ComplexData& out) override;

    // Shift functions
    void octantFourierShift(ComplexData& data) override;
    void inverseQuadrantShift(ComplexData& data) override;

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

    // Debug functions
    void hasNAN(const ComplexData& data) override;

    // Layer and visualization functions
    void reorderLayers(ComplexData& data) override;

    // Gradient and TV functions
    void gradientX(const ComplexData& image, ComplexData& gradX) override;
    void gradientY(const ComplexData& image, ComplexData& gradY) override;
    void gradientZ(const ComplexData& image, ComplexData& gradZ) override;
    void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) override;
    void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) override;


private:
    void initializeFFTPlans(const RectangleShape& cube);
    void destroyFFTPlans();
    fftw_plan forwardPlan;
    fftw_plan backwardPlan;
};