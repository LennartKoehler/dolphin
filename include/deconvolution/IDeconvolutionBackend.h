#pragma once

#include <string>
#include <vector>
#include <complex>
#include <stdexcept>
#include <typeinfo>
#include <opencv2/core/mat.hpp>
#include "complexType.h"
#include "RectangleShape.h"

struct ComplexData{
    complex* data;
    RectangleShape size;
};
struct InputData{
    ComplexData H;
    ComplexData g;
    ComplexData f;
};

// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

// should split into memory management and fftw backend?
class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};

    // Core functions - still pure virtual (must implement)
    virtual void init(const RectangleShape& shape) = 0;
    virtual void postprocess() = 0;
    virtual std::shared_ptr<IDeconvolutionBackend> clone() const = 0;
    virtual size_t getMemoryUsage() const = 0;

    // Data management - provide default implementations
    virtual void allocateMemoryOnDevice(ComplexData& data) {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }
    
    virtual void initializeFFTPlans(const RectangleShape& cube) {
        NOT_IMPLEMENTED(initializeFFTPlans);
    }
    
    virtual bool isOnDevice(void* data) {
        NOT_IMPLEMENTED(isOnDevice);
    }
    
    virtual ComplexData moveDataToDevice(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(moveDataToDevice);
    }
    
    virtual ComplexData moveDataFromDevice(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(moveDataFromDevice);
    }
    
    virtual void memCopy(const ComplexData& srcData, ComplexData& destdata) {
        NOT_IMPLEMENTED(memCopy);
    }
    
    virtual ComplexData copyData(const ComplexData& srcdata) {
        NOT_IMPLEMENTED(copyData);
    }
    
    virtual ComplexData allocateMemoryOnDevice(const RectangleShape& shape) {
        NOT_IMPLEMENTED(allocateMemoryOnDevice);
    }
    
    virtual void freeMemoryOnDevice(ComplexData& data) {
        NOT_IMPLEMENTED(freeMemoryOnDevice);
    }

    // Debug functions
    virtual void hasNAN(const ComplexData& data){
        NOT_IMPLEMENTED(hasNAN);
    }

    // Data manipulation
    virtual void reorderLayers(ComplexData& data) {
        NOT_IMPLEMENTED(reorderLayers);
    }

    // FFT functions
    virtual void forwardFFT(const ComplexData& in, ComplexData& out) {
        NOT_IMPLEMENTED(forwardFFT);
    }
    
    virtual void backwardFFT(const ComplexData& in, ComplexData& out) {
        NOT_IMPLEMENTED(backwardFFT);
    }

    // Shift operations
    virtual void octantFourierShift(ComplexData& data) {
        NOT_IMPLEMENTED(octantFourierShift);
    }
    
    virtual void inverseQuadrantShift(ComplexData& data) {
        NOT_IMPLEMENTED(inverseQuadrantShift);
    }
    
    virtual void quadrantShiftMat(cv::Mat& magI) {
        NOT_IMPLEMENTED(quadrantShiftMat);
    }

    // Complex arithmetic operations
    virtual void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) {
        NOT_IMPLEMENTED(complexMultiplication);
    }
    
    virtual void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
        NOT_IMPLEMENTED(complexDivision);
    }
    
    virtual void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) {
        NOT_IMPLEMENTED(complexAddition);
    }
    
    virtual void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) {
        NOT_IMPLEMENTED(scalarMultiplication);
    }
    
    virtual void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) {
        NOT_IMPLEMENTED(complexMultiplicationWithConjugate);
    }
    
    virtual void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) {
        NOT_IMPLEMENTED(complexDivisionStabilized);
    }

    // Advanced operations
    virtual void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) {
        NOT_IMPLEMENTED(calculateLaplacianOfPSF);
    }

    virtual void normalizeImage(ComplexData& resultImage, double epsilon) {
        NOT_IMPLEMENTED(normalizeImage);
    }
    
    virtual void rescaledInverse(ComplexData& data, double cubeVolume) {
        NOT_IMPLEMENTED(rescaledInverse);
    }

    // Gradient operations
    virtual void gradientX(const ComplexData& image, ComplexData& gradX) {
        NOT_IMPLEMENTED(gradientX);
    }
    
    virtual void gradientY(const ComplexData& image, ComplexData& gradY) {
        NOT_IMPLEMENTED(gradientY);
    }
    
    virtual void gradientZ(const ComplexData& image, ComplexData& gradZ) {
        NOT_IMPLEMENTED(gradientZ);
    }
    
    virtual void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) {
        NOT_IMPLEMENTED(computeTV);
    }
    
    virtual void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) {
        NOT_IMPLEMENTED(normalizeTV);
    }

    virtual bool isInitialized(){ return plansInitialized; }

protected:
    bool plansInitialized = false;
};

#undef NOT_IMPLEMENTED