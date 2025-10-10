#pragma once

#include <string>
#include <stdexcept>
#include <mutex>
#include "ComplexData.h"





// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

// be sure that implementations of this are threadsafe
class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};

    // Core functions - still pure virtual (must implement)
    virtual void init(const RectangleShape& shape) = 0;
    virtual void cleanup() = 0;
    
    virtual void initializeFFTPlans(const RectangleShape& cube) {
        NOT_IMPLEMENTED(initializeFFTPlans);
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


    virtual bool plansInitialized(){ 
        std::unique_lock lock(backendMutex);
        return plansInitialized_; }



protected:
    bool plansInitialized_ = false;
    std::mutex backendMutex;
};

#undef NOT_IMPLEMENTED