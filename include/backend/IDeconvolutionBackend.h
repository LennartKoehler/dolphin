/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
    

    // Debug functions
    virtual void hasNAN(const ComplexData& data) const {
        NOT_IMPLEMENTED(hasNAN);
    }

    // Data manipulation
    virtual void reorderLayers(ComplexData& data) const {
        NOT_IMPLEMENTED(reorderLayers);
    }

    // FFT functions
    virtual void forwardFFT(const ComplexData& in, ComplexData& out) const {
        NOT_IMPLEMENTED(forwardFFT);
    }
    
    virtual void backwardFFT(const ComplexData& in, ComplexData& out) const {
        NOT_IMPLEMENTED(backwardFFT);
    }

    // Shift operations
    virtual void octantFourierShift(ComplexData& data) const {
        NOT_IMPLEMENTED(octantFourierShift);
    }
    
    virtual void inverseQuadrantShift(ComplexData& data) const {
        NOT_IMPLEMENTED(inverseQuadrantShift);
    }
    
    // Complex arithmetic operations
    virtual void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
        NOT_IMPLEMENTED(complexMultiplication);
    }
    
    virtual void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
        NOT_IMPLEMENTED(complexDivision);
    }
    
    virtual void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
        NOT_IMPLEMENTED(complexAddition);
    }
    
    virtual void scalarMultiplication(const ComplexData& a, double scalar, ComplexData& result) const {
        NOT_IMPLEMENTED(scalarMultiplication);
    }
    
    virtual void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
        NOT_IMPLEMENTED(complexMultiplicationWithConjugate);
    }
    
    virtual void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, double epsilon) const {
        NOT_IMPLEMENTED(complexDivisionStabilized);
    }

    // Advanced operations
    virtual void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
        NOT_IMPLEMENTED(calculateLaplacianOfPSF);
    }

    virtual void normalizeImage(ComplexData& resultImage, double epsilon) const {
        NOT_IMPLEMENTED(normalizeImage);
    }
    
    virtual void rescaledInverse(ComplexData& data, double cubeVolume) const {
        NOT_IMPLEMENTED(rescaledInverse);
    }

    // Gradient operations
    virtual void gradientX(const ComplexData& image, ComplexData& gradX) const {
        NOT_IMPLEMENTED(gradientX);
    }
    
    virtual void gradientY(const ComplexData& image, ComplexData& gradY) const {
        NOT_IMPLEMENTED(gradientY);
    }
    
    virtual void gradientZ(const ComplexData& image, ComplexData& gradZ) const {
        NOT_IMPLEMENTED(gradientZ);
    }
    
    virtual void computeTV(double lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
        NOT_IMPLEMENTED(computeTV);
    }
    
    virtual void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, double epsilon) const {
        NOT_IMPLEMENTED(normalizeTV);
    }


    virtual bool plansInitialized() const { 
        std::unique_lock lock(backendMutex);
        return plansInitialized_; }



protected:
    bool plansInitialized_ = false;
    mutable std::mutex backendMutex;
};

#undef NOT_IMPLEMENTED