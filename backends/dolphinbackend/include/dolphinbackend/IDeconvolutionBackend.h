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
#include <memory>
#include "ComplexData.h"


class BackendConfig;

// currently the backends implement lazy initialization of the fftw plans, the user should be careful of the input shapes
// as initialization of plans takes very long, so usually its best to stick to 1 shape or as little as possible, to most profit from reusing plans
// the lazy initialization should however enable one thread to init a new plan for the shape it needs and all other threads to keep using initialized threads
// so that not all threads have to wait for the initialization of all plans. The init of plans is singlethreaded as i understand it


// Helper macro for cleaner not-implemented exceptions
#define NOT_IMPLEMENTED(func_name) \
    throw std::runtime_error(std::string(#func_name) + " not implemented in " + typeid(*this).name())

// be sure that implementations of this are threadsafe
class IDeconvolutionBackend{
public:
    IDeconvolutionBackend() = default;
    virtual ~IDeconvolutionBackend(){};
    
    /**
     * Get the device type of this backend
     * @return Device type string
     */
    virtual std::string getDeviceString() const noexcept {
        return "unknown";
    }

    // Core functions - still pure virtual (must implement)
    virtual void init(const BackendConfig& config) = 0;
    virtual void cleanup() = 0;

    // Synchronization - default implementation for non-async backends
    virtual void sync() {
        // Default no-op implementation for backends that don't need synchronization
    }

    
    // FFT plan management
    virtual void initializePlan(const CuboidShape& cube){
        NOT_IMPLEMENTED(initializePlan);
    }
    

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
    
    virtual void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
        NOT_IMPLEMENTED(complexDivision);
    }
    
    virtual void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
        NOT_IMPLEMENTED(complexAddition);
    }
    
    virtual void scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const {
        NOT_IMPLEMENTED(scalarMultiplication);
    }
    
    virtual void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const {
        NOT_IMPLEMENTED(complexMultiplicationWithConjugate);
    }
    
    virtual void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const {
        NOT_IMPLEMENTED(complexDivisionStabilized);
    }

    // Advanced operations
    virtual void calculateLaplacianOfPSF(const ComplexData& psf, ComplexData& laplacian) const {
        NOT_IMPLEMENTED(calculateLaplacianOfPSF);
    }

    virtual void normalizeImage(ComplexData& resultImage, real_t epsilon) const {
        NOT_IMPLEMENTED(normalizeImage);
    }
    
    virtual void rescaledInverse(ComplexData& data, real_t cubeVolume) const {
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
    
    virtual void computeTV(real_t lambda, const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& tv) const {
        NOT_IMPLEMENTED(computeTV);
    }
    
    virtual void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t epsilon) const {
        NOT_IMPLEMENTED(normalizeTV);
    }


};

#undef NOT_IMPLEMENTED