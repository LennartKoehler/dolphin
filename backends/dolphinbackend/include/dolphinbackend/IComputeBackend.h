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
#include <vector>
#include "ComplexData.h"


struct BackendConfig;


enum PlanDirection{
    FORWARD,
    BACKWARD,
};
enum PlanType{
    REAL,
    COMPLEX
};

struct FFTPlanDescription{
    PlanDirection direction;
    CuboidShape shape;
    PlanType type;
    bool inPlace;

    FFTPlanDescription(
        PlanDirection direction,
        PlanType type,
        CuboidShape shape,
        bool inPlace
    ):
        direction(direction),
        type(type),
        shape(shape),
        inPlace(inPlace){}

    virtual bool operator==(const FFTPlanDescription& other) const {
        return (shape == other.shape && direction == other.direction && type == other.type && inPlace == other.inPlace);
    }
};


// currently the backends implement lazy initialization of the fftw plans, the user should be careful of the input shapes
// as initialization of plans takes very long, so usually its best to stick to 1 shape or as little as possible, to most profit from reusing plans
// the lazy initialization should however enable one thread to init a new plan for the shape it needs and all other threads to keep using initialized threads
// so that not all threads have to wait for the initialization of all plans. The init of plans is singlethreaded as i understand it


// be sure that implementations of this are threadsafe
class IComputeBackend{
public:
    IComputeBackend() = default;
    virtual ~IComputeBackend(){};

    /**
     * Get the device type of this backend
     * @return Device type string
     */
    virtual std::string getDeviceString() const noexcept {
        return "unknown";
    }


    // Synchronization - default implementation for non-async backends
    virtual void sync() {
        // Default no-op implementation for backends that don't need synchronization
    }


    // FFT plan management
    virtual void initializePlan(const FFTPlanDescription& description) = 0;


    // Debug functions
    virtual void hasNAN(const ComplexData& data) const = 0;


    // FFT functions
    virtual void forwardFFT(const ComplexData& in, ComplexData& out) const = 0;

    virtual void forwardFFT(const RealData& in, ComplexData& out) const = 0;


    virtual void backwardFFT(const ComplexData& in, ComplexData& out) const = 0;

    virtual void backwardFFT(const ComplexData& in, RealData& out) const = 0;

    // Shift operations
    virtual void octantFourierShift(ComplexData& data) const = 0;

    virtual void octantFourierShift(RealData& data) const = 0;

    virtual void inverseQuadrantShift(ComplexData& data) const = 0;

    // Complex arithmetic operations
    virtual void complexMultiplication(const ComplexData& a, const ComplexData& b, ComplexData& result) const = 0;

    virtual void multiplication(const RealData& a, const RealData& b, RealData& result) const = 0;

    virtual void sum(const ComplexData& data, complex_t* result) const = 0;

    virtual void meanSquareError(const ComplexData& a, const ComplexData& b, real_t* result) const = 0;

    virtual void complexDivision(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const = 0;

    virtual void division(const RealData& a, const RealData& b, RealData& result, real_t epsilon) const = 0;

    virtual void complexAddition(complex_t** data, ComplexData& sum, size_t nImages, size_t imageVolume) const = 0;
    virtual void complexAddition(const ComplexData& a, const ComplexData& b, ComplexData& result) const = 0;

    virtual void sumToOne(real_t** data, size_t nImages, size_t imageVolume) const = 0;
    virtual void scalarMultiplication(const ComplexData& a, complex_t scalar, ComplexData& result) const = 0;

    virtual void scalarMultiplication(const RealData& a, real_t scalar, RealData& result) const = 0;

    virtual void complexMultiplicationWithConjugate(const ComplexData& a, const ComplexData& b, ComplexData& result) const = 0;

    virtual void complexDivisionStabilized(const ComplexData& a, const ComplexData& b, ComplexData& result, real_t epsilon) const = 0;


    // Gradient operations
    virtual void gradientX(const ComplexData& image, ComplexData& gradX) const = 0;

    virtual void gradientY(const ComplexData& image, ComplexData& gradY) const = 0;

    virtual void gradientZ(const ComplexData& image, ComplexData& gradZ) const = 0;


    // Gradient operations for real-valued data
    virtual void gradientX(const RealData& image, RealData& gradX) const = 0;

    virtual void gradientY(const RealData& image, RealData& gradY) const = 0;

    virtual void gradientZ(const RealData& image, RealData& gradZ) const = 0;
    virtual void gradient(const RealData& image, RealData& gradX, RealData& gradY, RealData& gradZ) const = 0;

    // Divergence operations (backward differences — adjoint of forward gradient)
    // Computes: div[i] = (gx[i] - gx[i-1]) + (gy[i] - gy[i-stride_y]) + (gz[i] - gz[i-stride_z])
    // with zero boundary conditions (values at i=0 along each axis use 0 for the i-1 term)
    virtual void divergence(const RealData& gx, const RealData& gy, const RealData& gz, RealData& result) const = 0;

    // Divergence for complex-valued vector fields (operates on real parts)
    virtual void divergence(const ComplexData& gx, const ComplexData& gy, const ComplexData& gz, ComplexData& result) const = 0;

    virtual void computeTV(real_t lambda, const ComplexData& div, ComplexData& tv) const = 0;

    // computeTV for real-valued divergence
    virtual void computeTV(real_t lambda, const RealData& div, RealData& tv) const = 0;

    // Smoothed TV subgradient: gx / sqrt(|∇f|² + β²)
    // β controls the transition between TV (edge-preserving) and Tikhonov (smooth) behavior.
    // - At edges (|∇f| >> β): behaves like standard TV (gx/|∇f|)
    // - In flat regions (|∇f| << β): behaves like Tikhonov (gx/β) — prevents noise amplification
    virtual void normalizeTV(ComplexData& gradX, ComplexData& gradY, ComplexData& gradZ, real_t beta) const = 0;

    // Smoothed TV subgradient for real-valued gradients
    virtual void normalizeTV(RealData& gradX, RealData& gradY, RealData& gradZ, real_t beta) const = 0;

};

