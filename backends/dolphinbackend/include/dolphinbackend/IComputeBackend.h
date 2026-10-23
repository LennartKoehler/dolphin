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
#include <stdexcept>
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
    virtual ~IComputeBackend() = default;

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

    // ========================================================================
    // Real-valued elementwise / reduction primitives.
    //
    // Default implementations throw so backends that do not (yet) support
    // them (e.g. CUDA) compile unchanged and fail fast at runtime. The CPU
    // backend overrides all of them. These power the Poisson-likelihood
    // (PoissonMLE) deconvolution algorithm.
    // ========================================================================

#define COMPUTE_NOT_IMPLEMENTED(fn) \
    throw std::runtime_error(std::string(#fn) + " not implemented in " + typeid(*this).name())

    // result = sum of all elements of a
    virtual void realSum(const RealData& a, real_t* result) const {
        COMPUTE_NOT_IMPLEMENTED(realSum);
    }

    // result[i] = sa*a[i] + sb*b[i]   (in-place on a is allowed)
    virtual void realLinearCombination(const RealData& a, real_t sa, const RealData& b, real_t sb, RealData& result) const {
        COMPUTE_NOT_IMPLEMENTED(realLinearCombination);
    }

    // result[i] = a[i] * scalar + add
    virtual void realAffine(const RealData& a, real_t scalar, real_t add, RealData& result) const {
        COMPUTE_NOT_IMPLEMENTED(realAffine);
    }

    // Fill with a constant value
    virtual void realFillConstant(RealData& a, real_t value) const {
        COMPUTE_NOT_IMPLEMENTED(realFillConstant);
    }

    // result[i] = ( sqrt(a[i]*scale + add) - 1 ) / lambda   (the PoissonMLE shrinkage map)
    // Guarded so the argument of sqrt is non-negative and lambda != 0.
    virtual void realSqrtAffine(const RealData& a, real_t scale, real_t add, real_t lambda, RealData& result) const {
        COMPUTE_NOT_IMPLEMENTED(realSqrtAffine);
    }

    // result[i] = max(a[i], lo)
    virtual void realClampMin(const RealData& a, real_t lo, RealData& result) const {
        COMPUTE_NOT_IMPLEMENTED(realClampMin);
    }

    // Statistics over all valid elements: min/max/mean and number of non-finite values
    // (NaN + Inf). min/max/mean are computed over finite elements only.
    virtual void realStats(const RealData& a, real_t* minOut, real_t* maxOut, real_t* meanOut, size_t* nonFiniteOut) const {
        COMPUTE_NOT_IMPLEMENTED(realStats);
    }

    // One-pass evaluation, at a given step length alpha, of the I-divergence (Poisson NLL
    // up to a constant) and its first two alpha-derivatives along the line m(alpha) = m + alpha*hd:
    //   J(alpha)    = sum_i [ g_i*ln(g_i/m_i) - g_i + m_i ]
    //   dJ/dalpha   = sum_i hd_i * (1 - g_i/m_i)
    //   d2J/dalpha2 = sum_i g_i * hd_i^2 / m_i^2
    // Safe division (denominators floored); g_i <= 0 contributes m_i to J.
    virtual void poissonLineSearchEval(const RealData& g, const RealData& m, const RealData& hd,
                                       real_t alpha, real_t* J, real_t* dJ, real_t* d2J) const {
        COMPUTE_NOT_IMPLEMENTED(poissonLineSearchEval);
    }

    // Fill a half-complex (r2c, W/2+1) buffer with the frequency-domain OTF of a
    // separable anisotropic 3D Gaussian whose spatial std-devs are sigmaX/Y/Z (in
    // voxels):  OTF(k) = exp(-2*pi^2 * (sx^2 fx^2 + sy^2 fy^2 + sz^2 fz^2)), real-valued,
    // with fx/fy/fz in cycles per sample (bin index / N, wrapped). DC at origin.
    virtual void fillGaussianOTF(ComplexData& out, real_t sigmaX, real_t sigmaY, real_t sigmaZ) const {
        COMPUTE_NOT_IMPLEMENTED(fillGaussianOTF);
    }

    // Scale a half-complex OTF buffer in place so its DC bin (element [0]'s real part,
    // = the sum of the spatial PSF) equals 1. The I-divergence (PoissonMLE) forward model
    // m = Hf + bg is only physically correct for a unit-sum PSF (RichardsonLucy is
    // scale-invariant and does not need this). Backends read the DC bin per their memory
    // model (direct host read on CPU, a single device readback on CUDA). No-op if
    // |DC| < epsilon.
    virtual void normalizePsfUnitSum(ComplexData& psf, real_t epsilon) const {
        COMPUTE_NOT_IMPLEMENTED(normalizePsfUnitSum);
    }

};

