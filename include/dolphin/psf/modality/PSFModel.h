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

#include <cstddef>
#include <optional>
#include <memory>
#include <variant>
#include <vector>

class PSFConfig;

/** Radially symmetric slice: |integral|^2 profile with h[0] at r = 0,
 *  tabulated at samplesPerPixel radial samples per lateral pixel, whose
 *  physical pitch is lateralPitchNm. Models use this representation as an
 *  internal optimization; the modality layer blits it onto the slice grid. */
struct RadialSlice {
    std::vector<double> h;
    size_t samplesPerPixel = 1;
    double lateralPitchNm = 0.0;   // physical pitch of one lateral pixel (nm)
};

/** Explicit 2D intensity slab (width * height, row-major). For models whose
 *  slices are not radially symmetric about the optical axis (astigmatic,
 *  phase-masked, ...). */
struct FullSlice {
    std::vector<float> data;
    size_t width = 0;
    size_t height = 0;
};

using ModelSlice = std::variant<RadialSlice, FullSlice>;

/** Slice geometry of a PSF box, in generic orchestrator terms. Slice k sees
 *  the model's defocus offset (firstOffset + k). */
struct ModelBox {
    size_t width = 0;
    size_t height = 0;
    size_t depth = 0;
    size_t lateralHalf = 0;   // composed slices span (2*lateralHalf+1)^2
    long firstOffset = 0;     // defocus offset of slice 0
};

/**
 * PSF model facade. Everything the modality layer needs from a model is
 * expressed against the abstract PSFConfig, so modalities stay model-free.
 * A model owns its numerics AND its orchestration data:
 *
 *   - how its auto-size box is derived (autoBox) - e.g. from optical
 *     resolution criteria, or from a cutoff threshold;
 *   - how a slice's defocus is parameterized (sliceConfig) - e.g. a working
 *     distance detour per slice, or no defocus at all;
 *   - at which wavelength it detects (detectionLambda) - std::nullopt for
 *     wavelength-less models.
 *
 * Purely numerical otherwise: no threads, no PSF assembly, no modality
 * knowledge beyond what the config carries.
 */
class PSFModel {
public:
    virtual ~PSFModel() = default;

    /** One-time tabulation of shared state covering slices of the given
     *  lateral extent. shortestLambda is the shortest wavelength the
     *  modality will request (std::nullopt for wavelength-less models).
     *  Single-threaded; must run before parallel intensitySlice calls. */
    virtual void init(const PSFConfig& config,
                      size_t sliceWidth,
                      size_t sliceHeight,
                      std::optional<double> shortestLambda) const = 0;

    /** Intensity of ONE wavelength at ONE defocus (the slice config carries
     *  the model's defocus parameter). lambda is the wavelength chosen by
     *  the modality - std::nullopt means "the model's own detection
     *  wavelength"; wavelength-less models ignore it. samplesPerPixel is the
     *  radial sampling density per lateral pixel - compositions may request
     *  more than the pixel grid needs. forcedCutoff > 0 pins the lateral
     *  half-width and disables the value-based early stop; 0 lets the model
     *  stop on config.cutoffThreshold. */
    virtual ModelSlice intensitySlice(const PSFConfig& sliceCfg,
                                      std::optional<double> lambda,
                                      size_t samplesPerPixel,
                                      size_t maxLateralPixels,
                                      size_t forcedCutoff) const = 0;

    /** Auto-size box geometry for the imaged volume described by the config.
     *  detectionLambda is the wavelength the modality will detect at (for
     *  resolution-based sizing); std::nullopt for wavelength-less models. */
    virtual ModelBox autoBox(const PSFConfig& config,
                             std::optional<double> detectionLambda) const = 0;

    /** The config for slice k of the box: applies the model's defocus
     *  parameterization to the offset. Returns a new config object; the
     *  input is not modified. */
    virtual std::shared_ptr<const PSFConfig> sliceConfig(const PSFConfig& config,
                                                         long defocusOffset) const = 0;

    /** Wavelength this model detects at by default. std::nullopt for
     *  wavelength-less models. */
    virtual std::optional<double> detectionLambda(const PSFConfig& config) const = 0;

    /** Validates the config for this model's numerics and throws on values
     *  that make computation impossible (non-positive wavelength or pixel
     *  pitch, negative geometry). Merely suspicious values are logged as
     *  warnings. Called once per generation, before autoBox/init. */
    virtual void validateConfig(const PSFConfig& config) const = 0;
};
