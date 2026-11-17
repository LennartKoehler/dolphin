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
#include <variant>
#include <vector>

class GibsonLanniPSFConfig;

/** Radially symmetric slice: |integral|^2 profile with h[0] at r = 0,
 *  tabulated at samplesPerPixel radial samples per lateral pixel. Models
 *  use this representation as an internal optimization; the modality layer
 *  blits it onto the slice grid. */
struct RadialSlice {
    std::vector<double> h;
    size_t samplesPerPixel = 1;
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

/**
 * PSF model: the blur physics of ONE wavelength at ONE defocus. Purely
 * numerical - no threads, no PSF assembly, no modality knowledge. Modality
 * strategies compose PSFs from these slices (see ModalityStrategy.h), so the
 * model/modality separation stays fixed as either side grows.
 *
 * Currently scoped to Gibson-Lanni-style optical configs
 * (GibsonLanniPSFConfig carries the optics and the per-slice defocus ti_nm).
 */
class PSFModel {
public:
    virtual ~PSFModel() = default;

    /** One-time tabulation of shared state (Bessel table) covering slices of
     *  the given lateral extent and the shortest wavelength that will be
     *  requested. Single-threaded; must run before parallel intensitySlice
     *  calls. */
    virtual void init(const GibsonLanniPSFConfig& config,
                         size_t sliceWidth,
                         size_t sliceHeight) const = 0;

    /** Intensity of ONE wavelength at ONE defocus (config.ti_nm carries the
     *  defocus). samplesPerPixel is the radial sampling density per lateral
     *  pixel - compositions may request more than the pixel grid needs.
     *  forcedCutoff > 0 pins the lateral half-width and disables the
     *  value-based early stop; 0 lets the model stop on config.cutoffThreshold. */
    virtual ModelSlice intensitySlice(const GibsonLanniPSFConfig& config,
                                      double lambda_nm,
                                      size_t samplesPerPixel,
                                      size_t maxLateralPixels,
                                      size_t forcedCutoff) const = 0;
};
