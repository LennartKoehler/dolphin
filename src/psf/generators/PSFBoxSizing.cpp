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

#include "dolphin/psf/generators/PSFBoxSizing.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace PSFBoxSizing {


// Rayleigh axial resolution: 2 * lambda * n / NA^2 (in-focus axial extent).
double rayleighAxial_nm(double lambda_nm, double NA, double n) {
    return 2.0 * lambda_nm * n / (NA * NA);
}

// FWHM axial resolution: 0.93 * lambda / (n - sqrt(n^2 - NA^2)).
double fwhmAxial_nm(double lambda_nm, double NA, double n) {
    const double denom = n - std::sqrt(std::max(0.0, n * n - NA * NA));
    if (denom <= 0.0) return rayleighAxial_nm(lambda_nm, NA, n);
    return 0.93 * lambda_nm / denom;
}

// Convert a Rayleigh axial extent to FWHM using the standard factor
// FWHM = Rayleigh / 1.4142 (sigma-to-FWHM relation
// rayleigh2fwhmAxial).
double rayleighToFwhmAxial(double rayleighAxial) {
    return rayleighAxial / std::sqrt(2.0);
}


ApproxResolution approximateResolution(const GibsonLanniPSFConfig& config, int criterion) {
    const double lambda = static_cast<double>(config.wavelength_nm);
    const double NA = static_cast<double>(config.NA);
    const double n = static_cast<double>(config.ns);

    if (NA <= 0.0) {
        return {0.0, 0.0};
    }

    ApproxResolution res;
    switch (static_cast<PSFConfig::ResolutionCriterion>(criterion)) {
        case PSFConfig::ResolutionCriterion::FWHM:
            res.lateral_nm = 0.51 * lambda / NA;
            res.axial_nm = fwhmAxial_nm(lambda, NA, n);
            break;
        case PSFConfig::ResolutionCriterion::SIGMA:
            res.lateral_nm = 0.408 * lambda / NA;
            res.axial_nm = rayleighToFwhmAxial(rayleighAxial_nm(lambda, NA, n));
            break;
        case PSFConfig::ResolutionCriterion::RAYLEIGH:
        default:
            res.lateral_nm = 0.61 * lambda / NA;
            res.axial_nm = rayleighAxial_nm(lambda, NA, n);
            break;
    }
    return res;
}
// asin argument must stay below 1 and away from tan(alpha) -> inf.
inline double asin_clamped(double x) {
    return std::asin(std::clamp(x, 0.0, 0.99));
}

BoxHalfExtent computeBoxHalfExtent(const ApproxResolution& resolution,
                                   const GibsonLanniPSFConfig& config)
{
    BoxHalfExtent box;

    // --- geometric defocus cone ---------------------------------------------
    // The defocused PSF spreads laterally by zMax * tan(ray tilt). The relevant
    // tilt is the absolute marginal-ray angle in the propagating media (large
    // sample-side tilt for immersion objectives), NOT just the RI-mismatch
    // differential: even a matched objective opens a cone of zMax*tan(alpha).
    // Taking the max of the sample/immersion tilts is a conservative bound on
    // the footprint regardless of which side is wider.
    const double alpha_i = asin_clamped(config.NA / config.ni);
    const double alpha_s = asin_clamped(config.NA / config.ns);
    const double tilt    = std::max(std::tan(alpha_i), std::tan(alpha_s));

    // --- farthest slice from the focal plane, in nm -------------------------
    // Focus plane unknown here -> assume mid-stack (worst case <= half depth).
    const size_t focusIndex = config.sizeZ / 2;
    const double focusIdx = static_cast<double>(focusIndex);
    const double zMax_nm = std::max(focusIdx,
                                    static_cast<double>(config.sizeZ) - 1.0 - focusIdx)
                           * config.pixelSizeAxial_nm;

    // --- half extents (physical, lateral) / (voxel, axial) -----------------
    const double halfLat_nm = RING_FACTOR * resolution.lateral_nm
                            + CONE_MARGIN * tilt * zMax_nm;

    // --- to voxels ----------------------------------------------------------
    if (config.pixelSizeLateral_nm > 0.0) {
        const size_t full = static_cast<size_t>(std::ceil(halfLat_nm / config.pixelSizeLateral_nm));
        const size_t ringFloor = static_cast<size_t>(
            std::ceil(RING_FACTOR * resolution.lateral_nm / config.pixelSizeLateral_nm));
        box.lateral = std::max(full, ringFloor);
    } else {
        box.lateral = 0;
    }

    // Axial: span the imaged defocus range, with a floor of a few axial
    // resolutions so shallow stacks still hold the defocused blur.
    if (config.pixelSizeAxial_nm > 0.0) {
        const double axialHalf_nm = std::max(zMax_nm, AXIAL_FLOOR_FACTOR * resolution.axial_nm);
        box.axial = static_cast<size_t>(std::ceil(axialHalf_nm / config.pixelSizeAxial_nm));
    } else {
        box.axial = 0;
    }

    return box;
}

void warnIfUndersampled(const ApproxResolution& resolution,
                        double pixelLateral_nm,
                        double pixelAxial_nm) {
    const double nyquistLateral_nm = 0.5 * resolution.lateral_nm;
    const double nyquistAxial_nm = 0.5 * resolution.axial_nm;

    if (pixelLateral_nm > nyquistLateral_nm) {
        spdlog::warn("GibsonLanni: lateral pitch {:.1f}nm undersamples resolution {:.1f}nm "
                     "(Nyquist {:.1f}nm); PSF box may be aliased",
                     pixelLateral_nm, resolution.lateral_nm, nyquistLateral_nm);
    }
    if (pixelAxial_nm > nyquistAxial_nm) {
        spdlog::warn("GibsonLanni: axial pitch {:.1f}nm undersamples resolution {:.1f}nm "
                     "(Nyquist {:.1f}nm); PSF box may be aliased",
                     pixelAxial_nm, resolution.axial_nm, nyquistAxial_nm);
    }
}

} // namespace PSFBoxSizing
