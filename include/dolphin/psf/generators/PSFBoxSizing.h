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

class GibsonLanniPSFConfig;

/**
 * Analytical PSF box sizing for the integral-based Gibson-Lanni generator.
 *
 * The box is derived from the optics AND the imaged axial extent; the config
 * sizes are the size of the IMAGE the PSF is produced for, not the PSF size.
 * Model (all physical terms in nm, converted to voxels by the pitch):
 *
 *   dTan       = |tan(asin(clamp(NA/ni))) - tan(asin(clamp(NA/ns)))|  (clamp 0.99)
 *   zMax_nm    = max(focusIdx, sizeZ-1-focusIdx) * pixelSizeAxial_nm
 *                (focus assumed mid-stack; worst-case distance to the rim)
 *   halfLat_nm = RING_FACTOR * resolution.lateral_nm
 *                + CONE_MARGIN * dTan * zMax_nm
 *   halfAx_nm  = zMax_nm + AXIAL_TAIL_FACTOR * resolution.axial_nm
 *   lateral    = max(ceil(halfLat_nm / pixelLat), ceil(RING_FACTOR * resLat / pixelLat))
 *   axial      = max(ceil(halfAx_nm  / pixelAx),  ceil(AXIAL_TAIL_FACTOR * resAx / pixelAx))
 *
 * Lateral: RING_FACTOR Airy/resolution rings plus the RI-mismatch cone spread
 * over the defocus range. Axial: the full imaged defocus range plus a tail of
 * AXIAL_TAIL_FACTOR axial resolutions. The box is never clamped by the image
 * size: a PSF support may legitimately exceed the imaged volume.
 *
 * The returned values are half-widths (in voxels) of a symmetric box.
 */
namespace PSFBoxSizing {

struct ApproxResolution {
    double lateral_nm = 0.0;
    double axial_nm = 0.0;
};

struct BoxHalfExtent {
    size_t lateral = 0;  // half-width in voxels (box side = 2*lateral + 1)
    size_t axial = 0;    // half-depth in voxels (box depth = 2*axial + 1)
};

// Covers 2 full rings; PSF ~1e-4 at the edge. Raise to 4.5 for hard deconv.
constexpr double RING_FACTOR          = 3.5;
// Lateral cone margin on the RI-mismatch term.
constexpr double CONE_MARGIN          = 1.1;
// Axial tail = this many axial FWHM beyond the stack. 8 widefield, ~12 confocal.
constexpr double AXIAL_TAIL_FACTOR    = 8.0;

/**
 * Approximate optical resolution (nm) from NA, wavelength and sample RI for the
 * given criterion (see PSFConfig::ResolutionCriterion).
 *   RAYLEIGH: lateral 0.61*lambda/NA,  axial 2*lambda*n/NA^2
 *   FWHM:     lateral 0.51*lambda/NA,  axial 0.93*lambda/(n - sqrt(n^2 - NA^2))
 *   SIGMA:    lateral 0.408*lambda/NA, axial Rayleigh axial converted to FWHM
 *             (standard rayleigh -> fwhm axial factor)
 */
ApproxResolution approximateResolution(const GibsonLanniPSFConfig& config, int criterion);

/**
 * Half-extents (voxels) of the symmetric PSF box from the optics and the
 * imaged axial extent. `imageDepthPixels` is the depth of the volume the PSF
 * will deconvolve (cube or image depth in voxels), NOT a PSF size cap.
 */
BoxHalfExtent computeBoxHalfExtent(const ApproxResolution& resolution,
                                   const GibsonLanniPSFConfig& config);

/**
 * Warn if the voxel pitch undersamples the resolution laterally (Nyquist).
 */
void warnIfUndersampled(const ApproxResolution& resolution,
                        double pixelLateral_nm,
                        double pixelAxial_nm);

} // namespace PSFBoxSizing
