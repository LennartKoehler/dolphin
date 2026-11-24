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
 *   tilt       = max(tan(asin(clamp(NA/ni))), tan(asin(clamp(NA/ns))))  (clamp 0.99)
 *   zMax_nm    = max(focusIdx, sizeZ-1-focusIdx) * pixelSizeAxial_nm
 *                (focus assumed mid-stack; worst-case distance to the rim)
 *   halfLat_nm = RING_FACTOR * resolution.lateral_nm
 *                + CONE_MARGIN * tilt * zMax_nm
 *   lateral    = min(max(ceil(halfLat_nm / pixelLat), ceil(RING_FACTOR * resLat / pixelLat)), MAX_LAT_HALF_WIDTH)
 *   axial      = ceil(max(zMax_nm, AXIAL_FLOOR_FACTOR * resolution.axial_nm) / pixelAx)
 *
 * Lateral: RING_FACTOR Airy/resolution rings plus the geometric defocus cone
 * (zMax * tan(marginal-ray tilt)) over the defocus range, capped at
 * MAX_LAT_HALF_WIDTH so very deep stacks do not explode the box. Axial: the
 * imaged defocus range (zMax) with a floor of AXIAL_FLOOR_FACTOR axial
 * resolutions for shallow stacks. The box is not clamped to the image size in
 * the vertical sense: the PSF support may exceed the imaged volume.
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
// Lateral cone margin on the geometric defocus term.
constexpr double CONE_MARGIN          = 1.1;
// Axial half-extent floor, in axial-resolution units, for shallow stacks where
// zMax alone would be too small to hold the defocused blur.
constexpr double AXIAL_FLOOR_FACTOR   = 2.0;
// Soft cap for the lateral half-extent: computeBoxHalfExtent does NOT
// truncate, but warns when the envelope exceeds this many voxels (box side
// 2*threshold + 1). The defocus cone term grows without bound with
// NA * stack depth (tilt ~7 at NA 1.4), so deep high-NA stacks can produce
// very large PSF boxes; the warning flags the cost, the size stays exact.
// Axial extent is never warned about or capped: it must span the imaged
// defocus range.
constexpr size_t SOFT_MAX_LATERAL_HALF_VOXELS = 150;
/**
 * Approximate optical resolution (nm) from NA, wavelength and sample RI for the
 * given criterion (see PSFConfig::ResolutionCriterion). Widefield formulas:
 *   RAYLEIGH: lateral 0.61*lambda/NA,  axial 2*lambda*n/NA^2
 *   FWHM:     lateral 0.51*lambda/NA,  axial 0.93*lambda/(n - sqrt(n^2 - NA^2))
 *   SIGMA:    lateral 0.408*lambda/NA, axial Rayleigh axial converted to FWHM
 *             (standard rayleigh -> fwhm axial factor)
 */
/** Approximate optical resolution (nm) from NA, wavelength and sample RI for the
 *  given criterion. lambda_nm is supplied by the caller (the modality's
 *  detection wavelength), not read from the config: in confocal mode the
 *  emission wavelength lives on the modality config.
 *
 *  Modality-aware: for a confocal config the resolution of the excitation x
 *  detection product is returned instead - the caller's lambda_nm is combined
 *  with the modality's excitation wavelength into
 *  lambda_eff = exc*det/sqrt(exc^2+det^2) (point-pinhole contraction
 *  lambda/sqrt(2) for equal wavelengths; a finite pinhole widens the true
 *  resolution toward the widefield value, which this estimate ignores). */
ApproxResolution approximateResolution(const GibsonLanniPSFConfig& config, double lambda_nm, int criterion);

/**
 * Half-extents (voxels) of the symmetric PSF box from the optics and the
 * imaged axial extent. `imageDepthPixels` is the depth of the volume the PSF
 * will deconvolve (cube or image depth in voxels), NOT a PSF size cap.
 * Nothing is truncated; a warning is emitted when the lateral half-extent
 * exceeds SOFT_MAX_LATERAL_HALF_VOXELS.
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
