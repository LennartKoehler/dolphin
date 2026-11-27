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
 *   lateral    = max(ceil(halfLat_nm / pixelLat), ceil(RING_FACTOR * resLat / pixelLat))
 *   axial      = ceil(max(zMax_nm, AXIAL_FLOOR_FACTOR * resolution.axial_nm) / pixelAx)
 *   side       = 2 * lateral + 1   (no cap: a warning is emitted above the soft cap)
 *
 * Lateral: RING_FACTOR Airy/resolution rings plus the geometric defocus cone
 * (zMax * tan(marginal-ray tilt)) over the defocus range. Axial: the imaged
 * defocus range (zMax) with a floor of AXIAL_FLOOR_FACTOR axial resolutions
 * for shallow stacks. The box is not clamped to the image size in the
 * vertical sense: the PSF support may exceed the imaged volume.
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
    size_t depth = 0;    // half-depth in voxels (box depth = 2*depth + 1)
};

// Covers 2 full rings; PSF ~1e-4 at the edge. Raise to 4.5 for hard deconv.
// Doubled in quadrature along the diagonal: a square box of side 2*L+1 is
// bounded by the Airy envelope at its corner, which is L*sqrt2 from the
// centre, so the ring count that has to fit is sqrt2 rings of half-width
// 2*RING_FACTOR. Without this the Airy tails are clipped diagonally.
constexpr double RING_FACTOR          = 3.5 * 1.4142135623730951;
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
 * Widefield (single-integral) axial extent, the RAYLEIGH criterion at the
 * sample index, in the 1-cos(alpha) structure:
 *
 *   resAx = lambda / (ns*(1 - cos(alpha)))    with n*sin(alpha) = NA
 *         = ns*lambda / (ns - sqrt(ns^2 - NA^2))
 *         = ns*lambda / NA^2     for NA << ns
 *
 * alpha is the marginal ray in the sample, sin(alpha) = NA/ns, clamped at
 * 0.99 so the width stays finite and positive when NA >= ns (oil objective
 * into a lower-index sample).
 *
 * `sampleIndex` is the sample-side index ns. The return is the full axial
 * extent at the effective wavelength supplied.
 */
double axialNWidth_nm(double lambda_nm, double NA, double sampleIndex);

/**
 * Widefield axial FWHM at the sample index, same 1-cos structure:
 *
 *   resAx = sqrt2 * lambda / (ns*(1 - cos(alpha)))
 *
 * equal to the classical 0.93*lambda/(ns - sqrt(ns^2 - NA^2)) form for
 * NA < ns (sqrt2*0.6577 = 0.9301).
 */
double axialFwhm_nm(double lambda_nm, double NA, double sampleIndex);

/**
 * Point-pinhole contraction factor of a confocal config, as a residual square
 * root of the excitation factor relative to a single widefield blur at
 * `detectionLambda_nm`:
 *
 *   k = (exc/em) / sqrt(1 + (exc/em)^2)
 *
 * A point-pinhole confocal PSF is the product of the excitation and detection
 * PSFs; combining Gaussian-equivalent widths gives
 *   res_confocal = res_widefield * k'
 * with k' = (exc/em)/sqrt(1+(exc/em)^2), i.e. 1/sqrt(2) for equal
 * wavelengths (the classical lambda/sqrt(2) contraction).
 *
 * Returns 1.0 for non-confocal configs (no contraction).
 */
double excitationResidual(const GibsonLanniPSFConfig& config, double detectionLambda_nm);

/**
 * Contract a widefield ApproxResolution by the point-pinhole confocal factor:
 *
 *   res_confocal = res_widefield * k
 *
 * Returns the widefield width for k = 1 (non-confocal), and the sharper
 * in-focus point-pinhole width for k = 1/sqrt(2). See approximateResolution()
 * for which of the two the BOX is sized from.
 */
ApproxResolution confocalResolution(const ApproxResolution& widefield, double k);

/**
 * Approximate optical resolution (nm) for the given criterion (see
 * PSFConfig::ResolutionCriterion):
 *   RAYLEIGH: lateral 0.61*lambda/NA,  axial axialNWidth_nm
 *   FWHM:     lateral 0.51*lambda/NA,  axial axialFwhm_nm
 *   SIGMA:    lateral 0.408*lambda/NA, axial Rayleigh axial / sqrt2
 *
 * Widefield formulas, evaluated at the EFFECTIVE wavelength of the modality:
 * for a confocal config the excitation and detection widths are combined
 * (Gaussian product) into
 *   lambda_eff = exc*em / sqrt(exc^2 + em^2)  = em * k
 * with k = excitationResidual(). So a closed pinhole (radius 0) yields the
 * CONFOCAL width of the PSF actually generated - the quantity to report and
 * to compare against the voxel pitch (Nyquist). Equal wavelengths give the
 * lambda/sqrt(2) contraction.
 *
 * A FINITE pinhole re-admits the detection blur, widening the true width
 * toward the widefield envelope (the exact value lies between and depends on
 * the pinhole size), so the contraction is skipped and the envelope is
 * returned. That keeps computeBoxHalfExtent() a support bound for every
 * pinhole size.
 *
 * lambda_nm is supplied by the caller (the modality's detection wavelength),
 * not read from the config: in confocal mode the emission wavelength lives on
 * the modality config.
 */
ApproxResolution approximateResolution(const GibsonLanniPSFConfig& config, double lambda_nm, int criterion);

/**
 * Sampling limits (nm) from the PSF's OTF bandwidth, per modality.
 *
 * A point-pinhole confocal PSF is the spatial product of the excitation and
 * detection integrals, so its OTF is the convolution of their incoherent OTFs
 * and the bandwidths ADD. The finest period (Abbe limit) and the Nyquist
 * voxel pitch (its half) are:
 *
 *   lateral:  Abbe  = 1 / (2 NA * (1/lamExc + 1/lamDet))
 *             Nyquist = 1 / (4 NA * (1/lamExc + 1/lamDet))
 *   axial:    Abbe  = 1 / (n (1-cos(alpha)) * (1/lamExc + 1/lamDet))
 *             Nyquist = 1 / (2n (1-cos(alpha)) * (1/lamExc + 1/lamDet))
 *
 * with the same alpha as axialNWidth_nm/axialFwhm_nm (sin(alpha) = NA/n,
 * clamped). Adding the bandwidths halves the confocal Abbe limit for equal
 * wavelengths (lateral lam/4NA instead of the widefield lam/2NA - the
 * "confocal doubles the resolution" fact in OTF terms). A widefield config
 * contributes a single factor (excitation = detection = lambda_nm).
 *
 * The Nyquist values are the sampling limits: a voxel pitch below them
 * undersamples the PSF's finest features (aliasing); above them oversamples.
 */
struct SamplingLimits {
    double abbeLateral_nm = 0.0;
    double nyquistLateral_nm = 0.0;
    double abbeAxial_nm = 0.0;
    double nyquistAxial_nm = 0.0;
};

SamplingLimits samplingLimits(const GibsonLanniPSFConfig& config, double lambda_nm);

/**
 * Half-extents (voxels) of the symmetric PSF box from the optics and the
 * imaged axial extent. `imageDepthPixels` is the depth of the volume the PSF
 * will deconvolve (cube or image depth in voxels), NOT a PSF size cap.
 *
 * `resolution` is the effective (modality- and pinhole-aware) width from
 * approximateResolution(): a confocal point pinhole contracts the ring term
 * and the axial floor, while a finite pinhole keeps the widefield envelope so
 * the box still bounds the re-admitted detection blur.
 *
 * Nothing is truncated; a warning is emitted when the lateral half-extent
 * exceeds SOFT_MAX_LATERAL_HALF_VOXELS.
 */
BoxHalfExtent computeBoxHalfExtent(const ApproxResolution& resolution,
                                   const GibsonLanniPSFConfig& config);

/**
 * Log the optical resolution (criterion from the config), the OTF-derived
 * sampling limits (Abbe finest period and Nyquist pitch) and the actual voxel
 * pitch, plus per-axis warnings when the pitch exceeds Nyquist (undersampled)
 * and per-axis info when it is well above 2x Nyquist (oversampled).
 * `resolution` is the value from approximateResolution(); the sampling limits
 * are computed from the OTF bandwidths, not from it.
 */
void logSampling(const GibsonLanniPSFConfig& config,
                 double lambda_nm,
                 int criterion,
                 const ApproxResolution& resolution,
                 double pixelLateral_nm,
                 double pixelAxial_nm);

} // namespace PSFBoxSizing
