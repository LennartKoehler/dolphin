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
#include <string>
#include <spdlog/spdlog.h>

namespace {

// asin argument must stay below 1 and away from tan(alpha) -> inf.
inline double asin_clamped(double x) {
    return std::asin(std::clamp(x, 0.0, 0.99));
}

// Marginal ray in the sample: sin(alpha) = NA/ns.
double axialSinClamped(double NA, double sampleIndex) {
    if (sampleIndex <= 0.0 || NA <= 0.0) return 0.0;
    // NA < ns gives the real marginal ray. NA >= ns (oil objective into a
    // lower-index sample) has no propagating marginal ray; the clamp 0.99,
    // exactly like the lateral tilt, keeps lambda/(ns(1-cos)) finite.
    return std::clamp(NA / sampleIndex, 0.0, 0.99);
}

const char* criterionName(int criterion) {
    switch (static_cast<PSFConfig::ResolutionCriterion>(criterion)) {
        case PSFConfig::ResolutionCriterion::FWHM: return "FWHM";
        case PSFConfig::ResolutionCriterion::SIGMA: return "sigma";
        case PSFConfig::ResolutionCriterion::RAYLEIGH:
        default: return "Rayleigh";
    }
}

// Human-readable modality state for the sampling log lines.
std::string modalityLabel(const GibsonLanniPSFConfig& config) {
    if (config.getModalityType() != MicroscopeModality::CONFOCAL)
        return "widefield";
    if (const auto* modalityCfg = config.modality<ConfocalModalityConfig>()) {
        if (modalityCfg->pinholeRadius_um > 0.0f)
            return "confocal, finite pinhole (" + std::to_string(modalityCfg->pinholeRadius_um) + " um)";
        return "confocal, point pinhole";
    }
    return "confocal";
}

} // namespace

namespace PSFBoxSizing {

// Convert a Rayleigh axial extent to FWHM using the standard factor
// FWHM = Rayleigh / 1.4142 (the sigma-to-FWHM relation).
double rayleighToFwhmAxial(double rayleighAxial) {
    return rayleighAxial / std::sqrt(2.0);
}

// Widefield (single-integral) axial extent at the sample index, in the
// 1-cos(alpha) structure:
//   RAYLEIGH:  ns*lambda / (ns - sqrt(ns^2 - NA^2)) = lambda / (ns*(1-cos(alpha)))
// with n*sin(alpha) = NA for the marginal ray. Equal to ns*lambda/NA^2 for
// NA << ns. Written as lambda/(ns(1-cos)) and not as ns*lambda/(ns*sin^2),
// which cancels ns and degenerates (sin -> 1) exactly at NA = ns.
double axialNWidth_nm(double lambda_nm, double NA, double sampleIndex) {
    const double s = axialSinClamped(NA, sampleIndex);
    if (s <= 0.0) return 0.0;
    return lambda_nm / (sampleIndex * (1.0 - std::sqrt(1.0 - s * s)));
}

// Widefield axial FWHM at the sample index, same 1-cos structure:
//   FWHM = sqrt2 * lambda / (ns*(1-cos(alpha)))
// which is the classical 0.93*lambda/(ns - sqrt(ns^2-NA^2)) (0.658 =
// sqrt2*0.4655; the 0.93/(lambda) coefficient absorbs the difference).
double axialFwhm_nm(double lambda_nm, double NA, double sampleIndex) {
    const double s = axialSinClamped(NA, sampleIndex);
    if (s <= 0.0) return 0.0;
    return std::sqrt(2.0) * lambda_nm / (sampleIndex * (1.0 - std::sqrt(1.0 - s * s)));
}

double excitationResidual(const GibsonLanniPSFConfig& config, double detectionLambda_nm) {
    const auto* modalityCfg = config.modality<ConfocalModalityConfig>();
    if (!modalityCfg)
        return 1.0;
    // Degenerate fallback (excitation <= 0): the excitation integral uses the
    // detection wavelength (see ConfocalModality::composeSlice), so the
    // residual factors out to the equal-wavelength value 1/sqrt2.
    const double excitation = modalityCfg->excitationWavelength_nm > 0.0f
        ? static_cast<double>(modalityCfg->excitationWavelength_nm)
        : detectionLambda_nm;
    if (excitation <= 0.0 || detectionLambda_nm <= 0.0)
        return 1.0;
    // k = (exc/em) / sqrt(1 + (exc/em)^2); scale-free, so no overflow for
    // realistic wavelength ratios.
    const double ratio = excitation / detectionLambda_nm;
    const double denom = std::sqrt(1.0 + ratio * ratio);
    if (!(denom > 0.0))
        return 1.0;
    return ratio / denom;
}

ApproxResolution confocalResolution(const ApproxResolution& widefield, double k) {
    const double c = std::clamp(k, 0.0, 1.0);
    return {widefield.lateral_nm * c, widefield.axial_nm * c};
}

ApproxResolution approximateResolution(const GibsonLanniPSFConfig& config, double lambda_nm, int criterion) {
    // Widefield formulas at the effective wavelength.
    //
    // A closed (point) pinhole forms the voxel-wise product of the
    // excitation and detection integrals, so its width contracts by
    // excitationResidual() (lambda_eff = exc*em/sqrt(exc^2+em^2), i.e.
    // lambda/sqrt(2) for equal wavelengths). A finite pinhole re-admits the
    // detection blur, widening the true width toward the widefield envelope,
    // so no contraction is applied and the envelope is returned: it bounds
    // the defocused off-axis support and keeps computeBoxHalfExtent() a
    // support bound for every pinhole size.
    double lambda = lambda_nm;
    if (config.getModalityType() == MicroscopeModality::CONFOCAL) {
        if (const auto* modalityCfg = config.modality<ConfocalModalityConfig>()) {
            const double excitation = modalityCfg->excitationWavelength_nm > 0.0f
                ? static_cast<double>(modalityCfg->excitationWavelength_nm)
                : lambda_nm;
            const double em = modalityCfg->emissionWavelength_nm > 0.0f
                ? static_cast<double>(modalityCfg->emissionWavelength_nm)
                : lambda_nm;
            if (em > 0.0 && excitation > 0.0 && modalityCfg->pinholeRadius_um <= 0.0f) {
                // k = excitationResidual(): the product of two Gaussian
                // equivalent widths of em and excitation collapses to em*k.
                const double k = excitation / std::sqrt(excitation * excitation + em * em);
                lambda = em * k;
            }
        }
    }
    const double NA = static_cast<double>(config.NA);
    const double n = static_cast<double>(config.ns);

    if (NA <= 0.0) {
        return {0.0, 0.0};
    }

    ApproxResolution res;
    switch (static_cast<PSFConfig::ResolutionCriterion>(criterion)) {
        case PSFConfig::ResolutionCriterion::FWHM:
            res.lateral_nm = 0.51 * lambda / NA;
            res.axial_nm = axialFwhm_nm(lambda, NA, n);
            break;
        case PSFConfig::ResolutionCriterion::SIGMA:
            res.lateral_nm = 0.408 * lambda / NA;
            res.axial_nm = rayleighToFwhmAxial(axialNWidth_nm(lambda, NA, n));
            break;
        case PSFConfig::ResolutionCriterion::RAYLEIGH:
        default:
            res.lateral_nm = 0.61 * lambda / NA;
            res.axial_nm = axialNWidth_nm(lambda, NA, n);
            break;
    }
    return res;
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
        if (box.lateral > SOFT_MAX_LATERAL_HALF_VOXELS)
            spdlog::warn("GibsonLanni: autosized PSF lateral half-extent {} voxels ({} nm) is above the soft cap of {} (box side {}); the PSF is generated at full size, but generation and deconvolution cost scale with it",
                         box.lateral, static_cast<long>(halfLat_nm), SOFT_MAX_LATERAL_HALF_VOXELS,
                         2 * box.lateral + 1);
    } else {
        box.lateral = 0;
    }

    // Axial: span the imaged defocus range, with a floor of a few axial
    // resolutions so shallow stacks still hold the defocused blur.
    if (config.pixelSizeAxial_nm > 0.0) {
        const double axialHalf_nm = std::max(zMax_nm, AXIAL_FLOOR_FACTOR * resolution.axial_nm);
        box.depth = static_cast<size_t>(std::ceil(axialHalf_nm / config.pixelSizeAxial_nm));
    } else {
        box.depth = 0;
    }

    return box;
}

SamplingLimits samplingLimits(const GibsonLanniPSFConfig& config, double lambda_nm) {
    // OTF bandwidths of the incoherent integrals: 2*NA/lambda laterally,
    // n*(1-cos(alpha))/lambda axially (with n*sin(alpha) = NA, clamped as in
    // axialNWidth_nm). The point-pinhole confocal PSF is the spatial PRODUCT
    // of the excitation and detection integrals, so its OTF is the
    // CONVOLUTION of theirs and the bandwidths ADD. Widefield: a single
    // factor (excitation = detection = lambda_nm).
    const double NA = static_cast<double>(config.NA);
    const double n = static_cast<double>(config.ns);
    if (NA <= 0.0 || n <= 0.0 || lambda_nm <= 0.0)
        return {};

    double excitation = lambda_nm;
    if (config.getModalityType() == MicroscopeModality::CONFOCAL) {
        if (const auto* modalityCfg = config.modality<ConfocalModalityConfig>()) {
            // Degenerate fallback (excitation <= 0): the excitation integral
            // uses the detection wavelength (see ConfocalModality::composeSlice).
            excitation = modalityCfg->excitationWavelength_nm > 0.0f
                ? static_cast<double>(modalityCfg->excitationWavelength_nm)
                : lambda_nm;
        }
    }

    // Widefield: a single incoherent integral, cutoff 2*NA/lambda (lateral)
    // and n*(1-cos(alpha))/lambda (axial). Confocal: the product of the
    // excitation and detection integrals, so the two cutoffs are summed.
    const double s = axialSinClamped(NA, n);
    if (s <= 0.0)
        return {};
    const double lateralCutoff = config.getModalityType() == MicroscopeModality::CONFOCAL
        ? 2.0 * NA * (1.0 / excitation + 1.0 / lambda_nm)
        : 2.0 * NA / lambda_nm;
    const double axialCutoff = config.getModalityType() == MicroscopeModality::CONFOCAL
        ? n * (1.0 - std::sqrt(1.0 - s * s)) * (1.0 / excitation + 1.0 / lambda_nm)
        : n * (1.0 - std::sqrt(1.0 - s * s)) / lambda_nm;
    if (lateralCutoff <= 0.0 || axialCutoff <= 0.0)
        return {};

    return {
        1.0 / lateralCutoff,   // Abbe (finest period), lateral
        0.5 / lateralCutoff,   // Nyquist pitch, lateral
        1.0 / axialCutoff,     // Abbe (finest period), axial
        0.5 / axialCutoff,     // Nyquist pitch, axial
    };
}

void logSampling(const GibsonLanniPSFConfig& config,
                 double lambda_nm,
                 int criterion,
                 const ApproxResolution& resolution,
                 double pixelLateral_nm,
                 double pixelAxial_nm) {
    const SamplingLimits lim = samplingLimits(config, lambda_nm);
    if (lim.nyquistLateral_nm <= 0.0 || lim.nyquistAxial_nm <= 0.0)
        return;

    const std::string label = modalityLabel(config);

    // The resolution is the PSF width the optics deliver (the criterion named
    // in the config); the sampling limits are the OTF-derived Abbe (finest
    // period) and Nyquist (half of it) pitches the voxel grid must not
    // exceed. All three, plus the actual pitch, in one line.
    spdlog::info("GibsonLanni [{}] optical resolution {} {:.1f}nm lateral / {:.1f}nm axial; "
                 "sampling limit (Abbe/Nyquist) {:.1f}/{:.1f}nm lateral, {:.1f}/{:.1f}nm axial; "
                 "voxel pitch {:.1f}/{:.1f}nm",
                 label, criterionName(criterion),
                 resolution.lateral_nm, resolution.axial_nm,
                 lim.abbeLateral_nm, lim.nyquistLateral_nm,
                 lim.abbeAxial_nm, lim.nyquistAxial_nm,
                 pixelLateral_nm, pixelAxial_nm);

    auto report = [&lim](const char* axis, double pitch_nm, double abbe_nm, double nyquist_nm) {
        if (pitch_nm <= 0.0)
            return;
        if (pitch_nm > nyquist_nm)
            spdlog::warn("GibsonLanni: {} voxel pitch {:.1f}nm exceeds the Nyquist limit {:.1f}nm "
                         "(Abbe finest period {:.1f}nm); the PSF is undersampled and may alias",
                         axis, pitch_nm, nyquist_nm, abbe_nm);
        else if (pitch_nm < 0.5 * nyquist_nm)
            spdlog::info("GibsonLanni: {} voxel pitch {:.1f}nm is more than 2x finer than the Nyquist limit {:.1f}nm - "
                         "the PSF is oversampled; a coarser pitch (up to {:.1f}nm) would still be lossless",
                         axis, pitch_nm, nyquist_nm, nyquist_nm);
    };
    report("lateral", pixelLateral_nm, lim.abbeLateral_nm, lim.nyquistLateral_nm);
    report("axial", pixelAxial_nm, lim.abbeAxial_nm, lim.nyquistAxial_nm);
}

} // namespace PSFBoxSizing
