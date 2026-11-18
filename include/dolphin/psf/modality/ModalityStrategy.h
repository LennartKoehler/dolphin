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
#include <vector>

#include "dolphin/ProgressTracking.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/psf/configs/ModalityConfig.h"

class PSFConfig;
class PSFModel;
class ThreadPool;

/** Composed lateral slice grid of (2*lateralCutoff+1)^2 voxels, row-major,
 *  centered on the optical axis. */
struct SliceData {
    std::vector<float> data;
    size_t lateralCutoff = 0;
};

/**
 * Modality composition layer - OWNS full PSF assembly.
 *
 * A microscope modality describes HOW PSF-model slices are combined into the
 * observed PSF; a PSFModel describes how ONE wavelength's intensity slice is
 * computed. The dependency is one-directional: the modality asks the model
 * for slices and assembles the PSF itself - the model never sees modalities,
 * and the generator only selects the strategy:
 *
 *   widefield  I = slice(emission)
 *   confocal   I = slice(excitation) * pinhole(slice(emission))
 *
 * generate() implements the shared pipeline (box sizing, model preparation,
 * parallel per-slice composition, progress, unit-flux normalization) purely
 * against the abstract PSFConfig and PSFModel interfaces - all model-specific
 * knowledge (auto-size box, defocus parameterization, wavelengths) comes from
 * the model; subclasses only provide composeSlice() and the wavelength
 * policy. A new modality is one subclass plus a case in forConfig() - no new
 * PSF model or generator. Modalities sharing the confocal image formation
 * (spinning disk) subclass ConfocalModality and override applyDetection().
 *
 * Strategies are stateless singletons; adding one never invalidates the
 * others.
 */
class ModalityStrategy {
public:
    virtual ~ModalityStrategy() = default;

    /** Full PSF assembly for this modality: box sizing (auto or fixed),
     *  model preparation, parallel per-slice composition, image assembly and
     *  unit-flux normalization (auto-size path only, preserving the historic
     *  behavior of the fixed-size path). The config is type-erased here;
     *  strategies are scoped to Gibson-Lanni configs and cast once. Modalities
     *  with once-per-generation setup reporting of their own override this to
     *  log and delegate (see ConfocalModality). */
    virtual PSF generate(const PSFModel& model,
                         const PSFConfig& config,
                         ThreadPool& threadPool,
                         ProgressTracking& progressTracker) const;

    /** Strategy implementing a config's microscope modality. */
    static const ModalityStrategy& forConfig(const PSFConfig& config);

protected:
    /** Detection wavelength this modality observes at, and the shortest
     *  wavelength it will request - std::nullopt for modalities over
     *  wavelength-less models. Drives auto-size box resolution and the
     *  model's init; the composition itself passes its own lambdas. */
    virtual std::optional<double> detectionLambda(const PSFModel& model,
                                                  const PSFConfig& config) const = 0;
    virtual std::optional<double> shortestLambda(const PSFModel& model,
                                                 const PSFConfig& config) const = 0;

    /** Compose one lateral slice at the defocus carried by the slice config.
     *  forcedCutoff > 0 pins the lateral half-width (the orchestrator passes
     *  the box half-extent); 0 falls back to the config-driven cutoff. Pure
     *  computation - progress accounting stays with the orchestrator. */
    virtual SliceData composeSlice(const PSFModel& model,
                                   const PSFConfig& sliceCfg,
                                   size_t forcedCutoff) const = 0;

    /** Samples a radial profile onto the (2*cutoff+1)^2 slice grid. */
    static SliceData blitRadialProfile(const std::vector<double>& h,
                                       size_t samplesPerPixel,
                                       size_t cutoffPixels);
};

/** Widefield image formation: a single detection integral at the emission
 *  wavelength. Accepts both slice representations; a FullSlice passes
 *  through unchanged (it already IS the widefield slice). */
class WidefieldModality : public ModalityStrategy {
public:
    static const WidefieldModality& instance();

protected:
    std::optional<double> detectionLambda(const PSFModel& model,
                                          const PSFConfig& config) const override;
    std::optional<double> shortestLambda(const PSFModel& model,
                                         const PSFConfig& config) const override;
    SliceData composeSlice(const PSFModel& model,
                           const PSFConfig& sliceCfg,
                           size_t forcedCutoff) const override;
};

/**
 * Confocal image formation: voxel-wise product of the excitation integral
 * (at excitationLambda(config)) and the detection integral (at the emission
 * wavelength). With a finite pinhole (pinholeRadius_nm > 0) the detection
 * profile is first averaged over the pinhole disc at the sample plane;
 * otherwise it is the point-pinhole (Airy-unit) product.
 */
class ConfocalModality : public ModalityStrategy {
public:
    static const ConfocalModality& instance();

    /** Wavelength actually used for the excitation integral. Configs without
     *  an excitation wavelength fall back to the emission wavelength,
     *  degenerating to a product of two identical integrals. */
    static double excitationLambda(const PSFConfig& cfg);

    /** One setup report per generation (wavelengths, pinhole), then the
     *  shared assembly pipeline. The log uses only the modality's own
     *  parameters, so it works for any confocal-capable model; model
     *  numerics diagnostics stay in the model's init. */
    PSF generate(const PSFModel& model,
                 const PSFConfig& config,
                 ThreadPool& threadPool,
                 ProgressTracking& progressTracker) const override;

protected:
    std::optional<double> detectionLambda(const PSFModel& model,
                                          const PSFConfig& config) const override;
    std::optional<double> shortestLambda(const PSFModel& model,
                                         const PSFConfig& config) const override;
    SliceData composeSlice(const PSFModel& model,
                           const PSFConfig& sliceCfg,
                           size_t forcedCutoff) const override;

    /** Detection-profile modification hook. The default applies the finite
     *  pinhole disc average (pinholeRadius_nm > 0, no-op otherwise).
     *  Modality variants with a different detection geometry (spinning disk
     *  microlenses, crosstalk) extend ConfocalModalityConfig with their own
     *  parameters and override this instead of composeSlice. */
    virtual std::vector<double> applyDetection(const ConfocalModalityConfig& modalityConfig,
                                               std::vector<double> hDetection,
                                               double samplesPerPixel,
                                               double pixelSizeLateral_nm) const;

    /** Linear interpolation of a radial profile h at physical radius r_nm.
     *  Zero beyond the tabulated support. */
    static double sampleRadialProfile(const std::vector<double>& h,
                                      double samplesPerPixel,
                                      double pixelSizeLateral_nm,
                                      double r_nm);

    /** Replace a radial intensity profile by its finite-pinhole average:
     *  I_pin(r) = (1/(pi R^2)) \int_{disc R} I(|r - s|) d^2s, evaluated in
     *  polar coordinates. R is the pinhole radius at the sample plane in nm.
     *  Normalised by the pinhole area so the point-pinhole limit (R -> 0)
     *  preserves the profile amplitude. Returns a profile on the same radial
     *  grid as the input. */
    static std::vector<double> convolveRadialWithPinhole(const std::vector<double>& h,
                                                         double samplesPerPixel,
                                                         double pixelSizeLateral_nm,
                                                         double pinholeRadius_nm);
};
