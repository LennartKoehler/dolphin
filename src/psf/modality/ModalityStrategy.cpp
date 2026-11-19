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

#include "dolphin/psf/modality/ModalityStrategy.h"

#include "dolphin/psf/configs/ModalityConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/modality/PSFModel.h"
#include "dolphin/ThreadPool.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <stdexcept>
#include <variant>
#include <vector>

#include <spdlog/spdlog.h>
#include <itkImage.h>
#include <itkImageRegionIterator.h>

namespace {

// FullSlice -> SliceData: slices are stored centered on the optical axis as
// square odd grids, so the lateral extent is (width-1)/2.
SliceData fullSliceToSliceData(const FullSlice& full) {
    if (full.width != full.height || full.width % 2 == 0 || full.data.size() < full.width * full.height)
        throw std::runtime_error("FullSlice must be a square grid with odd width to compose into the centered slice grid");
    return {full.data, (full.width - 1) / 2};
}

// Scale the PSF so its total integral is 1 (unit flux). No-op if the sum is 0.
void normalizeUnitFlux(PSF& psf) {
    double sum = 0.0;
    for (auto it = psf.cbegin(); it != psf.cend(); ++it)
        sum += *it;

    if (sum <= 0.0)
        return;

    const float scale = static_cast<float>(1.0 / sum);
    for (auto it = psf.begin(); it != psf.end(); ++it)
        *it *= scale;
}

} // namespace

const ModalityStrategy& ModalityStrategy::forConfig(const PSFConfig& config) {
    switch (config.getModalityType()){
        case MicroscopeModality::CONFOCAL:
            return ConfocalModality::instance();
        case MicroscopeModality::WIDEFIELD:
        default:
            return WidefieldModality::instance();
    }
}

void ModalityStrategy::validateConfig(const PSFConfig& config) const {
    // The radial profiles are tabulated at oversampling_factor samples per
    // lateral pixel; a non-positive factor truncates to zero samples and
    // divides by zero in the radial blit.
    if (!(config.oversampling_factor > 0.0f))
        throw std::runtime_error("ModalityStrategy: oversampling_factor must be positive, got "
                                 + std::to_string(config.oversampling_factor));
}

PSF ModalityStrategy::generate(const PSFModel& model,
                               const PSFConfig& config,
                               ThreadPool& threadPool,
                               ProgressTracking& progressTracker) const {
    // Reject nonsense before any box sizing or tabulation runs.
    validateConfig(config);
    model.validateConfig(config);

    const bool autoSize = config.autoSize || config.sizeX == 0 || config.sizeY == 0 || config.sizeZ == 0;

    // Sized geometry of the PSF box. In auto-size mode config.sizeX/sizeZ are
    // the size of the IMAGE the PSF is produced for (sizeZ drives the imaged
    // defocus range), not the PSF box; the box comes from the model. In
    // fixed mode the config sizes are the box and slices center on it.
    ModelBox box;
    if (autoSize) {
        box = model.autoBox(config, detectionLambda(model, config));
    } else {
        box.width = config.sizeX;
        box.height = config.sizeY;
        box.depth = config.sizeZ;
        box.lateralHalf = (std::min(config.sizeX, config.sizeY) - 1) / 2;
        // Center on the middle voxel like BIG ((nz-1)/2), not the floor
        // division of sizeZ/2, which shifts the axial center by half a pixel
        // for even sizeZ.
        box.firstOffset = -(static_cast<long>((config.sizeZ - 1) / 2));
    }

    model.init(config, box.width, box.height, shortestLambda(model, config));

    // forcedCutoff > 0 disables the value-based early stop, so no voxel
    // inside the box is left as a zero placeholder.
    progressTracker.setMax(static_cast<float>(box.depth));

    std::vector<std::future<SliceData>> sliceFutures;
    sliceFutures.reserve(box.depth);
    for (size_t k = 0; k < box.depth; ++k) {
        std::shared_ptr<const PSFConfig> sliceCfg = model.sliceConfig(config, box.firstOffset + static_cast<long>(k));
        sliceFutures.emplace_back(threadPool.enqueue([this, &model, &progressTracker, cfg = std::move(sliceCfg), lateralHalf = box.lateralHalf]() {
            SliceData slice = composeSlice(model, *cfg, lateralHalf);
            progressTracker.add(1);
            return slice;
        }));
    }

    ImageType::Pointer itkImage = ImageType::New();
    ImageType::SizeType imgSize;
    imgSize[0] = box.width;
    imgSize[1] = box.height;
    imgSize[2] = box.depth;
    ImageType::IndexType start;
    start.Fill(0);
    ImageType::RegionType region;
    region.SetSize(imgSize);
    region.SetIndex(start);
    itkImage->SetRegions(region);
    itkImage->Allocate();
    itkImage->FillBuffer(0.0f);

    // Fixed-size slices are centered on the image; auto-size slices already
    // span the full box, so the offset is zero.
    const size_t offsetX = autoSize ? 0 : config.sizeX / 2 - box.lateralHalf;
    const size_t offsetY = autoSize ? 0 : config.sizeY / 2 - box.lateralHalf;

    for (size_t k = 0; k < box.depth; ++k) {
        SliceData slice = sliceFutures[k].get();
        const size_t sliceW = 2 * slice.lateralCutoff + 1;
        for (size_t y = 0; y < sliceW && y < box.height; y++) {
            for (size_t x = 0; x < sliceW && x < box.width; x++) {
                itk::Index<3> idx;
                idx[0] = static_cast<itk::IndexValueType>(x + offsetX);
                idx[1] = static_cast<itk::IndexValueType>(y + offsetY);
                idx[2] = static_cast<itk::IndexValueType>(k);
                itkImage->SetPixel(idx, slice.data[y * sliceW + x]);
            }
        }
    }

    PSF psf(std::move(itkImage), config.ID);
    if (autoSize && config.normalizePSF)
        normalizeUnitFlux(psf);
    return psf;
}

SliceData ModalityStrategy::blitRadialProfile(const std::vector<double>& h, size_t samplesPerPixel, size_t cutoff) {
    const size_t gridSize = 2 * cutoff + 1;
    std::vector<float> sliceData(gridSize * gridSize, 0.0f);

    for (size_t y = 0; y < gridSize; y++) {
        for (size_t x = 0; x < gridSize; x++) {
            double dx = static_cast<double>(x) - static_cast<double>(cutoff);
            double dy = static_cast<double>(y) - static_cast<double>(cutoff);
            double rPixel = std::sqrt(dx * dx + dy * dy);
            size_t index = static_cast<size_t>(std::floor(rPixel * static_cast<double>(samplesPerPixel)));

            double value = 0.0;
            if (index + 1 < h.size()) {
                double r0 = static_cast<double>(index) / static_cast<double>(samplesPerPixel);
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r0) * static_cast<double>(samplesPerPixel);
            } else if (index < h.size()) {
                value = h[index];
            }

            sliceData[y * gridSize + x] = static_cast<float>(value);
        }
    }

    return {std::move(sliceData), cutoff};
}

// ---------------------------------------------------------------------------
// Widefield
// ---------------------------------------------------------------------------

const WidefieldModality& WidefieldModality::instance() {
    static const WidefieldModality strategy;
    return strategy;
}

std::optional<double> WidefieldModality::detectionLambda(const PSFModel& model, const PSFConfig& config) const {
    // The model owns its detection wavelength; std::nullopt for
    // wavelength-less models.
    return model.detectionLambda(config);
}

std::optional<double> WidefieldModality::shortestLambda(const PSFModel& model, const PSFConfig& config) const {
    return detectionLambda(model, config);
}

SliceData WidefieldModality::composeSlice(const PSFModel& model,
                                          const PSFConfig& sliceCfg,
                                          size_t forcedCutoff) const {
    const size_t samplesPerPixel = static_cast<size_t>(sliceCfg.oversampling_factor);
    const size_t maxLateral = forcedCutoff > 0 ? forcedCutoff : (std::min(sliceCfg.sizeX, sliceCfg.sizeY) / 2);

    // Widefield needs no two-factor sampling headroom; it also accepts a
    // model that reports its slice as an explicit FullSlice. The lambda is
    // the model's own detection wavelength (nullopt for wavelength-less
    // models, which then use whatever default they carry).
    const ModelSlice slice = model.intensitySlice(sliceCfg, model.detectionLambda(sliceCfg),
                                                  samplesPerPixel, maxLateral, forcedCutoff);
    if (const auto* full = std::get_if<FullSlice>(&slice))
        return fullSliceToSliceData(*full);

    const RadialSlice& radial = std::get<RadialSlice>(slice);
    const size_t cutoff = forcedCutoff > 0 ? forcedCutoff
        : static_cast<size_t>(std::ceil(static_cast<double>(radial.h.size() - 1) / static_cast<double>(radial.samplesPerPixel)));
    return blitRadialProfile(radial.h, radial.samplesPerPixel, cutoff);
}

// ---------------------------------------------------------------------------
// Confocal
// ---------------------------------------------------------------------------

const ConfocalModality& ConfocalModality::instance() {
    static const ConfocalModality strategy;
    return strategy;
}

PSF ConfocalModality::generate(const PSFModel& model,
                               const PSFConfig& config,
                               ThreadPool& threadPool,
                               ProgressTracking& progressTracker) const {
    // Once-per-generation setup report: the modality's own parameters only
    // (validation, including the excitation/emission warnings, has already
    // run in validateConfig via the base generate()).
    // Model-numerics diagnostics (e.g. NA*rho/ns) belong to the model's init.
    if (const ConfocalModalityConfig* modalityCfg = config.modality<ConfocalModalityConfig>()) {
        spdlog::debug("Confocal PSF: excitation {:.1f} nm, emission {:.1f} nm, pinhole radius {:.2f} um ({})",
                      excitationLambda(config),
                      static_cast<double>(modalityCfg->emissionWavelength_nm),
                      static_cast<double>(modalityCfg->pinholeRadius_um),
                      modalityCfg->pinholeRadius_um > 0.0f ? "finite-pinhole convolution" : "point pinhole");
    }
    return ModalityStrategy::generate(model, config, threadPool, progressTracker);
}

void ConfocalModality::validateConfig(const PSFConfig& config) const {
    ModalityStrategy::validateConfig(config);

    const ConfocalModalityConfig* modalityCfg = config.modality<ConfocalModalityConfig>();
    if (!modalityCfg)
        throw std::runtime_error("ConfocalModality: the PSF config does not carry a confocal modality config");

    // Without a detection wavelength there is no confocal product at all.
    if (!(modalityCfg->emissionWavelength_nm > 0.0f))
        throw std::runtime_error("ConfocalModality: emission_wavelength_nm must be positive, got "
                                 + std::to_string(modalityCfg->emissionWavelength_nm));
    // A negative pinhole radius is meaningless; 0 is the defined point-pinhole
    // limit.
    if (modalityCfg->pinholeRadius_um < 0.0f)
        throw std::runtime_error("ConfocalModality: pinhole_radius_um must not be negative, got "
                                 + std::to_string(modalityCfg->pinholeRadius_um));

    // Degenerate fallback (excitation <= 0 -> emission wavelength, product
    // degenerates toward widefield) and Stokes-shift violations are defined
    // behavior: warn, keep generating.
    if (modalityCfg->excitationWavelength_nm <= 0.0f)
        spdlog::warn("Confocal PSF: excitation_wavelength_nm not set (<= 0); falling back to the emission wavelength ({:.1f} nm). The product then degenerates toward the widefield model - set excitation_wavelength_nm for a true confocal PSF",
                     static_cast<double>(modalityCfg->emissionWavelength_nm));
    else if (modalityCfg->excitationWavelength_nm > modalityCfg->emissionWavelength_nm)
        spdlog::warn("Confocal PSF: excitation_wavelength_nm ({:.1f}) exceeds emission_wavelength_nm ({:.1f}) - violates the Stokes shift, the confocal product will be wider than physically expected",
                     static_cast<double>(modalityCfg->excitationWavelength_nm),
                     static_cast<double>(modalityCfg->emissionWavelength_nm));
}

double ConfocalModality::excitationLambda(const PSFConfig& cfg) {
    // Configs without an excitation wavelength fall back to the emission
    // wavelength, keeping the confocal path a strict product of two identical
    // integrals in that (degenerate) case.
    const ConfocalModalityConfig* modalityCfg = cfg.modality<ConfocalModalityConfig>();
    if (!modalityCfg)
        throw std::runtime_error("ConfocalModality: the PSF config does not carry a confocal modality config");
    if (modalityCfg->excitationWavelength_nm <= 0.0f)
        return static_cast<double>(modalityCfg->emissionWavelength_nm);
    return static_cast<double>(modalityCfg->excitationWavelength_nm);
}

std::optional<double> ConfocalModality::detectionLambda(const PSFModel& model, const PSFConfig& config) const {
    const ConfocalModalityConfig* modalityCfg = config.modality<ConfocalModalityConfig>();
    if (!modalityCfg)
        throw std::runtime_error("ConfocalModality: the PSF config does not carry a confocal modality config");
    return static_cast<double>(modalityCfg->emissionWavelength_nm);
}

std::optional<double> ConfocalModality::shortestLambda(const PSFModel& model, const PSFConfig& config) const {
    return std::min(*detectionLambda(model, config), excitationLambda(config));
}

SliceData ConfocalModality::composeSlice(const PSFModel& model,
                                         const PSFConfig& sliceCfg,
                                         size_t forcedCutoff) const {
    // One cast establishes the modality config for this slice; the confocal
    // parameters (emission, excitation, pinhole) live on it.
    const ConfocalModalityConfig* modalityCfg = sliceCfg.modality<ConfocalModalityConfig>();
    if (!modalityCfg)
        throw std::runtime_error("ConfocalModality: the PSF config does not carry a confocal modality config");

    // The confocal product needs two wavelengths; a model without any
    // wavelength concept cannot participate.
    if (!model.detectionLambda(sliceCfg))
        throw std::runtime_error("ConfocalModality: the model does not support wavelengths, which confocal image formation requires");

    // The product of two oscillatory profiles interleaves their zero sets, so
    // its finest features are at half the period of either factor: double the
    // radial sampling density to resolve it.
    const size_t samplesPerPixel = 2 * static_cast<size_t>(sliceCfg.oversampling_factor);
    const size_t maxLateral = forcedCutoff > 0 ? forcedCutoff : (std::min(sliceCfg.sizeX, sliceCfg.sizeY) / 2);

    // Degenerate fallback (excitation <= 0): the excitation integral uses the
    // emission wavelength and the product degenerates toward widefield.
    const double excitation_nm = modalityCfg->excitationWavelength_nm > 0.0f
        ? static_cast<double>(modalityCfg->excitationWavelength_nm)
        : static_cast<double>(modalityCfg->emissionWavelength_nm);

    const ModelSlice excSlice = model.intensitySlice(sliceCfg, excitation_nm,
                                                     samplesPerPixel, maxLateral, forcedCutoff);
    const ModelSlice detSlice = model.intensitySlice(sliceCfg, static_cast<double>(modalityCfg->emissionWavelength_nm),
                                                     samplesPerPixel, maxLateral, forcedCutoff);

    // Reject the unsupported model shape explicitly rather than silently
    // composing garbage.
    if (std::holds_alternative<FullSlice>(excSlice) || std::holds_alternative<FullSlice>(detSlice))
        throw std::runtime_error("ConfocalModality: model returned a non-radial slice, which confocal composition does not support");

    const RadialSlice& exc = std::get<RadialSlice>(excSlice);
    RadialSlice det = std::get<RadialSlice>(detSlice);

    det.h = applyDetection(*modalityCfg, std::move(det.h),
                           static_cast<double>(det.samplesPerPixel),
                           det.lateralPitchNm);

    // Per-profile early stops (forcedCutoff == 0) can end the profiles at
    // different radii; the product is only defined on the common support, and
    // beyond a threshold-stopped factor it is negligible anyway. The detection
    // modification deliberately does not extend the support, so the common
    // support still covers the product.
    const size_t len = std::min(exc.h.size(), det.h.size());
    std::vector<double> h(len);
    for (size_t i = 0; i < len; ++i)
        h[i] = exc.h[i] * det.h[i];

    const size_t cutoff = forcedCutoff > 0 ? forcedCutoff
        : static_cast<size_t>(std::ceil(static_cast<double>(len ? len - 1 : 0) / static_cast<double>(det.samplesPerPixel)));
    return blitRadialProfile(h, det.samplesPerPixel, cutoff);
}

std::vector<double> ConfocalModality::applyDetection(const ConfocalModalityConfig& modalityConfig,
                                                     std::vector<double> hDetection,
                                                     double samplesPerPixel,
                                                     double pixelSizeLateral_nm) const {
    // Finite pinhole: average the detection PSF over the pinhole disc at the
    // sample plane. A zero radius is the point-pinhole limit and leaves the
    // profile untouched.
    if (modalityConfig.pinholeRadius_um > 0.0f)
        return convolveRadialWithPinhole(std::move(hDetection), samplesPerPixel,
                                         pixelSizeLateral_nm,
                                         static_cast<double>(modalityConfig.pinholeRadius_um));
    return hDetection;
}

double ConfocalModality::sampleRadialProfile(const std::vector<double>& h,
                                             double samplesPerPixel,
                                             double pixelSizeLateral_nm,
                                             double r_nm) {
    if (h.empty()) return 0.0;
    // Profile sample index for a physical radius r_nm:
    //   r_nm = (index / samplesPerPixel) * pixelSizeLateral_nm
    const double t = (r_nm / pixelSizeLateral_nm) * samplesPerPixel;
    if (t < 0.0) return h[0];
    const size_t i = static_cast<size_t>(t);
    if (i + 1 >= h.size())
        return 0.0;  // beyond tabulated support
    const double f = t - static_cast<double>(i);
    return h[i] + (h[i + 1] - h[i]) * f;
}

std::vector<double> ConfocalModality::convolveRadialWithPinhole(
        const std::vector<double>& h,
        double samplesPerPixel,
        double pixelSizeLateral_nm,
        double pinholeRadius_um) {
    // Average I_det over the pinhole disc in the sample plane:
    //   I_pin(r) = (1 / (pi R^2)) * \int_0^R s ds \int_0^{2pi} I_det(|r - s|) dtheta
    // The disc is circular, so only the radial distance from the output point
    // to each source point matters. For an output point at radius r and a
    // source point at radius s offset by angle theta, that distance is
    //   d = sqrt(r^2 + s^2 - 2 r s cos(theta)).
    //
    // Quadrature: s in [0, R] (Simpson, fine enough for the detection PSF's
    // lateral variation over the disc), theta in [0, pi] doubled by symmetry.
    // The result is normalised by the disc area so the R -> 0 limit returns
    // the input amplitude.
    if (pinholeRadius_um <= 0.0 || h.empty())
        return h;

    const double rPinhole_nm = pinholeRadius_um * 1000.0;  // um -> nm, sample plane

    // Resolve the pinhole diameter radially and its circumference
    // azimuthally. The detection PSF's finest lateral feature is ~ the
    // sampling step, so a handful of samples per pinhole radius plus a fixed
    // azimuthal count is plenty; the pinhole is small (a few hundred nm).
    const double samplesPerNm = samplesPerPixel / pixelSizeLateral_nm;
    // The intensity profile varies smoothly over a sub-Airy pinhole; a few
    // hundred radial samples and a modest azimuthal ring resolve the disc.
    const size_t nS = std::clamp<size_t>(
        static_cast<size_t>(std::ceil(rPinhole_nm * samplesPerNm)), 32, 256) & ~size_t(1);
    constexpr size_t nTheta = 64;

    // Simpson nodes on [0, R] (plus the r=0 endpoint handled analytically).
    const double ds = rPinhole_nm / static_cast<double>(nS);
    std::vector<double> sNode(nS + 1), sWeight(nS + 1);
    for (size_t i = 0; i <= nS; ++i) {
        sNode[i] = static_cast<double>(i) * ds;
        // composite Simpson weights (0 and nS are half-weighted by /3 rule)
        double w;
        if (i == 0 || i == nS) w = 1.0;
        else w = (i % 2 == 1) ? 4.0 : 2.0;
        sWeight[i] = w * ds / 3.0;
    }

    const double dTheta = M_PI / static_cast<double>(nTheta);  // [0, pi], doubled
    std::vector<double> thetaNode(nTheta + 1), thetaWeight(nTheta + 1);
    for (size_t j = 0; j <= nTheta; ++j) {
        thetaNode[j] = static_cast<double>(j) * dTheta;
        double w;
        if (j == 0 || j == nTheta) w = 1.0;
        else w = (j % 2 == 1) ? 4.0 : 2.0;
        thetaWeight[j] = w * dTheta / 3.0;
    }

    // Radial output grid identical to the input (r = sample / samplesPerPixel,
    // expressed in nm).
    const size_t nR = h.size();
    std::vector<double> out(nR);

    const double area = M_PI * rPinhole_nm * rPinhole_nm;

    for (size_t ir = 0; ir < nR; ++ir) {
        const double r_nm = (static_cast<double>(ir) / samplesPerPixel) * pixelSizeLateral_nm;
        double acc = 0.0;
        for (size_t i = 0; i <= nS; ++i) {
            const double s = sNode[i];
            double inner = 0.0;
            for (size_t j = 0; j <= nTheta; ++j) {
                const double th = thetaNode[j];
                const double d2 = r_nm * r_nm + s * s - 2.0 * r_nm * s * std::cos(th);
                const double d = std::sqrt(std::max(0.0, d2));
                inner += thetaWeight[j] * sampleRadialProfile(h, samplesPerPixel,
                                                              pixelSizeLateral_nm, d);
            }
            // theta integral over [0, 2*pi] = 2 * [0, pi].
            acc += sWeight[i] * s * 2.0 * inner;
        }
        out[ir] = acc / area;
    }

    return out;
}
