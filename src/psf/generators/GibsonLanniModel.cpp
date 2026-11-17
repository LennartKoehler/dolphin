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

#include "dolphin/psf/generators/GibsonLanniModel.h"

#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/modality/ModalityStrategy.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace {

// True if the cached radial profile already satisfies the request: either a
// forced lateral cutoff is fully covered, or the profile already crossed the
// threshold (or reached the sample cap).
bool cachedProfileComplete(const std::vector<double>& h, size_t forcedCutoff, size_t maxSamples, double threshold) {
    if (h.empty()) return false;
    if (forcedCutoff > 0) return h.size() > maxSamples;
    if (h.size() > 1 && h.back() < threshold * h[0]) return true;
    return h.size() >= maxSamples;
}

} // namespace

void GibsonLanniModel::invalidateCache() {
    std::lock_guard<std::mutex> lock(cacheMutex);
    cachedRadialProfiles.clear();
    besselInitialized = false;
}

void GibsonLanniModel::init(const GibsonLanniPSFConfig& config,
                               size_t sliceWidth, size_t sliceHeight) const {
    // Once-per-generation diagnostics.
    // The old per-evaluation "NA*rho/ns > 1" log fired on every integrand
    // evaluation (thousands of lines per slice). Emit it once per generation
    // instead: the violation is monotonic in rho, so if it holds at the upper
    // bound b it holds somewhere in the domain.
    const double rhoMax = integrationUpperBound(config);
    if (config.NA * rhoMax / config.ns > 1.0)
        spdlog::warn("GibsonLanni: NA*rho/ns > 1 within the integration domain (ns={}, NA={}, rho_max={})",
                     config.ns, config.NA, rhoMax);

    if (config.isConfocalPSF()) {
        const ConfocalModalityConfig* modalityCfg = config.modality<ConfocalModalityConfig>();
        const float excitation_nm = modalityCfg ? modalityCfg->excitationWavelength_nm : 0.0f;
        const float pinholeRadius_nm = modalityCfg ? modalityCfg->pinholeRadius_nm : 0.0f;
        const double emission_nm = modalityCfg ? static_cast<double>(modalityCfg->emissionWavelength_nm)
                                               : static_cast<double>(config.wavelength_nm);
        if (excitation_nm <= 0.0f) {
            spdlog::warn("GibsonLanni confocal PSF: excitation_wavelength_nm not set (<= 0); falling back to the emission wavelength ({:.1f} nm). The product then degenerates toward the widefield model - set excitation_wavelength_nm for a true confocal PSF",
                         emission_nm);
        }
        const double airyRadius_nm = 0.61 * emission_nm / config.NA;
        const double pinholeAU = pinholeRadius_nm / airyRadius_nm;
        spdlog::debug("GibsonLanni confocal PSF: excitation {:.1f} nm, emission {:.1f} nm, pinhole radius {:.1f} nm = {:.2f} AU ({})",
                     ConfocalModality::excitationLambda(config), emission_nm,
                     pinholeRadius_nm, pinholeAU,
                     pinholeRadius_nm > 0.0f ? "finite-pinhole convolution" : "point pinhole");
    }

    initBesselHelper(config, sliceWidth, sliceHeight);
}

ModelSlice GibsonLanniModel::intensitySlice(const GibsonLanniPSFConfig& sliceCfg,
                                            double lambda_nm,
                                            size_t samplesPerPixel,
                                            size_t maxLateralPixels,
                                            size_t forcedCutoff) const {
    if (!besselInitialized)
        throw std::runtime_error("GibsonLanniModel: prepare() must run before intensitySlice()");

    std::vector<double> h = radialProfile(sliceCfg, lambda_nm, samplesPerPixel, maxLateralPixels, forcedCutoff);
    return RadialSlice{std::move(h), samplesPerPixel};
}

void GibsonLanniModel::initBesselHelper(const GibsonLanniPSFConfig& config,
                                        size_t sizeX, size_t sizeY) const {
    double nx = static_cast<double>(sizeX);
    double ny = static_cast<double>(sizeY);
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    // One shared J0 table serves every requested wavelength: it must cover
    // the largest argument k0*NA*r*rho, which grows as the wavelength
    // shrinks - hence the caller passes the shortest requested lambda.
    double k0 = 2.0 * M_PI / config.wavelength_nm;
    size_t maxRadius = static_cast<size_t>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;

    double max_k0NAr = k0 * config.NA * maxRadius * config.pixelSizeLateral_nm;
    double maxRho = std::min(float(1), config.ns / config.NA);

    // maxRadius covers the image corner, so any sizeX/sizeY asymmetry is included
    double maxValue = max_k0NAr * maxRho;
    double dx = 0.1;

    besselHelper.init(maxValue, dx);
    besselInitialized = true;
}

double GibsonLanniModel::integrationUpperBound(const GibsonLanniPSFConfig& cfg) {
    const double NA = cfg.NA;
    // Each sqrt(1 - (NA*rho/n)^2) term requires rho <= n/NA, so the upper
    // bound is the tightest such limit across every medium in the path (and
    // 1.0 at the pupil edge). For standard configs the sample (ns) is the
    // smallest RI, so this reduces to the original min(1.0, ns/NA); only
    // exotic configs tighten the bound.
    return std::min({1.0, cfg.ns / NA, cfg.ni / NA, cfg.ni0 / NA,
                     cfg.ng / NA, cfg.ng0 / NA});
}

std::vector<double> GibsonLanniModel::radialProfile(
        const GibsonLanniPSFConfig& sliceCfg, double lambda_nm,
        size_t samplesPerPixel, size_t maxLateralPixels, size_t forcedCutoff) const {
    const double threshold = static_cast<double>(sliceCfg.cutoffThreshold);
    const size_t maxSamples = maxLateralPixels * samplesPerPixel;
    const std::pair<double, double> cacheKey(sliceCfg.ti_nm, lambda_nm);

    std::vector<double> h;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cachedRadialProfiles.find(cacheKey);
        if (it != cachedRadialProfiles.end())
            h = it->second;
    }

    if (!cachedProfileComplete(h, forcedCutoff, maxSamples, threshold)) {
        // Fixed-node quadrature: phase tabulated once, then one MAC per radius.
        // The integrand is fully determined by (k0, geometry); route the
        // wavelength through a config copy so Quadrature/Integrand stay
        // parameter-free.
        GibsonLanniPSFConfig cfg = sliceCfg;
        cfg.wavelength_nm = static_cast<float>(lambda_nm);

        const double k0 = 2.0 * M_PI / lambda_nm;
        GibsonLanniQuadrature quadrature;
        const double b = integrationUpperBound(cfg);
        const double maxR_nm = (static_cast<double>(maxSamples) / static_cast<double>(samplesPerPixel)) * cfg.pixelSizeLateral_nm;
        const double maxArg = k0 * cfg.NA * maxR_nm * b;
        quadrature.init(cfg, b, GibsonLanniQuadrature::nodeCount(maxArg, cfg.integration_accuracy));

        h.clear();
        h.push_back(quadrature.evaluateSquared(0.0, besselHelper));
        for (size_t n = 1; n <= maxSamples; ++n) {
            double r_px = static_cast<double>(n) / static_cast<double>(samplesPerPixel);
            double k0NAr = k0 * cfg.NA * r_px * cfg.pixelSizeLateral_nm;
            double val = quadrature.evaluateSquared(k0NAr, besselHelper);
            h.push_back(val);
            if (forcedCutoff == 0 && val < threshold * h[0]) break;
        }

        std::lock_guard<std::mutex> lock(cacheMutex);
        cachedRadialProfiles[cacheKey] = h;
    }

    return h;
}

GibsonLanniIntegrand::GibsonLanniIntegrand(const GibsonLanniPSFConfig& config, double r, const BesselHelper& besselHelper)
    : config(config), r(r), besselHelper(besselHelper) {
        k0 = 2.0 * M_PI / config.wavelength_nm;
        k0NAr = k0 * config.NA * r;
    }

std::array<double, 2> GibsonLanniIntegrand::operator()(double rho) const {
    std::array<double, 2> I = {0.0, 0.0};

    double BesselValue = besselHelper.get(k0NAr * rho);

    double naRho = config.NA * rho;
    double naRhoOverNs = naRho / config.ns;
    double naRhoOverNg = naRho / config.ng;
    double naRhoOverNg0 = naRho / config.ng0;
    double naRhoOverNi = naRho / config.ni;
    double naRhoOverNi0 = naRho / config.ni0;

    double OPD1 = config.ns * config.particleAxialPosition_nm *
        std::sqrt(std::max(0.0, 1.0 - naRhoOverNs * naRhoOverNs));

    double OPD2 = config.ng * config.tg_nm *
            std::sqrt(std::max(0.0, 1.0 - naRhoOverNg * naRhoOverNg))
        - config.ng0 * config.tg0_nm *
            std::sqrt(std::max(0.0, 1.0 - naRhoOverNg0 * naRhoOverNg0));

    double OPD3 = config.ni * config.ti_nm *
            std::sqrt(std::max(0.0, 1.0 - naRhoOverNi * naRhoOverNi))
        - config.ni0 * config.ti0_nm *
            std::sqrt(std::max(0.0, 1.0 - naRhoOverNi0 * naRhoOverNi0));

    double OPD = OPD1 + OPD2 + OPD3;

    double W = k0 * OPD;

    I[0] = BesselValue * std::cos(W) * rho;
    I[1] = BesselValue * std::sin(W) * rho;

    return I;
}

size_t GibsonLanniQuadrature::nodeCount(double maxArgument, int accuracy) {
    // The integrand oscillates like J0(maxArgument * rho); resolve its
    // oscillations with ~2*accuracy nodes per half period (accuracy 1..10).
    int k = accuracy < 1 ? 1 : (accuracy > 10 ? 10 : accuracy);
    double oscillations = maxArgument / M_PI;
    size_t n = static_cast<size_t>(std::max(256.0, 2.0 * k * oscillations));
    n = std::min<size_t>(n, 1u << 16);
    // Composite Simpson needs an even node count.
    return n + (n % 2);
}

void GibsonLanniQuadrature::init(const GibsonLanniPSFConfig& config, double upperBound, size_t n) {
    if (n < 2) n = 2;
    const double del = upperBound / static_cast<double>(n);
    const double k0 = 2.0 * M_PI / config.wavelength_nm;

    rho.resize(n);
    cosW.resize(n);
    sinW.resize(n);

    for (size_t i = 0; i < n; ++i) {
        const double r = (i + 1) * del;
        rho[i] = r;

        const double naRho = config.NA * r;
        const double naRhoOverNs = naRho / config.ns;
        const double naRhoOverNg = naRho / config.ng;
        const double naRhoOverNg0 = naRho / config.ng0;
        const double naRhoOverNi = naRho / config.ni;
        const double naRhoOverNi0 = naRho / config.ni0;

        const double OPD =
            config.ns * config.particleAxialPosition_nm *
                std::sqrt(std::max(0.0, 1.0 - naRhoOverNs * naRhoOverNs))
            + config.ng * config.tg_nm *
                std::sqrt(std::max(0.0, 1.0 - naRhoOverNg * naRhoOverNg))
            - config.ng0 * config.tg0_nm *
                std::sqrt(std::max(0.0, 1.0 - naRhoOverNg0 * naRhoOverNg0))
            + config.ni * config.ti_nm *
                std::sqrt(std::max(0.0, 1.0 - naRhoOverNi * naRhoOverNi))
            - config.ni0 * config.ti0_nm *
                std::sqrt(std::max(0.0, 1.0 - naRhoOverNi0 * naRhoOverNi0));

        const double W = k0 * OPD;

        // Composite Simpson weights (4 odd, 2 even); rho factor folded in.
        const double weight = ((i + 1) % 2 == 1 ? 4.0 : 2.0) * del / 3.0;
        cosW[i] = std::cos(W) * r * weight;
        sinW[i] = std::sin(W) * r * weight;
    }
}

double GibsonLanniQuadrature::evaluateSquared(double k0NAr, const BesselHelper& besselHelper) const {
    double realSum = 0.0;
    double imagSum = 0.0;
    const size_t n = rho.size();
    for (size_t i = 0; i < n; ++i) {
        const double besselValue = besselHelper.get(k0NAr * rho[i]);
        realSum += besselValue * cosW[i];
        imagSum += besselValue * sinW[i];
    }
    return realSum * realSum + imagSum * imagSum;
}
