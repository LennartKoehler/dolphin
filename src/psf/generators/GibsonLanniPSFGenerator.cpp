/**
 * PSFGenerator
 *
 * Authors: Daniel Sage and Hagai Kirshner
 * Organization: Biomedical Imaging Group (BIG), Ecole Polytechnique Federale de Lausanne
 * Address: EPFL-STI-IMT-LIB, 1015 Lausanne, Switzerland
 * Information: http://bigwww.epfl.ch/algorithms/psfgenerator/
 *
 * References:
 * [1] H. Kirshner, F. Aguet, D. Sage, M. Unser
 * 3-D PSF Fitting for Fluorescence Microscopy: Implementation and Localization Application
 * Journal of Microscopy, vol. 249, no. 1, pp. 13-25, January 2013.
 * Available at: http://bigwww.epfl.ch/publications/kirshner1301.html
 *
 * [2] A. Griffa, N. Garin, D. Sage
 * Comparison of Deconvolution Software in 3D Microscopy: A User Point of View
 * G.I.T. Imaging & Microscopy, vol. 12, no. 1, pp. 43-45, March 2010.
 * Available at: http://bigwww.epfl.ch/publications/griffa1001.html
 *
 * Conditions of use:
 * Conditions of use: You are free to use this software for research or
 * educational purposes. In addition, we expect you to include adequate
 * citations and acknowledgments whenever you present or publish results that
 * are based on it.
 */

/**
 * Copyright 2010-2017 Biomedical Imaging Group at the EPFL.
 *
 * This file is part of PSFGenerator.
 *
 * PSFGenerator is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * PSFGenerator is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * PSFGenerator. If not, see <http://www.gnu.org/licenses/>.
 */


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

#include "dolphin/psf/generators/GibsonLanniPSFGenerator.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/psf/generators/BesselHelper.h"
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <spdlog/spdlog.h>

GibsonLanniPSFGenerator::GibsonLanniPSFGenerator(std::unique_ptr<NumericalIntegrator> integrator)
    : numericalIntegrator(std::move(integrator)){}

bool GibsonLanniPSFGenerator::hasConfig(){
    return config != nullptr;
}

void GibsonLanniPSFGenerator::setIntegrator(std::unique_ptr<NumericalIntegrator> integrator){
    this->numericalIntegrator = std::move(integrator);
}

void GibsonLanniPSFGenerator::setConfig(const std::shared_ptr<const PSFConfig> config){
    auto* ucfg = dynamic_cast<const GibsonLanniPSFConfig*>(config.get());
    if (!ucfg) throw std::runtime_error("Wrong config type");
    this->config = std::make_unique<GibsonLanniPSFConfig>(*ucfg);
    // Profiles are cached keyed on ti_nm only, but the integrand also
    // depends on every other optical parameter. Invalidate on config change.
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cachedRadialProfiles.clear();
    }
}

PSF GibsonLanniPSFGenerator::generatePSF() const {

    if (!threadPool)
        throw std::runtime_error("GibsonLanniPSFGenerator threadpool not configured");

    // cutoffThreshold <= 0 means "not set" (e.g. standalone PSF generation
    // without a padding strategy); fall back to the MEDIUM trim margin.
    // this->config is a private copy, so this never leaks to the caller.
    if (config->cutoffThreshold <= 0.0f)
        config->cutoffThreshold = 1e-6f;

    // The old per-evaluation "NA*rho/ns > 1" log fired on every integrand
    // evaluation (thousands of lines per slice). Emit it once per generation
    // instead: the violation is monotonic in rho, so if it holds at the upper
    // bound b it holds somewhere in the domain.
    const double rhoMax = integrationUpperBound(*config);
    if (config->NA * rhoMax / config->ns > 1.0)
        spdlog::warn("GibsonLanni: NA*rho/ns > 1 within the integration domain (ns={}, NA={}, rho_max={})",
                     config->ns, config->NA, rhoMax);

    size_t effX = config->sizeX > 0 ? config->sizeX : 256;
    size_t effY = config->sizeY > 0 ? config->sizeY : 256;

    initBesselHelper(effX, effY);

    if (config->autoSize || config->sizeX == 0 || config->sizeY == 0 || config->sizeZ == 0) {
        return generateAutoSizePSF();
    }
    return generateFixedSizePSF();
}

PSF GibsonLanniPSFGenerator::generateFixedSizePSF() const {
    size_t sizeX = config->sizeX;
    size_t sizeY = config->sizeY;
    size_t sizeZ = config->sizeZ;
    size_t lateralHalf = (std::min(sizeX, sizeY) - 1) / 2;
    double pixelSizeAxial = static_cast<double>(config->pixelSizeAxial_nm);

    progressTracker.setMax(sizeZ);

    ImageType::Pointer itkImage = ImageType::New();
    ImageType::SizeType imgSize;
    imgSize[0] = sizeX;
    imgSize[1] = sizeY;
    imgSize[2] = sizeZ;
    ImageType::IndexType start;
    start.Fill(0);
    ImageType::RegionType region;
    region.SetSize(imgSize);
    region.SetIndex(start);
    itkImage->SetRegions(region);
    itkImage->Allocate();
    itkImage->FillBuffer(0.0f);

    // Center on the middle voxel like BIG ((nz-1)/2), not the floor
    // division of sizeZ/2, which shifts the axial center by half a pixel
    // for even sizeZ.
    long zCenter = static_cast<long>((sizeZ - 1) / 2);
    size_t offsetX = sizeX / 2 - lateralHalf;
    size_t offsetY = sizeY / 2 - lateralHalf;

    auto makeSliceCfg = [&](size_t z) {
        GibsonLanniPSFConfig cfg = *config;
        long offset = static_cast<long>(z) - zCenter;
        cfg.ti_nm = cfg.ti0_nm + pixelSizeAxial * static_cast<double>(offset);
        return cfg;
    };

    std::vector<std::future<SliceData>> sliceFutures;
    sliceFutures.reserve(sizeZ);
    for (size_t z = 0; z < sizeZ; z++) {
        sliceFutures.emplace_back(threadPool->enqueue([this, cfg = makeSliceCfg(z), lateralHalf]() {
            return singlePlanePSF(cfg, lateralHalf);
        }));
    }

    for (size_t z = 0; z < sizeZ; z++) {
        SliceData slice = sliceFutures[z].get();
        size_t sliceW = 2 * slice.lateralCutoff + 1;

        for (size_t y = 0; y < sliceW; y++) {
            for (size_t x = 0; x < sliceW; x++) {
                itk::Index<3> idx;
                idx[0] = static_cast<itk::IndexValueType>(x + offsetX);
                idx[1] = static_cast<itk::IndexValueType>(y + offsetY);
                idx[2] = static_cast<itk::IndexValueType>(z);
                itkImage->SetPixel(idx, slice.data[y * sliceW + x]);
            }
        }
    }

    return PSF(std::move(itkImage), config->ID);
}



PSF GibsonLanniPSFGenerator::generateAutoSizePSF() const
{
    const size_t effX = config->sizeX > 0 ? config->sizeX : 256;
    const size_t effY = config->sizeY > 0 ? config->sizeY : 256;
    const size_t effZ = config->sizeZ > 0 ? config->sizeZ : 256;

    const double threshold = static_cast<double>(config->cutoffThreshold);

    const double pixelSizeAxial =
        static_cast<double>(config->pixelSizeAxial_nm);

    auto makeCfg = [&](long offset) {
        GibsonLanniPSFConfig cfg = *config;
        cfg.sizeX = effX;
        cfg.sizeY = effY;
        cfg.ti_nm =
            cfg.ti0_nm +
            pixelSizeAxial * static_cast<double>(offset);
        return cfg;
    };

    const long maxOffset = static_cast<long>(effZ / 2);

    struct GeneratedSlice {
        long offset;
        SliceData slice;
    };

    /*
     * Pass 1: axial extent. Only the on-axis value of every candidate slice
     * is computed (one integration per slice, all in parallel). The axial
     * support is the contiguous region around the focus whose on-axis value
     * stays above threshold * global peak.
     */
    const long scanRange = 2 * maxOffset + 1;
    std::vector<std::future<double>> centerFutures;
    centerFutures.reserve(scanRange);
    for (long offset = -maxOffset; offset <= maxOffset; ++offset) {
        centerFutures.emplace_back(threadPool->enqueue([this, cfg = makeCfg(offset)]() {
            return centerValue(cfg);
        }));
    }

    double peakVal = 0.0;
    std::vector<double> centerVals(scanRange);
    for (long i = 0; i < scanRange; ++i) {
        centerVals[i] = centerFutures[i].get();
        peakVal = std::max(peakVal, centerVals[i]);
    }
    if (peakVal <= 0.0) {
        throw std::runtime_error("GibsonLanni: PSF has zero intensity at every scanned slice");
    }

    const double axialCutoff = threshold * peakVal;
    const auto valueAt = [&](long offset) {
        return centerVals[offset + maxOffset];
    };

    long extentNeg = 0;
    while (extentNeg < maxOffset &&
           valueAt(-(extentNeg + 1)) >= axialCutoff) {
        ++extentNeg;
    }
    long extentPos = 0;
    while (extentPos < maxOffset &&
           valueAt(extentPos + 1) >= axialCutoff) {
        ++extentPos;
    }

    const bool axialCapped =
        extentNeg == maxOffset || extentPos == maxOffset;
    if (axialCapped) {
        spdlog::warn("GibsonLanni: axial cutoff {} not reached, PSF truncated by the size cap ({} slices)",
                     threshold, 2 * maxOffset + 1);
    }

    /*
     * Pass 2: full radial profiles, only for the kept slices, all in
     * parallel. The per-slice lateral cutoff logic in singlePlanePSF is
     * unchanged.
     */
    progressTracker.setMax(static_cast<float>(extentNeg + extentPos + 1));

    std::vector<std::future<SliceData>> sliceFutures;
    sliceFutures.reserve(extentNeg + extentPos + 1);
    for (long offset = -extentNeg; offset <= extentPos; ++offset) {
        sliceFutures.emplace_back(threadPool->enqueue([this, cfg = makeCfg(offset)]() {
            return singlePlanePSF(cfg);
        }));
    }

    std::vector<GeneratedSlice> slices;
    slices.reserve(sliceFutures.size());
    size_t maxCutoff = 0;
    const size_t lateralCap = std::min(effX, effY) / 2;
    bool lateralCapped = false;
    for (size_t i = 0; i < sliceFutures.size(); ++i) {
        SliceData slice = sliceFutures[i].get();
        const long offset = static_cast<long>(i) - extentNeg;
        maxCutoff = std::max(maxCutoff, slice.lateralCutoff);
        lateralCapped = lateralCapped || slice.lateralCutoff >= lateralCap;
        slices.push_back({offset, std::move(slice)});
    }
    if (lateralCapped) {
        spdlog::warn("GibsonLanni: lateral cutoff {} not reached on at least one slice, PSF truncated by the size cap ({})",
                     threshold, 2 * maxCutoff + 1);
    }

    // Final physical order: -N, ..., -1, 0, 1, ..., N (already ordered)
    const long maxNeg = extentNeg;
    const long maxPos = extentPos;
    long zHalfExtent = std::max(maxNeg, maxPos);
    size_t psfD = static_cast<size_t>(2 * zHalfExtent + 1);
    size_t psfW = 2 * maxCutoff + 1;

    ImageType::Pointer itkImage = ImageType::New();
    ImageType::SizeType size;
    size[0] = psfW;
    size[1] = psfW;
    size[2] = psfD;
    ImageType::IndexType start;
    start.Fill(0);
    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    itkImage->SetRegions(region);
    itkImage->Allocate();
    itkImage->FillBuffer(0.0f);

    for (const auto& s : slices) {
        long targetZ = s.offset + zHalfExtent;
        if (targetZ < 0 || targetZ >= static_cast<long>(psfD))
            continue;

        size_t sliceW = 2 * s.slice.lateralCutoff + 1;
        long pad = static_cast<long>(maxCutoff) - static_cast<long>(s.slice.lateralCutoff);

        for (size_t y = 0; y < sliceW; y++) {
            for (size_t x = 0; x < sliceW; x++) {
                itk::Index<3> idx;
                idx[0] = static_cast<itk::IndexValueType>(x + pad);
                idx[1] = static_cast<itk::IndexValueType>(y + pad);
                idx[2] = static_cast<itk::IndexValueType>(targetZ);
                itkImage->SetPixel(idx, s.slice.data[y * sliceW + x]);
            }
        }
    }

    return PSF(std::move(itkImage), config->ID);
}

void GibsonLanniPSFGenerator::initBesselHelper(size_t sizeX, size_t sizeY) const {
    assert (config != nullptr && "Config not initialized");

    double nx = static_cast<double>(sizeX);
    double ny = static_cast<double>(sizeY);
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    double k0 = 2.0 * M_PI / config->wavelength_nm;
    size_t maxRadius = static_cast<size_t>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;

    double max_k0NAr = k0 * config->NA * maxRadius * config->pixelSizeLateral_nm;
    double maxRho = std::min(float(1), config->ns / config->NA);

    // maxRadius covers the image corner, so any sizeX/sizeY asymmetry is included
    double maxValue = max_k0NAr * maxRho;
    double dx = 0.1;

    besselHelper.init(maxValue, dx);
}

double GibsonLanniPSFGenerator::integrationUpperBound(const GibsonLanniPSFConfig& cfg) {
    const double NA = cfg.NA;
    // Each sqrt(1 - (NA*rho/n)^2) term requires rho <= n/NA, so the upper
    // bound is the tightest such limit across every medium in the path (and
    // 1.0 at the pupil edge). For standard configs the sample (ns) is the
    // smallest RI, so this reduces to the original min(1.0, ns/NA); only
    // exotic configs tighten the bound.
    return std::min({1.0, cfg.ns / NA, cfg.ni / NA, cfg.ni0 / NA,
                     cfg.ng / NA, cfg.ng0 / NA});
}

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

double GibsonLanniPSFGenerator::centerValue(const GibsonLanniPSFConfig& config) const {
    // On-axis value: only the r=0 integral is needed (J0(0)=1). Uses the
    // cached radial profile if this slice was already computed.
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cachedRadialProfiles.find(config.ti_nm);
        if (it != cachedRadialProfiles.end() && !it->second.empty())
            return it->second[0];
    }

    double a = 0.0;
    double b = integrationUpperBound(config);
    GibsonLanniIntegrand integrand(config, 0.0, besselHelper);
    return numericalIntegrator->integrateComplex(
        integrand, a, b, 1E-6, config.integration_accuracy);
}

GibsonLanniPSFGenerator::SliceData GibsonLanniPSFGenerator::singlePlanePSF(const GibsonLanniPSFConfig& config, size_t forcedCutoff) const {
    int oversampling_factor = config.oversampling_factor;
    double NA = config.NA;
    double pixelSizeLateral_nm = config.pixelSizeLateral_nm;
    double threshold = static_cast<double>(config.cutoffThreshold);

    size_t maxLateral = forcedCutoff > 0 ? forcedCutoff : (std::min(config.sizeX, config.sizeY) / 2);
    size_t maxSamples = maxLateral * oversampling_factor;

    std::vector<double> h;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cachedRadialProfiles.find(config.ti_nm);
        if (it != cachedRadialProfiles.end())
            h = it->second;
    }

    const double k0 = 2.0 * M_PI / config.wavelength_nm;

    if (!cachedProfileComplete(h, forcedCutoff, maxSamples, threshold)) {
        // Fixed-node quadrature: phase tabulated once, then one MAC per radius.
        const size_t maxRadiusSamples = forcedCutoff > 0
            ? maxSamples
            : maxSamples; // the per-radius break below stops early anyway

        GibsonLanniQuadrature quadrature;
        const double b = integrationUpperBound(config);
        const double maxR_nm = (static_cast<double>(maxRadiusSamples) / oversampling_factor) * pixelSizeLateral_nm;
        const double maxArg = k0 * NA * maxR_nm * b;
        quadrature.init(config, b, GibsonLanniQuadrature::nodeCount(maxArg, config.integration_accuracy));

        h.clear();
        h.push_back(quadrature.evaluateSquared(0.0, besselHelper));
        for (size_t n = 1; n <= maxRadiusSamples; ++n) {
            double r_px = static_cast<double>(n) / static_cast<double>(oversampling_factor);
            double k0NAr = k0 * NA * r_px * pixelSizeLateral_nm;
            double val = quadrature.evaluateSquared(k0NAr, besselHelper);
            h.push_back(val);
            if (forcedCutoff == 0 && val < threshold * h[0]) break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cachedRadialProfiles[config.ti_nm] = h;
    }

    size_t cutoff = forcedCutoff > 0 ? forcedCutoff
        : static_cast<size_t>(std::ceil(static_cast<double>(h.size() - 1) / oversampling_factor));
    size_t gridSize = 2 * cutoff + 1;
    std::vector<float> sliceData(gridSize * gridSize, 0.0f);

    for (size_t y = 0; y < gridSize; y++) {
        for (size_t x = 0; x < gridSize; x++) {
            double dx = static_cast<double>(x) - static_cast<double>(cutoff);
            double dy = static_cast<double>(y) - static_cast<double>(cutoff);
            double rPixel = std::sqrt(dx * dx + dy * dy);
            size_t index = static_cast<size_t>(std::floor(rPixel * oversampling_factor));

            double value = 0.0;
            if (index + 1 < h.size()) {
                double r0 = static_cast<double>(index) / static_cast<double>(oversampling_factor);
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r0) * oversampling_factor;
            } else if (index < h.size()) {
                value = h[index];
            }

            sliceData[y * gridSize + x] = static_cast<float>(value);
        }
    }
    progressTracker.add(1);

    return {std::move(sliceData), cutoff};
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
