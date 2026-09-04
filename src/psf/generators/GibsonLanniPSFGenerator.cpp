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
#include <future>
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
}

PSF GibsonLanniPSFGenerator::generatePSF() const {

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

    long zCenter = static_cast<long>(sizeZ / 2);
    size_t offsetX = sizeX / 2 - lateralHalf;
    size_t offsetY = sizeY / 2 - lateralHalf;

    for (size_t z = 0; z < sizeZ; z++) {
        long offset = static_cast<long>(z) - zCenter;
        GibsonLanniPSFConfig cfg = *config;
        cfg.ti_nm = cfg.ti0_nm + pixelSizeAxial * static_cast<double>(offset);

        auto slice = SinglePlanePSFAsVector(cfg, lateralHalf);
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

PSF GibsonLanniPSFGenerator::generateAutoSizePSF() const {
    size_t effX = config->sizeX > 0 ? config->sizeX : 256;
    size_t effY = config->sizeY > 0 ? config->sizeY : 256;
    size_t effZ = config->sizeZ > 0 ? config->sizeZ : 256;
    double threshold = static_cast<double>(config->cutoffThreshold);
    double pixelSizeAxial = static_cast<double>(config->pixelSizeAxial_nm);

    auto makeCfg = [&](long offset) {
        GibsonLanniPSFConfig cfg = *config;
        cfg.sizeX = effX;
        cfg.sizeY = effY;
        cfg.ti_nm = cfg.ti0_nm + pixelSizeAxial * static_cast<double>(offset);
        return cfg;
    };

    struct GeneratedSlice { long offset; SliceData slice; double centerVal; };
    std::vector<GeneratedSlice> slices;

    auto centerSlice = SinglePlanePSFAsVector(makeCfg(0));
    size_t cc = centerSlice.lateralCutoff;
    double peakVal = centerSlice.data[cc * (2 * cc + 1) + cc];
    long peakOffset = 0;
    size_t maxCutoff = centerSlice.lateralCutoff;
    slices.push_back({0, std::move(centerSlice), peakVal});

    progressTracker.setMax(effZ);

    long negOffset = 0;
    while (slices.back().centerVal >= threshold * peakVal) {
        if (negOffset - 1 < -static_cast<long>(effZ / 2)) break;
        negOffset--;
        auto slice = SinglePlanePSFAsVector(makeCfg(negOffset));
        size_t c = slice.lateralCutoff;
        double val = slice.data[c * (2 * c + 1) + c];
        if (val > peakVal) { peakVal = val; peakOffset = negOffset; }
        maxCutoff = std::max(maxCutoff, slice.lateralCutoff);
        slices.push_back({negOffset, std::move(slice), val});
    }

    long posOffset = 0;
    while (slices[0].centerVal >= threshold * peakVal) {
        if (posOffset + 1 > static_cast<long>(effZ / 2)) break;
        posOffset++;
        auto slice = SinglePlanePSFAsVector(makeCfg(posOffset));
        size_t c = slice.lateralCutoff;
        double val = slice.data[c * (2 * c + 1) + c];
        if (val > peakVal) { peakVal = val; peakOffset = posOffset; }
        maxCutoff = std::max(maxCutoff, slice.lateralCutoff);
        slices.insert(slices.begin(), {posOffset, std::move(slice), val});
    }

    long maxNeg = std::abs(peakOffset - negOffset);
    long maxPos = posOffset - peakOffset;
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
        long targetZ = s.offset - peakOffset + zHalfExtent;
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

    BesselHelper& besselHelper = BesselHelper::instance();
    double nx = static_cast<double>(sizeX);
    double ny = static_cast<double>(sizeY);
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    double k0 = 2.0 * M_PI / config->lambda_nm;
    size_t maxRadius = static_cast<size_t>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;

    double max_k0NAr = k0 * config->NA * maxRadius * config->pixelSizeLateral_nm;
    double maxRho = std::min(float(1), config->ns / config->NA);

    double maxValue = max_k0NAr * maxRho;
    double dx = 0.1;

    besselHelper.init(0, maxValue, dx);
}

GibsonLanniPSFGenerator::SliceData GibsonLanniPSFGenerator::SinglePlanePSFAsVector(const GibsonLanniPSFConfig& config, size_t forcedCutoff) const {
    int OVER_SAMPLING = config.OVER_SAMPLING;
    double NA = config.NA;
    double pixelSizeLateral_nm = config.pixelSizeLateral_nm;
    double threshold = static_cast<double>(config.cutoffThreshold);

    size_t maxLateral = forcedCutoff > 0 ? forcedCutoff : (std::min(config.sizeX, config.sizeY) / 2);
    size_t maxSamples = maxLateral * OVER_SAMPLING;

    std::vector<double> cachedProfile;
    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        auto it = cachedRadialProfiles.find(config.ti_nm);
        if (it != cachedRadialProfiles.end())
            cachedProfile = it->second;
    }

    double a = 0.0;
    double b = std::min(1.0, config.ns / NA);
    double integrationTolerance = 1E-1;
    int integrationAccuracy = config.accuracy;

    std::vector<double> h;
    if (!cachedProfile.empty()) {
        h = cachedProfile;
    } else {
        GibsonLanniIntegrand integrand0(config, 0.0);
        h.push_back(numericalIntegrator->integrateComplex(integrand0, a, b, integrationTolerance, integrationAccuracy));
    }

    if (forcedCutoff > 0) {
        size_t n = h.size();
        while (n <= maxSamples) {
            double r_px = static_cast<double>(n) / static_cast<double>(OVER_SAMPLING);
            GibsonLanniIntegrand integrand(config, r_px * pixelSizeLateral_nm);
            double val = numericalIntegrator->integrateComplex(integrand, a, b, integrationTolerance, integrationAccuracy);
            h.push_back(val);
            n++;
        }
    } else if (!(h.size() > 1 && h.back() < threshold * h[0])) {
        size_t n = h.size();
        while (n < maxSamples) {
            double r_px = static_cast<double>(n) / static_cast<double>(OVER_SAMPLING);
            GibsonLanniIntegrand integrand(config, r_px * pixelSizeLateral_nm);
            double val = numericalIntegrator->integrateComplex(integrand, a, b, integrationTolerance, integrationAccuracy);
            h.push_back(val);
            n++;
            if (val < threshold * h[0]) break;
        }
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex);
        cachedRadialProfiles[config.ti_nm] = h;
    }

    size_t cutoff = forcedCutoff > 0 ? forcedCutoff
        : static_cast<size_t>(std::ceil(static_cast<double>(h.size() - 1) / OVER_SAMPLING));
    size_t gridSize = 2 * cutoff + 1;
    std::vector<float> sliceData(gridSize * gridSize, 0.0f);

    for (size_t y = 0; y < gridSize; y++) {
        for (size_t x = 0; x < gridSize; x++) {
            double dx = static_cast<double>(x) - static_cast<double>(cutoff);
            double dy = static_cast<double>(y) - static_cast<double>(cutoff);
            double rPixel = std::sqrt(dx * dx + dy * dy);
            size_t index = static_cast<size_t>(std::floor(rPixel * OVER_SAMPLING));

            double value = 0.0;
            if (index + 1 < h.size()) {
                double r0 = static_cast<double>(index) / static_cast<double>(OVER_SAMPLING);
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r0) * OVER_SAMPLING;
            } else if (index < h.size()) {
                value = h[index];
            }

            sliceData[y * gridSize + x] = static_cast<float>(value);
        }
    }
    progressTracker.add(1);

    return {std::move(sliceData), cutoff};
}


GibsonLanniIntegrand::GibsonLanniIntegrand(const GibsonLanniPSFConfig& config, double r)
    : config(config), r(r) {
        k0 = 2.0 * M_PI / config.lambda_nm;
        k0NAr = k0 * config.NA * r;
    }

std::array<double, 2> GibsonLanniIntegrand::operator()(double rho) const {
    std::array<double, 2> I = {0.0, 0.0};

    const BesselHelper& besselHelper = BesselHelper::instance();
    double BesselValue = besselHelper.get(k0NAr * rho);

    if ((config.NA * rho / config.ns) > 1.0)
        spdlog::info("Warning: NA*rho/ns > 1, (ns,NA,rho)=({}, {}, {})\n", config.ns, config.NA, rho);

    double naRho = config.NA * rho;

    double OPD1 = config.ns * config.particleAxialPosition_nm *
        std::sqrt(std::max(0.0, 1.0 - std::pow(naRho / config.ns, 2)));

    double OPD2 = config.ng * config.tg_nm *
            std::sqrt(std::max(0.0, 1.0 - std::pow(naRho / config.ng, 2)))
        - config.ng0 * config.tg0_nm *
            std::sqrt(std::max(0.0, 1.0 - std::pow(naRho / config.ng0, 2)));

    double OPD3 = config.ni * config.ti_nm *
            std::sqrt(std::max(0.0, 1.0 - std::pow(naRho / config.ni, 2)))
        - config.ni0 * config.ti0_nm *
            std::sqrt(std::max(0.0, 1.0 - std::pow(naRho / config.ni0, 2)));

    double OPD = OPD1 + OPD2 + OPD3;

    double W = k0 * OPD;

    I[0] = BesselValue * std::cos(W) * rho;
    I[1] = BesselValue * std::sin(W) * rho;

    return I;
}
