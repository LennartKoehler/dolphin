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

//TODO
CuboidShape GibsonLanniPSFGenerator::getPadding(PaddingStrategyType paddingType) const {
    switch(paddingType){
    case(PARENT):
        // TODO
        return CuboidShape{config->sizeX / 2, config->sizeY / 2, config->sizeZ / 2};
    case(FULL_PSF):
        return CuboidShape{config->sizeX, config->sizeY, config->sizeZ};
    default:
        return CuboidShape{SIZE_MAX, SIZE_MAX, SIZE_MAX};
    }
}

void GibsonLanniPSFGenerator::initBesselHelper() const {
    assert (config != nullptr && "Config not initialized");

    BesselHelper& besselHelper = BesselHelper::instance();
    double nx = config->sizeX;
    double ny = config->sizeY;
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    double k0 = 2.0 * M_PI / config->lambda_nm;
    size_t maxRadius = static_cast<size_t>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;

    double max_k0NAr = k0 * config->NA * maxRadius * config->pixelSizeLateral_nm;
    double maxRho = std::min(float(1), config->ns / config->NA);

    double maxValue = max_k0NAr * maxRho; // TODO IMPORTANT is maxvalue just sizeX or sizeY?
    double dx = 0.1;

    besselHelper.init(0, maxValue, dx);
}

LateralClip GibsonLanniPSFGenerator::clipSize() const {
    size_t nx = config->sizeX;
    size_t ny = config->sizeY;
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    // Characteristic lateral scale: Airy disk radius in pixels
    double airyRadius_px = 0.61 * config->lambda_nm / (config->NA * config->pixelSizeLateral_nm);

    // Generous cutoff to capture all significant energy including sidelobes
    constexpr double cutoffFactor = 10.0;
    double cutoffRadius = cutoffFactor * airyRadius_px;

    size_t xMin = static_cast<size_t>(std::max(0L, static_cast<long>(std::floor(x0 - cutoffRadius))));
    size_t xMax = std::min(nx - 1, static_cast<size_t>(std::max(0L, static_cast<long>(std::ceil(x0 + cutoffRadius)))));
    size_t yMin = static_cast<size_t>(std::max(0L, static_cast<long>(std::floor(y0 - cutoffRadius))));
    size_t yMax = std::min(ny - 1, static_cast<size_t>(std::max(0L, static_cast<long>(std::ceil(y0 + cutoffRadius)))));

    return {xMin, xMax, yMin, yMax};
}

PSF GibsonLanniPSFGenerator::generatePSF() const {
    initBesselHelper();

    // Create ITK 3D image
    ImageType::Pointer itkImage = ImageType::New();

    // Set the image dimensions
    ImageType::SizeType size;
    size[0] = config->sizeX;
    size[1] = config->sizeY;
    size[2] = config->sizeZ;

    ImageType::IndexType start;
    start.Fill(0);

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    itkImage->SetRegions(region);
    itkImage->Allocate();
    itkImage->FillBuffer(0.0f);

    // Compute the effective lateral region where the PSF is non-negligible
    LateralClip clip = clipSize();

    // Process each z-slice using threading
    std::vector<std::future<std::vector<float>>> tempSphereLayers;
    tempSphereLayers.reserve(config->sizeZ);

    progressTracker.setMax(config->sizeZ);

    for (size_t z = 0; z < config->sizeZ; z++){
        GibsonLanniPSFConfig configCopy = *(this->config);
        configCopy.ti_nm = configCopy.ti0_nm + configCopy.pixelSizeAxial_nm * (static_cast<double>(z) - (config->sizeZ - 1.0) / 2.0);
        tempSphereLayers.emplace_back(threadPool->enqueue([this, configCopy, clip](){
            return SinglePlanePSFAsVector(configCopy, clip);
        }));
    }

    size_t clippedWidth = clip.xMax - clip.xMin + 1;

    // Copy data from computed slices into the clipped sub-region of the ITK image
    for (size_t z = 0; z < config->sizeZ; z++) {
        std::vector<float> sliceData = tempSphereLayers[z].get();

        ImageType::IndexType sliceStart;
        sliceStart[0] = static_cast<itk::IndexValueType>(clip.xMin);
        sliceStart[1] = static_cast<itk::IndexValueType>(clip.yMin);
        sliceStart[2] = static_cast<itk::IndexValueType>(z);

        ImageType::SizeType sliceSize;
        sliceSize[0] = clippedWidth;
        sliceSize[1] = clip.yMax - clip.yMin + 1;
        sliceSize[2] = 1;

        ImageType::RegionType sliceRegion;
        sliceRegion.SetIndex(sliceStart);
        sliceRegion.SetSize(sliceSize);

        itk::ImageRegionIterator<ImageType> it(itkImage, sliceRegion);

        size_t dataIndex = 0;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++dataIndex) {
            it.Set(sliceData[dataIndex]);
        }
    }

    return PSF(std::move(itkImage), config->ID);
}


std::vector<float> GibsonLanniPSFGenerator::SinglePlanePSFAsVector(const GibsonLanniPSFConfig& config, const LateralClip& clip) const {
    size_t nx = config.sizeX;
    size_t ny = config.sizeY;
    int OVER_SAMPLING = config.OVER_SAMPLING;
    double NA = config.NA;
    double lambda_nm = config.lambda_nm;
    double pixelSizeLateral_nm = config.pixelSizeLateral_nm;
    double pixelSizeAxial_nm = config.pixelSizeAxial_nm;


    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;

    // Lateral particle position in units of [pixels]
    double xp = x0;
    double yp = y0;

    // Calculate maximum radius — limited to the clipped region
    double dxMax = std::max(std::abs(static_cast<double>(clip.xMax) - x0), std::abs(static_cast<double>(clip.xMin) - x0));
    double dyMax = std::max(std::abs(static_cast<double>(clip.yMax) - y0), std::abs(static_cast<double>(clip.yMin) - y0));
    size_t maxRadius = static_cast<size_t>(std::round(std::sqrt(dxMax * dxMax + dyMax * dyMax))) + 1;

    std::vector<double> r(maxRadius * OVER_SAMPLING);
    std::vector<double> h(r.size());

    //TODO set tolerance and K/accuracy for numerical integrator
    double a = 0.0;
    double b = std::min(1.0, config.ns / NA);
    int integrationAccuracy = config.accuracy;
        double integrationTolerance = 1E-1;

    for (size_t n = 0; n < r.size(); n++) { // get kirchhoffdiffraction for specific radius

        r[n] = static_cast<double>(n) / static_cast<double>(OVER_SAMPLING);
        GibsonLanniIntegrand integrand(config, r[n] * pixelSizeLateral_nm);
        h[n] = numericalIntegrator->integrateComplex(integrand, a, b, integrationTolerance, integrationAccuracy);
    }

    // Linear interpolation of the pixel values — only within the clipped region
    size_t clippedWidth = clip.xMax - clip.xMin + 1;
    size_t clippedHeight = clip.yMax - clip.yMin + 1;
    std::vector<float> sliceData(clippedWidth * clippedHeight, 0.0f);
    double rPixel, value;
    size_t index;

    for (size_t x = clip.xMin; x <= clip.xMax; x++) {
        for (size_t y = clip.yMin; y <= clip.yMax; y++) {
            rPixel = std::sqrt((static_cast<double>(x) - xp) * (static_cast<double>(x) - xp) + (static_cast<double>(y) - yp) * (static_cast<double>(y) - yp));
            index = static_cast<size_t>(std::floor(rPixel * OVER_SAMPLING));

            if (index + 1 < h.size()) {
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r[index]) * OVER_SAMPLING;
            } else if (index < h.size()) {
                value = h[index];
            } else {
                value = 0.0;
            }
            sliceData[(y - clip.yMin) * clippedWidth + (x - clip.xMin)] = static_cast<float>(value);
        }
    }
    progressTracker.add(1);

    return sliceData;
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

    double OPD1 = config.ns * config.particleAxialPosition_nm * std::sqrt(1 - std::pow(config.NA * rho / config.ns, 2));
    double OPD3 = config.ni * (config.ti_nm - config.ti0_nm) * std::sqrt(1 - std::pow(config.NA * rho / config.ni, 2));
    double OPD = OPD1 + OPD3;

    double W = k0 * OPD;

    I[0] = BesselValue * std::cos(W) * rho;
    I[1] = BesselValue * std::sin(W) * rho;

    return I;
}
