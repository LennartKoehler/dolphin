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

#include "psf/generators/GibsonLanniPSFGenerator.h"
#include "psf/configs/GibsonLanniPSFConfig.h"
#include "ThreadPool.h"
#include "psf/generators/BesselHelper.h"

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


void GibsonLanniPSFGenerator::initBesselHelper() const {
    assert (config != nullptr && "Config not initialized");
    
    BesselHelper& besselHelper = BesselHelper::instance();
    double nx = config->sizeX;
    double ny = config->sizeY;
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;
    
    double k0 = 2.0 * M_PI / config->lambda_nm;
    int maxRadius = static_cast<int>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;

    double max_k0NAr = k0 * config->NA * maxRadius * config->pixelSizeLateral_nm;
    double maxRho = std::min(float(1), config->ns / config->NA);

    double maxValue = max_k0NAr * maxRho;
    double dx = 0.01;
    besselHelper.init(0, maxValue, dx);
}

PSF GibsonLanniPSFGenerator::generatePSF() const {
    std::vector<cv::Mat> sphereLayers;
    std::vector<std::future<cv::Mat>> tempSphereLayers;
    sphereLayers.reserve(config->sizeZ);

    initBesselHelper();


    for (int z = 0; z < config->sizeZ; z++){
        GibsonLanniPSFConfig config = *(this->config);
        config.ti_nm = config.ti0_nm + config.pixelSizeAxial_nm * (z - (config.sizeZ - 1.0) / 2.0);
        tempSphereLayers.emplace_back(threadPool->enqueue([this, config](){
            return SinglePlanePSF(config);
        })); 
    }
    for (auto& future : tempSphereLayers){
        sphereLayers.push_back(future.get());
    }
    Image3D psfImage;
    psfImage.slices = sphereLayers;
    PSF psf;
    psf.image = psfImage;
    return psf;
}


cv::Mat GibsonLanniPSFGenerator::SinglePlanePSF(const GibsonLanniPSFConfig& config) const {    
    int nx = config.sizeX;
    int ny = config.sizeY;
    int OVER_SAMPLING = config.OVER_SAMPLING;
    double NA = config.NA;
    double lambda_nm = config.lambda_nm;
    double pixelSizeLateral_nm = config.pixelSizeLateral_nm;
    double pixelSizeAxial = config.pixelSizeAxial_nm;

    
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;
    
    // Lateral particle position in units of [pixels]
    double xp = x0; // 0.0/pixelSize;
    double yp = y0; // 0.0/pixelSize;
    
    // Calculate maximum radius
    int maxRadius = static_cast<int>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;
    
    std::vector<double> r(maxRadius * OVER_SAMPLING);
    std::vector<double> h(r.size());
    
    //TODO set tolerance and K/accuracy for numerical integrator
    //TODO what do i want to pass to the kirchhoffequation as parameteres, what is r, what is rho? do i want to pass r or rho
    double a = 0.0;
    double b = std::min(1.0, config.ns / NA);
    int integrationAccuracy = config.accuracy;
        double integrationTolerance = 1E-1;

    for (size_t n = 0; n < r.size(); n++) { // get kirchhoffdiffraction for specific radius

        r[n] = static_cast<double>(n) / static_cast<double>(OVER_SAMPLING);
        GibsonLanniIntegrand integrand(config, r[n] * pixelSizeLateral_nm);
        h[n] = numericalIntegrator->integrateComplex(integrand, a, b, integrationTolerance, integrationAccuracy);
    }
    
    // Linear interpolation of the pixel values
    cv::Mat slice(nx, ny, CV_32F);
    double rPixel, value;
    int index;
    
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            rPixel = std::sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp));
            index = static_cast<int>(std::floor(rPixel * OVER_SAMPLING));
            
            if (index + 1 < static_cast<int>(h.size())) {
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r[index]) * OVER_SAMPLING;
            } else {
                value = h[index];
            }
            slice.at<float>(x,y) = value;
        }
    }
    
    return slice;
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
        std::cout << "Warning: NA*rho/ns > 1, (ns,NA,rho)=(" 
                  << config.ns << ", " << config.NA << ", " << rho << ")\n";

    double OPD1 = config.ns * config.particleAxialPosition_nm * std::sqrt(1 - std::pow(config.NA * rho / config.ns, 2));
    double OPD3 = config.ni * (config.ti_nm - config.ti0_nm) * std::sqrt(1 - std::pow(config.NA * rho / config.ni, 2));
    double OPD = OPD1 + OPD3;

    double W = k0 * OPD;

    I[0] = BesselValue * std::cos(W) * rho;
    I[1] = BesselValue * std::sin(W) * rho;

    return I;
}
