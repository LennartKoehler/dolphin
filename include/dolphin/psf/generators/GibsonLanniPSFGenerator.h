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

#include <memory>
#include <map>
#include <mutex>
#include <vector>
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/generators/BasePSFGenerator.h"
#include "dolphin/psf/generators/SimpsonIntegrator.h"

class GibsonLanniPSFConfig;

class GibsonLanniPSFGenerator : public BasePSFGenerator {
public:
	GibsonLanniPSFGenerator(std::unique_ptr<NumericalIntegrator> integrator = std::make_unique<SimpsonIntegrator>());
    PSF generatePSF() const override;

    void setConfig(const std::shared_ptr<const PSFConfig> config) override;
    bool hasConfig() override;
	void setIntegrator(std::unique_ptr<NumericalIntegrator> integrator);

	struct SliceData {
		std::vector<float> data;
		size_t lateralCutoff;
	};
	SliceData SinglePlanePSFAsVector(const GibsonLanniPSFConfig& config, size_t forcedCutoff = 0) const;

private:
	void initBesselHelper(size_t sizeX, size_t sizeY) const;
	PSF generateFixedSizePSF() const;
	PSF generateAutoSizePSF() const;
	std::unique_ptr<NumericalIntegrator> numericalIntegrator;
    std::shared_ptr<GibsonLanniPSFConfig> config;

	mutable std::map<double, std::vector<double>> cachedRadialProfiles;
	mutable std::mutex cacheMutex;
};


class GibsonLanniIntegrand {
public:
    GibsonLanniIntegrand(const GibsonLanniPSFConfig& config, double r);
	std::array<double, 2> operator()(double rho) const;

private:
	const GibsonLanniPSFConfig& config;
	const double r;
	double k0;
	double k0NAr;
};

