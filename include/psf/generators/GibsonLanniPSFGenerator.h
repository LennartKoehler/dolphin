#pragma once

#include <memory>
#include "psf/configs/PSFConfig.h"
#include "BasePSFGenerator.h"
#include "SimpsonIntegrator.h"

class GibsonLanniPSFConfig;

class GibsonLanniPSFGenerator : public BasePSFGenerator {
public:
	GibsonLanniPSFGenerator(std::unique_ptr<NumericalIntegrator> integrator = std::make_unique<SimpsonIntegrator>());
    PSF generatePSF() const override;

    void setConfig(const std::shared_ptr<const PSFConfig> config) override;
    bool hasConfig() override;
	void setIntegrator(std::unique_ptr<NumericalIntegrator> integrator);
	cv::Mat SinglePlanePSF(const GibsonLanniPSFConfig& config) const; // gibson lanni equation for one z-slice

private:
	void initBesselHelper() const;
	std::unique_ptr<NumericalIntegrator> numericalIntegrator;
    std::shared_ptr<GibsonLanniPSFConfig> config;

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

