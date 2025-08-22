#pragma once

#include <memory>
#include "PSFConfig.h"
#include "BasePSFGenerator.h"
#include "SimpsonIntegrator.h"

class GibsonLanniPSFConfig;

class GibsonLanniPSFGenerator : public BasePSFGenerator {
public:
	GibsonLanniPSFGenerator(std::unique_ptr<NumericalIntegrator> integrator = std::make_unique<SimpsonIntegrator>());
    PSF generatePSF() const override;

    void setConfig(std::unique_ptr<PSFConfig> config) override;
    bool hasConfig() override;
	void setIntegrator(std::unique_ptr<NumericalIntegrator> integrator);
	cv::Mat SinglePlanePSF(const GibsonLanniPSFConfig& config) const; // gibson lanni equation for one z-slice

private:
	std::unique_ptr<NumericalIntegrator> numericalIntegrator;
    std::unique_ptr<GibsonLanniPSFConfig> config;

};


class GibsonLanniIntegrand {
public:
    GibsonLanniIntegrand(const GibsonLanniPSFConfig& config, double r);
	std::array<double, 2> operator()(double rho) const;

private:
	const GibsonLanniPSFConfig& config;
	const double r;
};


class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig() {
	}


	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& p)
	: PSFConfig(p){
		this->ng = p.ng;
		this->ng0 = p.ng0;
		this->ni = p.ni;
		this->ni0 = p.ni0;
		this->ns = p.ns;
		this->particleAxialPosition_nm = p.particleAxialPosition_nm;
		this->tg_nm = p.tg_nm;
		this->tg0_nm = p.tg0_nm;
		this->ti_nm = p.ti_nm;
		this->ti0_nm = p.ti0_nm;
		this->lambda_nm = p.lambda_nm;
        this->accuracy = p.accuracy;
        this->OVER_SAMPLING = p.OVER_SAMPLING;
	}
    bool loadFromJSON(const json& jsonData) override;
    void printValues() override;
	std::string getName() override;

	std::string psfModelName = "GibsonLanni";
	double OVER_SAMPLING;
	double lambda_nm;
	int accuracy;
	/**
	 * Working distance of the objective (design value). This is also the width
	 * of the immersion layer.
	 */
	double	ti0_nm;

	/**
	 * Working distance of the objective (experimental value). influenced by the
	 * stage displacement.
	 */
	double	ti_nm;

	/** Immersion medium refractive index (design value). */
	double	ni0;

	/** Immersion medium refractive index (experimental value). */
	double	ni;

	/** Coverslip thickness (design value). */
	double	tg0_nm;

	/** Coverslip thickness (experimental value). */
	double	tg_nm;

	/** Coverslip refractive index (design value). */
	double	ng0	= 1.5;

	/** Coverslip refractive index (experimental value). */
	double	ng	= 1.5;

	/** Sample refractive index. */
	double	ns;

	/** Axial position of the particle. */
	double	particleAxialPosition_nm;


};
