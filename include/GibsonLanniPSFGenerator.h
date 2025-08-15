#pragma once

#include <memory>
#include "PSFConfig.h"
#include "BasePSFGenerator.h"

class GibsonLanniPSFConfig;

class GibsonLanniPSFGenerator : public BasePSFGenerator {
public:

    PSF generatePSF() const override;
    void setConfig(std::unique_ptr<PSFConfig> config) override;
    bool hasConfig() override;

private:
    std::unique_ptr<GibsonLanniPSFConfig> config;
    int accuracy;
    int z;
};


class GibsonLanniPSFConfig : public PSFConfig{
public:
	GibsonLanniPSFConfig() {
	}


	GibsonLanniPSFConfig(const GibsonLanniPSFConfig& p) {
		this->ng = p.ng;
		this->ng0 = p.ng0;
		this->ni = p.ni;
		this->ni0 = p.ni0;
		this->ns = p.ns;
		this->particleAxialPosition = p.particleAxialPosition;
		this->tg = p.tg;
		this->tg0 = p.tg0;
		this->ti = p.ti;
		this->ti0 = p.ti0;
	}
    bool loadFromJSON(const json& jsonData) override;
    void printValues() override;
	std::string getName() override;

	std::string psfModelName = "GibsonLanni";
	/**
	 * Working distance of the objective (design value). This is also the width
	 * of the immersion layer.
	 */
	double	ti0;

	/**
	 * Working distance of the objective (experimental value). influenced by the
	 * stage displacement.
	 */
	double	ti;

	/** Immersion medium refractive index (design value). */
	double	ni0;

	/** Immersion medium refractive index (experimental value). */
	double	ni;

	/** Coverslip thickness (design value). */
	double	tg0;

	/** Coverslip thickness (experimental value). */
	double	tg;

	/** Coverslip refractive index (design value). */
	double	ng0	= 1.5;

	/** Coverslip refractive index (experimental value). */
	double	ng	= 1.5;

	/** Sample refractive index. */
	double	ns;

	/** Axial position of the particle. */
	double	particleAxialPosition;


};
