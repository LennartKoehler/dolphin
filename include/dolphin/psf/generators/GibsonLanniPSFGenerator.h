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
#include "dolphin/psf/generators/BesselHelper.h"

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
	SliceData singlePlanePSF(const GibsonLanniPSFConfig& config, size_t forcedCutoff = 0) const;

private:
	void initBesselHelper(size_t sizeX, size_t sizeY) const;
	PSF generateFixedSizePSF() const;
	PSF generateAutoSizePSF() const;
	static double integrationUpperBound(const GibsonLanniPSFConfig& cfg);
	double centerValue(const GibsonLanniPSFConfig& config) const;

	std::unique_ptr<NumericalIntegrator> numericalIntegrator;
    std::shared_ptr<GibsonLanniPSFConfig> config;
    mutable BesselHelper besselHelper;

	mutable std::map<double, std::vector<double>> cachedRadialProfiles;
	mutable std::mutex cacheMutex;
};


class GibsonLanniIntegrand {
public:
    GibsonLanniIntegrand(const GibsonLanniPSFConfig& config, double r, const BesselHelper& besselHelper);
	std::array<double, 2> operator()(double rho) const;

private:
	const GibsonLanniPSFConfig& config;
	const double r;
	double k0;
	double k0NAr;
    const BesselHelper& besselHelper;
};

/**
 * Fixed-node composite Simpson quadrature for the Gibson-Lanni radial
 * integral. The phase W(rho) depends only on rho, not on the evaluation
 * radius, so it is tabulated once per slice. Every radius is then a plain
 * multiply-accumulate over tabulated J0 values (BesselHelper), which avoids
 * the per-radius adaptive restarts of SimpsonIntegrator.
 */
class GibsonLanniQuadrature {
public:
    // Node count for a given max argument of J0(k0*NA*r*rho) and the
    // accuracy setting (1..10); scaled to resolve the fastest oscillation.
    static size_t nodeCount(double maxArgument, int accuracy);

    // Tabulates phase terms on (0, upperBound] with n nodes (n even).
    void init(const GibsonLanniPSFConfig& config, double upperBound, size_t n);

    // |integral|^2 of the Gibson-Lanni integrand at radius r_nm (in nm).
    double evaluateSquared(double k0NAr, const BesselHelper& besselHelper) const;

    size_t nodeCount() const { return rho.size(); }

private:
    std::vector<double> rho;      // nodes on (0, upperBound]
    std::vector<double> cosW;     // cos(W(rho_j)) * rho_j * weight_j
    std::vector<double> sinW;     // sin(W(rho_j)) * rho_j * weight_j
};

