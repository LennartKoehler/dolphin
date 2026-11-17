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

#include <cstddef>
#include <map>
#include <mutex>
#include <optional>
#include <memory>
#include <vector>

#include "dolphin/psf/generators/BesselHelper.h"
#include "dolphin/psf/modality/PSFModel.h"

class GibsonLanniPSFConfig;

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

/**
 * Gibson-Lanni diffraction model. Produces the |integral|^2 intensity of one
 * wavelength at one defocus. Radial symmetry about the optical axis is an
 * internal optimization: slices are computed as radial profiles (fixed-node
 * quadrature over a shared Bessel table) and returned as RadialSlice for the
 * modality layer to blit.
 *
 * Orchestration data (PSFModel facade):
 *   - autoBox: the PSFBoxSizing defocus-cone math from NA, RI and pitch;
 *   - sliceConfig: detunes the working distance ti_nm by pitch * offset;
 *   - detectionLambda: the optical model's wavelength_nm.
 *
 * Stateless apart from transparent caches (radial profiles per
 * (ti_nm, lambda_nm), the Bessel table); invalidateCache() drops them when
 * the optical configuration changes.
 */
class GibsonLanniModel : public PSFModel {
public:
    void init(const PSFConfig& config,
              size_t sliceWidth,
              size_t sliceHeight,
              std::optional<double> shortestLambda) const override;

    ModelSlice intensitySlice(const PSFConfig& sliceCfg,
                              std::optional<double> lambda,
                              size_t samplesPerPixel,
                              size_t maxLateralPixels,
                              size_t forcedCutoff) const override;

    ModelBox autoBox(const PSFConfig& config,
                     std::optional<double> detectionLambda) const override;

    std::shared_ptr<const PSFConfig> sliceConfig(const PSFConfig& config,
                                                 long defocusOffset) const override;

    std::optional<double> detectionLambda(const PSFConfig& config) const override;

    /** Pupil-domain integration upper bound (tightest rho limit across the
     *  media in the path). Public: used for the once-per-generation
     *  NA*rho/ns warning. */
    static double integrationUpperBound(const GibsonLanniPSFConfig& cfg);

    /** Drop cached radial profiles and the Bessel table. */
    void invalidateCache();

private:
    /** Typed view of a config handed to this model. */
    static const GibsonLanniPSFConfig& glConfig(const PSFConfig& config);

    /** Radial |integral|^2 profile of one wavelength, cached per
     *  (ti_nm, lambda_nm). */
    std::vector<double> radialProfile(const GibsonLanniPSFConfig& sliceCfg,
                                      double lambda_nm,
                                      size_t samplesPerPixel,
                                      size_t maxLateralPixels,
                                      size_t forcedCutoff) const;

    void initBesselHelper(const GibsonLanniPSFConfig& config,
                          size_t sizeX, size_t sizeY,
                          double shortestLambda_nm) const;

    mutable BesselHelper besselHelper;
    mutable bool besselInitialized = false;
    mutable std::map<std::pair<double, double>, std::vector<double>> cachedRadialProfiles;
    mutable std::mutex cacheMutex;
};
