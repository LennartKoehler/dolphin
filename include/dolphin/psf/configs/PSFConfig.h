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

#include <string>
#include <vector>

#include "dolphin/Config.h"
#include "dolphinbackend/CuboidShape.h"

class PSFConfig : public Config{
public:
    PSFConfig();

    virtual ~PSFConfig(){};

    PSFConfig(const PSFConfig& other);

    std::string getName() const override{
        return std::string("PSFConfig " + getModelName());
    }

    std::string getModelName() const {
        return psfModelName;
    }

    CuboidShape getShape() const {
        return CuboidShape{sizeX, sizeY, sizeZ};
    }

    static std::shared_ptr<PSFConfig> createFromJSONFile(const std::string& filePath);
    static std::shared_ptr<PSFConfig> createFromJSON(const json& jsonData);

    bool compareDim(const PSFConfig &other);


    std::string psfModelName;
    std::string ID;
    bool autoSize = false;

    size_t sizeX = 0;
    size_t sizeY = 0;
    size_t sizeZ = 0;

    /** Relative intensity cutoff for the auto-size PSF support (Gaussian model).
     *  GibsonLanni ignores this: its box is derived analytically from the
     *  resolution criterion (see resolutionCriterion). */
    float cutoffThreshold = 0.0f;

    /** Optical-resolution criterion used to size the auto-size PSF box.
     *  Mirrors Huygens' image_approxResol_crit. The box is derived from the
     *  approximate resolution and the voxel pitch, not from a value cutoff. */
    enum ResolutionCriterion : int {
        RAYLEIGH = 0,  // rayleigh
        FWHM = 1,      // fwhm
        SIGMA = 2,     // sigma
    };
    int resolutionCriterion = ResolutionCriterion::RAYLEIGH;

    /** Normalise the generated PSF so its total integral is 1 (unit flux),
     *  matching Huygens's flux normalisation. */
    bool normalizePSF = true;

protected:
    void registerAllParameters();


};

