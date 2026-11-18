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
#include <string>
#include <unordered_set>

#include "dolphin/Config.h"
#include "dolphin/psf/configs/ModalityConfig.h"
#include "dolphinbackend/CuboidShape.h"

class PSFConfig : public Config{
public:
    PSFConfig();

    virtual ~PSFConfig(){};

    PSFConfig(const PSFConfig& other);

    // Serialized modality tag; the config OBJECT below is the source of truth
    // in memory (see modalityConfig).
    using MicroscopeModality = ::MicroscopeModality;

    std::string getName() const override{
        return std::string("PSFConfig " + getModelName());
    }

    std::string getModelName() const {
        return psfModelName;
    }

    std::string getModalityName() const {
        return modalityConfig->modalityName();
    }

    MicroscopeModality getModalityType() const{
        return modalityConfig->modality();
    }

    /** Typed access to the modality config; dynamic_cast once, then read the
     *  modality's parameters directly:
     *   auto* confocal = config.modality<ConfocalModalityConfig>(); */
    template<typename ModalityConfigT>
    const ModalityConfigT* modality() const {
        return dynamic_cast<const ModalityConfigT*>(modalityConfig.get());
    }

    /** Replace the modality config object (rebuilding the forwarded
     *  parameters). No-op when the modality is already active. */
    void selectModality(MicroscopeModality modality);

    /** Modality tag of a JSON config, defaulting to widefield. Invalid tags
     *  warn and fall back to widefield. Callers that load a config should
     *  call selectModality() with this BEFORE visiting the parameter
     *  registry: the modality config must exist for its parameters to load. */
    static MicroscopeModality modalityFromJSON(const json& jsonData);

    CuboidShape getShape() const {
        return CuboidShape{sizeX, sizeY, sizeZ};
    }

    static std::shared_ptr<PSFConfig> createFromJSONFile(const std::string& filePath);
    static std::shared_ptr<PSFConfig> createFromJSON(const json& jsonData);

    bool loadFromJSON(const json& jsonData) override;

    /** Registry-driven JSON plus the selected modality, so a round-trip
     *  reproduces the config's modality. */
    json writeToJSON() const override;

    bool compareDim(const PSFConfig &other);


    std::string psfModelName;
    std::string ID;
    bool autoSize = false;

    /** Config object of the active microscope modality. Its concrete type
     *  selects the image-formation model (e.g. Gibson-Lanni computes a
     *  widefield or a confocal PSF) and carries that modality's parameters.
     *  The modality's parameters are forwarded into this config's flat
     *  parameter registry, so they load/save exactly like other parameters.
     *  This is the single source of truth for the modality: there is no
     *  separate enum member to keep in sync. */
    std::shared_ptr<ModalityConfig> modalityConfig = std::make_shared<WidefieldModalityConfig>();

    /** Base radial sampling density (samples per lateral pixel) for models
     *  that subsample their slices. Modalities may request a multiple of it
     *  when their composition needs finer sampling. */
    float oversampling_factor = 4.0f;

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

private:
    /** Rebuild the flat parameter registry: base parameters plus the
     *  forwarded parameters of the active modality config, followed by the
     *  derived model's parameters (via the virtual chain). */
    void rebuildParameters();

    /** Forward the active modality config's parameters into this registry.
     *  The entries point into the modality object, which stays stable until
     *  the modality is switched. */
    void forwardModalityParameters();

    /** Warn about JSON keys that belong to a modality other than the active
     *  one: they are ignored, and the reason is made explicit instead of
     *  surfacing as a bare unknown key. */
    void warnOnForeignModalityKeys(const json& jsonData) const;

    /** Union of every modality's JSON keys, used to recognise foreign keys. */
    static const std::unordered_set<std::string>& foreignModalityKeys();

};

