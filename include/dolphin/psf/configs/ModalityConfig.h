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
#include <vector>

#include "dolphin/Config.h"

/** Image-formation model of a PSF. Serialized via the "microscope_modality"
 *  JSON key / --microscope_modality CLI flag on every PSF config. */
enum class MicroscopeModality {
    WIDEFIELD = 0,
    CONFOCAL = 1,
};

/**
 * Config object of one microscope modality. A PSFConfig owns one instance
 * (PSFConfig::modalityConfig); the concrete type identifies the modality and
 * carries exactly the parameters of that modality's image-formation model.
 *
 * Generation code dynamic_casts the pointer once to obtain the typed config:
 *
 *   const auto* confocal = dynamic_cast<const ConfocalModalityConfig*>(config.modalityConfig.get());
 *
 * The config's parameters are forwarded into the owning PSFConfig's flat
 * parameter registry (see PSFConfig::registerAllParameters), so JSON keys,
 * CLI flags and round-trips behave exactly like ordinary PSF config
 * parameters.
 */
class ModalityConfig : public Config{
public:
    virtual ~ModalityConfig() = default;

    virtual MicroscopeModality modality() const = 0;

    /** Short modality name for logging: "widefield", "confocal", ... */
    virtual const char* modalityName() const = 0;

    virtual std::unique_ptr<ModalityConfig> clone() const = 0;

    /** JSON keys owned by this modality, derived from its own parameter
     *  registry. The owning PSFConfig uses this to decide which modality-only
     *  keys are meaningful for the active modality - a key owned by another
     *  modality is reported as unknown instead of silently ignored. */
    std::vector<std::string> jsonTags() {
        std::vector<std::string> tags;
        visitParams([&tags]<typename T>(T&, ConfigParameter& param) {
            tags.emplace_back(param.jsonTag);
        });
        return tags;
    }
};

/** Widefield image formation: a single detection integral. Currently adds no
 *  parameters of its own. */
class WidefieldModalityConfig : public ModalityConfig{
public:
    WidefieldModalityConfig();
    WidefieldModalityConfig(const WidefieldModalityConfig& other);

    std::string getName() const override {
        return std::string("ModalityConfig widefield");
    }
    MicroscopeModality modality() const override { return MicroscopeModality::WIDEFIELD; }
    const char* modalityName() const override { return "widefield"; }
    std::unique_ptr<ModalityConfig> clone() const override {
        return std::make_unique<WidefieldModalityConfig>(*this);
    }

protected:
    void registerAllParameters() override;
};

/**
 * Confocal image formation: the excitation integral and the detection
 * integral are combined (voxel-wise product, with the detection PSF averaged
 * over the pinhole disc for a finite pinhole). Carries the confocal-specific
 * parameters, including both fluorescence wavelengths.
 */
class ConfocalModalityConfig : public ModalityConfig{
public:
    ConfocalModalityConfig();
    ConfocalModalityConfig(const ConfocalModalityConfig& other);

    std::string getName() const override {
        return std::string("ModalityConfig confocal");
    }
    MicroscopeModality modality() const override { return MicroscopeModality::CONFOCAL; }
    const char* modalityName() const override { return "confocal"; }
    std::unique_ptr<ModalityConfig> clone() const override {
        return std::make_unique<ConfocalModalityConfig>(*this);
    }

    /** Emission (detection) wavelength in nm. The detection integral of the
     *  confocal product is computed at this wavelength. */
    float emissionWavelength_nm = 520.0f;

    /** Excitation wavelength in nm (Stokes shift:
     *  excitationWavelength_nm < emissionWavelength_nm). Values <= 0 fall
     *  back to emissionWavelength_nm, degenerating the confocal PSF toward
     *  the widefield model. */
    float excitationWavelength_nm = 488.0f;

    /** Pinhole radius at the sample plane in um. 0 = point pinhole (the
     *  detection PSF is used directly in the confocal product); > 0 = finite
     *  pinhole, averaging the detection PSF over the pinhole disc. */
    float pinholeRadius_um = 0.0f;

protected:
    void registerAllParameters() override;

private:
    void copyMembers(const ConfocalModalityConfig& other);
};

/** Factory for the modality config of a given modality tag. */
std::unique_ptr<ModalityConfig> makeModalityConfig(MicroscopeModality modality);
