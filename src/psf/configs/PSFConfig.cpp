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

#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/PSFGeneratorFactory.h"

#include <algorithm>
#include <unordered_set>
#include <spdlog/spdlog.h>

ConfigMap resolutionCriterionMap{{
    {"rayleigh", static_cast<int>(PSFConfig::ResolutionCriterion::RAYLEIGH)},
    {"fwhm", static_cast<int>(PSFConfig::ResolutionCriterion::FWHM)},
    {"sigma", static_cast<int>(PSFConfig::ResolutionCriterion::SIGMA)},
}};

ConfigMap microscopeModalityMap{{
    {"widefield", static_cast<int>(MicroscopeModality::WIDEFIELD)},
    {"confocal", static_cast<int>(MicroscopeModality::CONFOCAL)},
}};

PSFConfig::PSFConfig(){
    registerAllParameters();
}

PSFConfig::PSFConfig(const PSFConfig& other)
    : Config()  // Delegate to default constructor first (registers parameters)
{
    // Clone the modality config BEFORE registering: the forwarded parameter
    // entries point into this instance's modality object.
    modalityConfig = other.modalityConfig
        ? std::shared_ptr<ModalityConfig>(other.modalityConfig->clone())
        : nullptr;
    // Then copy the values
    autoSize = other.autoSize;
    ID = other.ID;
    psfModelName = other.psfModelName;
    sizeX = other.sizeX;
    sizeY = other.sizeY;
    sizeZ = other.sizeZ;
    cutoffThreshold = other.cutoffThreshold;
    resolutionCriterion = other.resolutionCriterion;
    oversampling_factor = other.oversampling_factor;
    normalizePSF = other.normalizePSF;
    registerAllParameters();

    // Copy any other members
}

void PSFConfig::selectModality(MicroscopeModality modality){
    if (modalityConfig && modalityConfig->modality() == modality)
        return;
    modalityConfig = makeModalityConfig(modality);
    rebuildParameters();
}

void PSFConfig::rebuildParameters(){
    parameters.clear();
    // Virtual call: rebuilds the whole chain (base + forwarded modality
    // parameters + derived model parameters).
    registerAllParameters();
}

MicroscopeModality PSFConfig::modalityFromJSON(const json& jsonData){
    const auto it = jsonData.find("microscope_modality");
    if (it == jsonData.end() || !it->is_string())
        return MicroscopeModality::WIDEFIELD;

    const std::string modality = it->get<std::string>();
    // Resolve against the shared map so the accepted spellings live in one
    // place. lookupConfigMap warns + falls back on an unknown tag.
    const auto& entries = microscopeModalityMap.getMap();
    for (const auto& [name, value] : entries) {
        if (name == modality)
            return static_cast<MicroscopeModality>(value);
    }
    spdlog::get("config")->warn("Unknown microscope_modality '{}' in {}; using '{}'",
                                modality, jsonData.value("model_name", std::string("PSF config")),
                                entries.front().first);
    return static_cast<MicroscopeModality>(entries.front().second);
}

bool PSFConfig::loadFromJSON(const json& jsonData){
    // Select the modality config object BEFORE the generic pass: its
    // parameters are forwarded into this registry and must exist when the
    // pass loads them.
    const MicroscopeModality modality = modalityFromJSON(jsonData);
    selectModality(modality);
    // The optical model's wavelength is unused in confocal mode; the
    // emission/excitation wavelengths live on the modality config.
    if (modality == MicroscopeModality::CONFOCAL && jsonData.contains("wavelength_nm"))
        spdlog::warn("({}) 'wavelength_nm' is unused for confocal PSFs; set 'emission_wavelength_nm' instead", getName());

    warnOnForeignModalityKeys(jsonData);
    return Config::loadFromJSON(jsonData);
}

json PSFConfig::writeToJSON() const {
    // The modality is carried by the config object, not by a registry
    // parameter, so it is written back explicitly to keep round-trips exact.
    json jsonData = Config::writeToJSON();
    jsonData["microscope_modality"] = getModalityName();
    return jsonData;
}




bool PSFConfig::compareDim(const PSFConfig &other) {
    if(this->sizeX != other.sizeX || this->sizeY != other.sizeY || this->sizeZ != other.sizeZ) {
        spdlog::error("All PSFs have to be the same size");
        return false;
    }
    return true;
}

std::shared_ptr<PSFConfig> PSFConfig::createFromJSONFile(const std::string& filePath){
    json jsonData = loadJSONFile(filePath);

    std::shared_ptr<PSFConfig> config ;
    if (jsonData.contains("psf_configs")){
        json psfJsonData = jsonData["psf_configs"];
        if (psfJsonData.is_array()){
            config = PSFConfig::createFromJSON(psfJsonData[0]);
        }
        else {
            config = PSFConfig::createFromJSON(psfJsonData);
        }
    }
    if (!config) {
        throw std::runtime_error("Failed to parse config file: " + filePath);
    }
    return config;
}


std::shared_ptr<PSFConfig> PSFConfig::createFromJSON(const json& jsonData){
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    std::shared_ptr<PSFConfig> config = factory.createConfig(jsonData);

    return config;

}

void PSFConfig::registerAllParameters(){
    parameters.clear();
    // Base PSF parameters
    // struct ConfigParameter: {type, value, name, optional, jsonTag, cliFlag, cliDesc, cliRequired, hasRange, minVal, maxVal, selection}
    parameters.push_back({ParameterType::Int, &sizeX, "Size X", false, "size_x", "--size_x", "PSF size X (0 = auto)", false, true, 0, 1024, nullptr});
    parameters.push_back({ParameterType::Int, &sizeY, "Size Y", false, "size_y", "--size_y", "PSF size Y (0 = auto)", false, true, 0, 1024, nullptr});
    parameters.push_back({ParameterType::Int, &sizeZ, "Size Z", false, "size_z", "--size_z", "PSF size Z (0 = auto)", false, true, 0, 512, nullptr});
    parameters.push_back({ParameterType::String, &ID, "ID", true, "id", "--id", "PSF identifier", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::String, &psfModelName, "Model Name", true, "model_name", "--model_name", "PSF model name", false, false, 0.0, 0.0, nullptr});
    parameters.push_back({ParameterType::Float, &oversampling_factor, "Oversampling Factor", true, "oversampling_factor", "--oversampling_factor", "Oversampling factor for PSF computation", false, true, 1.0, 10.0, nullptr});
    parameters.push_back({ParameterType::Float, &cutoffThreshold, "Cutoff Threshold", true, "cutoff_threshold", "--cutoff_threshold", "Relative intensity cutoff for auto-size PSF (0 = derive from padding strategy)", false, true, 0.0, 1.0, nullptr});
    const void* resolutionCriterionMap_p = static_cast<const void*>(&resolutionCriterionMap);
    parameters.push_back({ParameterType::Map, &resolutionCriterion, "Resolution Criterion", true, "resolution_criterion", "--resolution_criterion", "Optical-resolution criterion used to size the auto-size PSF box: rayleigh, fwhm, sigma", false, true, 0.0, 2.0, resolutionCriterionMap_p});
    parameters.push_back({ParameterType::Bool, &normalizePSF, "Normalize PSF", true, "normalize_psf", "--normalize_psf", "Normalise the generated PSF to unit integral flux", false, false, 0.0, 0.0, nullptr});
    forwardModalityParameters();
}

void PSFConfig::forwardModalityParameters(){
    if (!modalityConfig)
        return;
    // The entries point into the modality config object, which is stable
    // until selectModality() replaces it (and triggers a registry rebuild).
    modalityConfig->visitParams([this](auto& value, ConfigParameter& param) {
        parameters.push_back(param);
    });
}

void PSFConfig::warnOnForeignModalityKeys(const json& jsonData) const {
    // A key owned by a different modality than the active one is meaningless
    // here (e.g. confocal keys on a widefield config). The generic pass would
    // report it as unknown only if it is not forwarded; this makes the reason
    // explicit instead of a bare "unknown key".
    if (!modalityConfig)
        return;
    const std::vector<std::string> owned = modalityConfig->jsonTags();
    for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
        if (std::find(owned.begin(), owned.end(), it.key()) != owned.end())
            continue;
        if (!foreignModalityKeys().contains(it.key()))
            continue;
        spdlog::get("config")->warn("({}) '{}' belongs to a different microscope modality; the active one is '{}' - ignoring",
                                    getName(), it.key(), getModalityName());
    }
}

const std::unordered_set<std::string>& PSFConfig::foreignModalityKeys() {
    // Union of every modality config's keys, so a key is recognised as
    // foreign regardless of which modality happens to be active.
    static const std::unordered_set<std::string> keys = [] {
        std::unordered_set<std::string> all;
        for (const auto& [name, value] : microscopeModalityMap.getMap()) {
            auto cfg = makeModalityConfig(static_cast<MicroscopeModality>(value));
            for (const auto& tag : cfg->jsonTags())
                all.insert(tag);
        }
        return all;
    }();
    return keys;
}
