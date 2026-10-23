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

#include "dolphin/DefaultParameters.h"
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr const char* kConfigFileName = "configs_template/default_parameters.json";
constexpr const char* kConfigPathEnvVar = "DOLPHIN_DEFAULT_PARAMETERS";

std::filesystem::path getExecutableDir() {
#ifdef _WIN32
    char buffer[4096];
    while (DWORD length = GetModuleFileNameA(nullptr, buffer, sizeof(buffer) - 1)) {
        if (length < sizeof(buffer) - 1) {
            return std::filesystem::path(buffer).parent_path();
        }
    }
#elif __APPLE__
    char buffer[4096];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0 && size < sizeof(buffer)) {
        return std::filesystem::path(buffer).parent_path();
    }
#else
    char buffer[4096];
    ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (length > 0) {
        buffer[length] = '\0';
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

}

void DefaultParameters::registerAllParameters(){
    parameters.push_back({ParameterType::Float, &lowPaddingThreshold_, "Low padding threshold", true, "low_padding_threshold", "--low_padding_threshold", "Low padding threshold", false, true, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Float, &mediumPaddingThreshold_, "Medium padding threshold", true, "medium_padding_threshold", "--medium_padding_threshold", "Medium padding threshold", false, true, 0.0, 1.0, nullptr});
    parameters.push_back({ParameterType::Float, &generationThresholdMultiplier_, "Generation Threshold Multiplier", true, "generation_threshold_multiplier", "--generation_threshold_multiplier", "Generation threshold multiplier", false, true, 0.0, 1.0, nullptr});
}

std::string DefaultParameters::resolveConfigPath() {
    if (const char* envPath = std::getenv(kConfigPathEnvVar)) {
        return std::string(envPath);
    }

    const auto execDir = getExecutableDir();
    const auto candidates = {
        execDir / kConfigFileName,
        execDir.parent_path() / kConfigFileName,
        execDir.parent_path() / "share" / "dolphin" / kConfigFileName,
    };
    for (const auto& candidate : candidates) {
        std::ifstream file(candidate);
        if (file.good()) {
            return candidate.string();
        }
    }

    return {};
}

const DefaultParameters& DefaultParameters::getDefaultParameters() {
    static const DefaultParameters& instance = [] -> const DefaultParameters& {
        static DefaultParameters config;
        const std::string path = resolveConfigPath();
        if (path.empty()) {
            spdlog::get("config")->warn("DefaultParameters: config file not found ({}), using built-in defaults", kConfigFileName);
        } else {
            json jsonData = loadJSONFile(path);
            if (!config.loadFromJSON(jsonData)) {
                throw std::runtime_error("Failed to parse DefaultParameters config file: " + path);
            }
            spdlog::get("config")->info("DefaultParameters loaded from: {}", path);
        }
        return config;
    }();
    return instance;
}
