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

#include "CLIFrontend.h"
#include <spdlog/spdlog.h>
#include <string>
#include <dolphin/Dolphin.h>
#include <dolphin/PSFCreator.h>

CLIFrontend::CLIFrontend(Dolphin* dolphin, int argc, char** argv)
    : IFrontend(dolphin){
        this->argc = argc;
        this->argv = argv;
        psfCLI = app.add_subcommand("psfgenerator", "Generate PSF file");
        deconvolutionCLI = app.add_subcommand("deconvolution", "Run deconvolution");
    }


bool CLIFrontend::parseCLI(){
    try{
        app.parse(argc, argv);
        return true;
    }
    catch (const CLI::CallForHelp&) {
        std::cout << app.help() << std::endl;
        return false;
    }
    catch (const CLI::ParseError& e) {
        spdlog::error("{}", e.what());
        std::cout << app.help() << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        std::cout << app.help() << std::endl;
        return false;
    }
}



void CLIFrontend::run() {
    psfgenerator();
    deconvolution();

    bool success = parseCLI();
    if (!success) {
        return;
    }

    if (*psfCLI) {
        if (!configPath.empty()) {
            try {
                loadPSFJSONBundle(configPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!setupConfigPath.empty()) {
            try {
                loadSetupConfigFromFile(setupConfigPath, jsonPsfBundle.setupConfig);
                jsonPsfBundle.hasSetup = true;
                spdlog::info("Setup config loaded from: {}", setupConfigPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!psfConfigPaths.empty()) {
            try {
                for (const auto& path : psfConfigPaths) {
                    auto configs = loadPSFConfigsFromFile(path);
                    jsonPsfBundle.psfConfigs.insert(jsonPsfBundle.psfConfigs.end(), configs.begin(), configs.end());
                }
                jsonPsfBundle.hasPSF = true;
                spdlog::info("Loaded {} PSF config(s) from {} file(s)", jsonPsfBundle.psfConfigs.size(), psfConfigPaths.size());
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        PSFConfigBundle merged = mergePSFBundles(jsonPsfBundle, cliPsfBundle);

        if (!handlePSFGeneration(merged)) {
            return;
        }

        PSFGenerationRequest request = generatePSFRequest(merged);
        dolphin->generatePSF(request);
    }
    else if (*deconvolutionCLI) {
        if (!configPath.empty()) {
            try {
                loadJSONBundle(configPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!setupConfigPath.empty()) {
            try {
                loadSetupConfigFromFile(setupConfigPath, jsonBundle.setupConfig);
                jsonBundle.hasSetup = true;
                spdlog::info("Setup config loaded from: {}", setupConfigPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!deconvConfigPath.empty()) {
            try {
                loadDeconvConfigFromFile(deconvConfigPath, jsonBundle.deconvConfig);
                jsonBundle.hasDeconv = true;
                spdlog::info("Deconvolution config loaded from: {}", deconvConfigPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!psfConfigPaths.empty()) {
            try {
                for (const auto& path : psfConfigPaths) {
                    auto configs = loadPSFConfigsFromFile(path);
                    jsonBundle.psfConfigs.insert(jsonBundle.psfConfigs.end(), configs.begin(), configs.end());
                }
                jsonBundle.hasPSF = true;
                spdlog::info("Loaded {} PSF config(s) from {} file(s)", jsonBundle.psfConfigs.size(), psfConfigPaths.size());
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        ConfigBundle merged = mergeBundles(jsonBundle, cliBundle);

        if (!handleDeconvolution(merged)) {
            return;
        }

        DeconvolutionRequest request = generateDeconvRequest(merged);
        dolphin->deconvolve(request);
    }
    else {
        std::cout << app.help() << std::endl;
    }
}

void CLIFrontend::psfgenerator() {
    CLI::Option_group* psf_group = psfCLI->add_option_group("PSF Options", "PSF generation options");
    psf_group->add_option("-c,--config", configPath, "Path to combined configuration file");
    psf_group->add_option("-s,--setup_config", setupConfigPath, "Path to setup config JSON file");
    psf_group->add_option("-p,--psf_configs", psfConfigPaths, "Path(s) to PSF config JSON file(s)")->multi_option_policy(CLI::MultiOptionPolicy::TakeAll);

    psfconfigGroup = psf_group;
    psfcli_group = psfCLI->add_option_group("CLI", "PSF Commandline options");

    addParameters(cliPsfBundle.setupConfig, psfcli_group);
    cliPsfBundle.hasSetup = true;
}

void CLIFrontend::deconvolution() {
    readCLISetupConfigPath();
    readSetupConfigParameters();
    readCLIParametersDeconvolution();
}

bool CLIFrontend::handlePSFGeneration(const PSFConfigBundle& bundle) {
    std::vector<std::string> missingParams = checkRequired(const_cast<SetupConfigPSF&>(bundle.setupConfig));
    if (!missingParams.empty()) {
        std::cout << psfCLI->help() << std::endl;
        return false;
    }

    if (!bundle.hasPSF) {
        spdlog::error("No PSF config provided — use -p/--psf_configs or inline psf_configs in JSON");
        std::cout << psfCLI->help() << std::endl;
        return false;
    }

    return true;
}

void CLIFrontend::loadJSONBundle(const std::string& path) {
    json jsonData = Config::loadJSONFile(path);

    if (loadSetupConfigFromJSON(jsonData, jsonBundle.setupConfig)) {
        jsonBundle.hasSetup = true;
    }

    if (jsonData.contains("deconvolution_config")) {
        loadDeconvConfigFromJSON(jsonData, jsonBundle.deconvConfig);
        jsonBundle.hasDeconv = true;
    }

    auto psfConfigs = loadPSFConfigsFromJSON(jsonData);
    if (!psfConfigs.empty()) {
        jsonBundle.psfConfigs.insert(jsonBundle.psfConfigs.end(), psfConfigs.begin(), psfConfigs.end());
        jsonBundle.hasPSF = true;
        spdlog::info("Loaded {} inline PSF config(s) from JSON", psfConfigs.size());
    }

    spdlog::info("Configuration loaded from: {}", path);
}

ConfigBundle CLIFrontend::mergeBundles(const ConfigBundle& jsonBundle, const ConfigBundle& cliBundle) {
    ConfigBundle merged;

    if (jsonBundle.hasSetup) {
        if (cliBundle.hasSetup) {
            spdlog::warn("Setup config loaded from JSON — CLI setup args ignored");
        }
        merged.setupConfig = jsonBundle.setupConfig;
    } else {
        merged.setupConfig = cliBundle.setupConfig;
    }
    merged.hasSetup = true;

    if (jsonBundle.hasDeconv) {
        if (cliBundle.hasDeconv) {
            spdlog::warn("Deconvolution config loaded from JSON — CLI deconvolution args ignored");
        }
        merged.deconvConfig = jsonBundle.deconvConfig;
    } else {
        merged.deconvConfig = cliBundle.deconvConfig;
    }
    merged.hasDeconv = true;

    if (jsonBundle.hasPSF || cliBundle.hasPSF) {
        if (jsonBundle.hasPSF) {
            merged.psfConfigs.insert(merged.psfConfigs.end(), jsonBundle.psfConfigs.begin(), jsonBundle.psfConfigs.end());
        }
        if (cliBundle.hasPSF) {
            merged.psfConfigs.insert(merged.psfConfigs.end(), cliBundle.psfConfigs.begin(), cliBundle.psfConfigs.end());
        }
        merged.hasPSF = true;
    }

    return merged;
}

void CLIFrontend::loadPSFJSONBundle(const std::string& path) {
    json jsonData = Config::loadJSONFile(path);

    if (loadSetupConfigFromJSON(jsonData, jsonPsfBundle.setupConfig)) {
        jsonPsfBundle.hasSetup = true;
    }

    auto psfConfigs = loadPSFConfigsFromJSON(jsonData);
    if (!psfConfigs.empty()) {
        jsonPsfBundle.psfConfigs.insert(jsonPsfBundle.psfConfigs.end(), psfConfigs.begin(), psfConfigs.end());
        jsonPsfBundle.hasPSF = true;
        spdlog::info("Loaded {} inline PSF config(s) from JSON", psfConfigs.size());
    }

    spdlog::info("Configuration loaded from: {}", path);
}

PSFConfigBundle CLIFrontend::mergePSFBundles(const PSFConfigBundle& jsonBundle, const PSFConfigBundle& cliBundle) {
    PSFConfigBundle merged;

    if (jsonBundle.hasSetup) {
        if (cliBundle.hasSetup) {
            spdlog::warn("Setup config loaded from JSON — CLI setup args ignored");
        }
        merged.setupConfig = jsonBundle.setupConfig;
    } else {
        merged.setupConfig = cliBundle.setupConfig;
    }
    merged.hasSetup = true;

    if (jsonBundle.hasPSF || cliBundle.hasPSF) {
        if (jsonBundle.hasPSF) {
            merged.psfConfigs.insert(merged.psfConfigs.end(), jsonBundle.psfConfigs.begin(), jsonBundle.psfConfigs.end());
        }
        if (cliBundle.hasPSF) {
            merged.psfConfigs.insert(merged.psfConfigs.end(), cliBundle.psfConfigs.begin(), cliBundle.psfConfigs.end());
        }
        merged.hasPSF = true;
    }

    return merged;
}


bool CLIFrontend::handleDeconvolution(const ConfigBundle& bundle) {
    std::vector<std::string> missingParams = checkRequired(const_cast<DeconvolutionConfig&>(bundle.deconvConfig));
    std::vector<std::string> missingParamsSetup = checkRequired(const_cast<SetupConfig&>(bundle.setupConfig));
    missingParams.insert(missingParams.end(), missingParamsSetup.begin(), missingParamsSetup.end());

    if (!missingParams.empty()) {
        spdlog::error("Required parameter(s) missing:");
        for (const auto& p : missingParams) {
            spdlog::error("  - {}", p);
        }
        spdlog::info("{}", deconvolutionCLI->help());
        return false;
    }

    if (!bundle.hasPSF && bundle.setupConfig.psfFilePaths.empty()) {
        spdlog::error("PSF has to be provided either as json config or as a tiff file");
        spdlog::info("{}", deconvolutionCLI->help());
        return false;
    }

    return true;
}




void CLIFrontend::readSetupConfigParameters() {
    setupCliGroup = deconvolutionCLI->add_option_group("Setup CLI", "Setup commandline options");
    addParameters(cliBundle.setupConfig, setupCliGroup);
    cliBundle.hasSetup = true;
}

void CLIFrontend::readCLIParametersDeconvolution() {
    deconvCliGroup = deconvolutionCLI->add_option_group("Deconvolution CLI", "Deconvolution commandline options");
    addParameters(cliBundle.deconvConfig, deconvCliGroup);
    cliBundle.hasDeconv = true;
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", configPath, "Path to combined configuration file");
    config_group->add_option("-s,--setup_config", setupConfigPath, "Path to setup config JSON file");
    config_group->add_option("-d,--deconv_config", deconvConfigPath, "Path to deconvolution config JSON file");
    config_group->add_option("-p,--psf_configs", psfConfigPaths, "Path(s) to PSF config JSON file(s)")->multi_option_policy(CLI::MultiOptionPolicy::TakeAll);
}


void CLIFrontend::addParameters(Config& config, CLI::Option_group* group){

    config.visitParams([this, group]<typename T>(T& value, ConfigParameter& param){

        if constexpr (std::is_same_v<T, std::array<int, 3>>){
            return;
        }
        else {
            if constexpr (std::is_same_v<T, bool>){
                auto opt = group->add_flag(param.cliFlag, value, param.cliDesc);
                opt->configurable(false);
                return;
            }
            if (std::string(param.cliFlag) == "--psf_file_paths"){
                if (!psfPathGroup) {
                    psfPathGroup = group->add_option_group("PSF Path", "PSF file path options");
                }
                auto opt = psfPathGroup->add_option(param.cliFlag, value, param.cliDesc);
                opt->configurable(false);
                opt->ignore_case();
                return;
            }
            if (param.type == ParameterType::StringSelection && param.selection) {
                const auto* options = static_cast<const std::vector<std::string>*>(param.selection);
                auto opt = group->add_option(param.cliFlag, value, param.cliDesc);
                opt->check(CLI::IsMember(*options));
                opt->configurable(false);
                return;
            }
            if (param.type == ParameterType::Map && param.selection) {
                const ConfigMap& map = *reinterpret_cast<const ConfigMap*>(param.selection);
                std::vector<std::string> options = map.getStrings();
                auto opt = group->add_option(param.cliFlag, value, param.cliDesc);
                opt->check(CLI::IsMember(options));
                opt->configurable(false);
                return;
            }
            auto opt = group->add_option(param.cliFlag, value, param.cliDesc);
            opt->configurable(false);
        }
    });
}


std::vector<std::string> CLIFrontend::checkRequired(Config& config) const {
    std::vector<std::string> missingParams;

    auto checkRequired = [&missingParams]<typename T>(T& value, ConfigParameter& param) {
        if (!param.cliRequired) return;
        if constexpr (std::is_same_v<T, std::string>) {
            if (value.empty()) {
                missingParams.push_back(std::string(param.name) + " (" + param.cliFlag + ")");
            }
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            if (value.empty()) {
                missingParams.push_back(std::string(param.name) + " (" + param.cliFlag + ")");
            }
        }
    };

    config.visitParams(checkRequired);
    return missingParams;
}



std::shared_ptr<PSFConfig> CLIFrontend::loadPSFConfigFromPath(const std::string& path) {
    return PSFCreator::generatePSFConfigFromConfigPath(path);
}

bool CLIFrontend::loadSetupConfigFromJSON(const json& jsonData, SetupConfigPSF& config) {
    if (jsonData.contains("setup_config")) {
        if (!config.loadFromJSON(jsonData["setup_config"])) {
            throw std::runtime_error("Failed to parse setup config");
        }
        return true;
    }
    return false;
}

void CLIFrontend::loadDeconvConfigFromJSON(const json& jsonData, DeconvolutionConfig& config) {
    if (jsonData.contains("deconvolution_config")) {
        if (!config.loadFromJSON(jsonData["deconvolution_config"])) {
            throw std::runtime_error("Failed to parse deconvolution config");
        }
    }
}

std::vector<std::shared_ptr<PSFConfig>> CLIFrontend::loadPSFConfigsFromJSON(const json& jsonData) {
    std::vector<std::shared_ptr<PSFConfig>> configs;

    if (jsonData.contains("psf_configs")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        for (const auto& psfJson : jsonData["psf_configs"]) {
            configs.push_back(factory.createConfig(psfJson));
        }
    } else if (jsonData.contains("model_name")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        configs.push_back(factory.createConfig(jsonData));
    }

    return configs;
}

void CLIFrontend::loadSetupConfigFromFile(const std::string& path, SetupConfigPSF& config) {
    json jsonData = Config::loadJSONFile(path);
    const json& setupData = jsonData.contains("setup_config") ? jsonData["setup_config"] : jsonData;
    if (!config.loadFromJSON(setupData)) {
        throw std::runtime_error("Failed to parse setup config file: " + path);
    }
}

void CLIFrontend::loadDeconvConfigFromFile(const std::string& path, DeconvolutionConfig& config) {
    json jsonData = Config::loadJSONFile(path);
    const json& deconvData = jsonData.contains("deconvolution_config") ? jsonData["deconvolution_config"] : jsonData;
    if (!config.loadFromJSON(deconvData)) {
        throw std::runtime_error("Failed to parse deconvolution config file: " + path);
    }
}

std::vector<std::shared_ptr<PSFConfig>> CLIFrontend::loadPSFConfigsFromFile(const std::string& path) {
    json jsonData = Config::loadJSONFile(path);
    auto configs = loadPSFConfigsFromJSON(jsonData);
    if (configs.empty()) {
        throw std::runtime_error("PSF config file must contain 'psf_configs' array or a single PSF config with 'model_name': " + path);
    }
    return configs;
}


void progressVisualization(std::atomic<float>& current, float max){
    float barWidth = 50.0f;
    int pos = static_cast<int>((current * barWidth) / max);
    int progress = static_cast<int>((current * 100) / max);
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
      << std::setw(3)
      << progress << "%";
    std::cout.flush();

    if(current >= max){
        std::cout <<std::endl;
    }
}

void loggingCallback(spdlog::level::level_enum level, const std::string& message){
    if (level >= spdlog::level::info){
        std::cout << "[" << spdlog::level::to_string_view(level).data() << "] " <<  message << "\n";
    }
}

PSFGenerationRequest CLIFrontend::generatePSFRequest(const PSFConfigBundle& bundle) {
    auto setupConfig = std::make_shared<SetupConfigPSF>(bundle.setupConfig);
    PSFGenerationRequest request(setupConfig, loggingCallback, progressVisualization);
    if (bundle.hasPSF) {
        request.setInlinePSFConfigs(bundle.psfConfigs);
    }
    return request;
}

DeconvolutionRequest CLIFrontend::generateDeconvRequest(const ConfigBundle& bundle) {
    auto setupConfig = std::make_shared<SetupConfig>(bundle.setupConfig);
    auto deconvConfig = std::make_shared<DeconvolutionConfig>(bundle.deconvConfig);
    DeconvolutionRequest request(setupConfig, deconvConfig, loggingCallback, progressVisualization);
    if (bundle.hasPSF) {
        request.setInlinePSFConfigs(bundle.psfConfigs);
    }
    return request;
}
