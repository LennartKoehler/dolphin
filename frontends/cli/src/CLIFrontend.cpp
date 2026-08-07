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
        if (!setupConfigPath.empty()) {
            try {
                loadPSFJSONBundle(setupConfigPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!cliPsfConfigPath.empty()) {
            try {
                cliPsfBundle.psfConfigs.push_back(loadPSFConfigFromPath(cliPsfConfigPath));
                cliPsfBundle.hasPSF = true;
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
        if (!setupConfigPath.empty()) {
            try {
                loadJSONBundle(setupConfigPath);
            } catch (const std::exception& e) {
                spdlog::error("{}", e.what());
                return;
            }
        }

        if (!cliPsfConfigPaths.empty()) {
            try {
                for (const auto& path : cliPsfConfigPaths) {
                    cliBundle.psfConfigs.push_back(loadPSFConfigFromPath(path));
                }
                cliBundle.hasPSF = true;
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
    psf_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
    psf_group->add_option("-i,--psf_config_path", cliPsfConfigPath, "Path to PSF config JSON file");

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
        spdlog::error("No PSF config provided — use --psf_config_path or inline psf_configs in JSON");
        std::cout << psfCLI->help() << std::endl;
        return false;
    }

    return true;
}

void CLIFrontend::loadJSONBundle(const std::string& path) {
    json jsonData = Config::loadJSONFile(path);

    if (jsonData.contains("setup_config")) {
        jsonBundle.setupConfig.loadFromJSON(jsonData["setup_config"]);
        jsonBundle.hasSetup = true;
    } else {
        json rootData = jsonData;
        rootData.erase("deconvolution_config");
        rootData.erase("psf_configs");
        if (!rootData.empty()) {
            jsonBundle.setupConfig.loadFromJSON(rootData);
            jsonBundle.hasSetup = true;
        }
    }

    if (jsonData.contains("deconvolution_config")) {
        jsonBundle.deconvConfig.loadFromJSON(jsonData["deconvolution_config"]);
        jsonBundle.hasDeconv = true;
    }

    if (jsonData.contains("psf_configs")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        for (const auto& psfJson : jsonData["psf_configs"]) {
            jsonBundle.psfConfigs.push_back(factory.createConfig(psfJson));
        }
        jsonBundle.hasPSF = true;
        spdlog::info("Loaded {} inline PSF config(s) from JSON", jsonBundle.psfConfigs.size());
    }

    if (jsonData.contains("psf_config_paths")) {
        for (const auto& path : jsonData["psf_config_paths"]) {
            jsonBundle.psfConfigs.push_back(loadPSFConfigFromPath(path));
        }
        jsonBundle.hasPSF = true;
        spdlog::info("Loaded {} PSF config(s) from file paths in JSON", jsonData["psf_config_paths"].size());
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

    if (jsonData.contains("setup_config")) {
        jsonPsfBundle.setupConfig.loadFromJSON(jsonData["setup_config"]);
        jsonPsfBundle.hasSetup = true;
    } else {
        json rootData = jsonData;
        rootData.erase("deconvolution_config");
        rootData.erase("psf_configs");
        if (!rootData.empty()) {
            jsonPsfBundle.setupConfig.loadFromJSON(rootData);
            jsonPsfBundle.hasSetup = true;
        }
    }

    if (jsonData.contains("psf_configs")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        for (const auto& psfJson : jsonData["psf_configs"]) {
            jsonPsfBundle.psfConfigs.push_back(factory.createConfig(psfJson));
        }
        jsonPsfBundle.hasPSF = true;
        spdlog::info("Loaded {} inline PSF config(s) from JSON", jsonPsfBundle.psfConfigs.size());
    }

    if (jsonData.contains("psf_config_path")) {
        std::string path = jsonData["psf_config_path"];
        jsonPsfBundle.psfConfigs.push_back(loadPSFConfigFromPath(path));
        jsonPsfBundle.hasPSF = true;
        spdlog::info("Loaded PSF config from file path: {}", path);
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
    return true;
}




void CLIFrontend::readSetupConfigParameters() {
    setupCliGroup = deconvolutionCLI->add_option_group("Setup CLI", "Setup commandline options");
    addParameters(cliBundle.setupConfig, setupCliGroup);
    setupCliGroup->add_option("--multiple_psf_config_paths", cliPsfConfigPaths, "PSF config JSON file paths")
        ->configurable(false)->ignore_case();
    cliBundle.hasSetup = true;
}

void CLIFrontend::readCLIParametersDeconvolution() {
    deconvCliGroup = deconvolutionCLI->add_option_group("Deconvolution CLI", "Deconvolution commandline options");
    addParameters(cliBundle.deconvConfig, deconvCliGroup);
    cliBundle.hasDeconv = true;
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
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

std::shared_ptr<PSFConfig> CLIFrontend::loadPSFConfigFromPath(const std::string& path) {
    return PSFCreator::generatePSFConfigFromConfigPath(path);
}
