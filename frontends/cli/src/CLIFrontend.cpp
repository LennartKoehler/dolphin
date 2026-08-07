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
        bool success = readPSFFromConfigFile();
        success = success && handlePSFGeneration();
        if (success) {
            PSFGenerationRequest request = generatePSFRequest(std::make_shared<SetupConfigPSF>(psfConfig));
            dolphin->generatePSF(request);
        }
    }
    else if (*deconvolutionCLI) {
        bool success = readDeconvolutionFromConfigFile();
        success = success && handleDeconvolution();
        if (success) {
            DeconvolutionRequest request = generateDeconvRequest(std::make_shared<SetupConfig>(setupConfig), std::make_shared<DeconvolutionConfig>(deconvolutionConfig));
            dolphin->deconvolve(request);
        }
    }
    else {
        std::cout << app.help() << std::endl;
    }
}

void CLIFrontend::psfgenerator() {
    CLI::Option_group* psf_group = psfCLI->add_option_group("PSF Options", "PSF generation options");
    psf_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");

    psfconfigGroup = psf_group;
    psfcli_group = psfCLI->add_option_group("CLI", "PSF Commandline options");

    addParameters(psfConfig, psfcli_group);
}

void CLIFrontend::deconvolution() {
    readCLISetupConfigPath();
    readSetupConfigParameters();
    readCLIParametersDeconvolution();
}

bool CLIFrontend::handlePSFGeneration() {
    std::vector<std::string> missingParams = checkRequired(psfConfig);
    if (!missingParams.empty()) {
        std::cout << psfCLI->help() << std::endl;
        return false;
    }

    return true;
}

bool CLIFrontend::readDeconvolutionFromConfigFile() {
    if (setupConfigPath.empty()) return true;

    try {
        json jsonData = Config::loadJSONFile(setupConfigPath);

        bool hasSetupConfig = jsonData.contains("setup_config") ||
            (!jsonData.contains("deconvolution_config") && !jsonData.contains("psf_configs"));
        bool hasDeconvConfig = jsonData.contains("deconvolution_config");

        if (jsonData.contains("setup_config")) {
            hasSetupConfig = true;
        }

        if (hasSetupConfig) {
            if (groupHasOptions(setupCliGroup)) {
                spdlog::warn("Setup config loaded from JSON — CLI setup args ignored");
            }
            json setupData = jsonData.contains("setup_config")
                ? jsonData["setup_config"]
                : jsonData;
            setupData.erase("deconvolution_config");
            setupData.erase("psf_configs");
            setupConfig.loadFromJSON(setupData);
        }

        if (hasDeconvConfig) {
            if (groupHasOptions(deconvCliGroup)) {
                spdlog::warn("Deconvolution config loaded from JSON — CLI deconvolution args ignored");
            }
            deconvolutionConfig.loadFromJSON(jsonData["deconvolution_config"]);
        }

        if (jsonData.contains("psf_configs")) {
            PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
            for (const auto& psfJson : jsonData["psf_configs"]) {
                inlinePsfConfigs.push_back(factory.createConfig(psfJson));
            }
            spdlog::info("Loaded {} inline PSF config(s) from JSON", inlinePsfConfigs.size());
        }

        spdlog::info("Configuration loaded from: {}", setupConfigPath);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return false;
    }
    return true;
}

bool CLIFrontend::readPSFFromConfigFile() {
    if (setupConfigPath.empty()) return true;

    try {
        json jsonData = Config::loadJSONFile(setupConfigPath);

        bool hasSetupConfig = jsonData.contains("setup_config") ||
            (!jsonData.contains("deconvolution_config") && !jsonData.contains("psf_configs"));

        if (jsonData.contains("setup_config")) {
            hasSetupConfig = true;
        }

        if (hasSetupConfig) {
            if (groupHasOptions(psfcli_group)) {
                spdlog::warn("Setup config loaded from JSON — CLI args ignored");
            }
            json setupData = jsonData.contains("setup_config")
                ? jsonData["setup_config"]
                : jsonData;
            setupData.erase("deconvolution_config");
            setupData.erase("psf_configs");
            psfConfig.loadFromJSON(setupData);
        }

        spdlog::info("Configuration loaded from: {}", setupConfigPath);
    } catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        return false;
    }
    return true;
}


bool CLIFrontend::handleDeconvolution() {
    std::vector<std::string> missingParams = checkRequired(deconvolutionConfig);
    std::vector<std::string> missingParamsSetup = checkRequired(setupConfig);
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
    addParameters(setupConfig, setupCliGroup);
}

void CLIFrontend::readCLIParametersDeconvolution() {
    deconvCliGroup = deconvolutionCLI->add_option_group("Deconvolution CLI", "Deconvolution commandline options");
    addParameters(deconvolutionConfig, deconvCliGroup);
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
    configGroup = config_group;
}


bool CLIFrontend::groupHasOptions(CLI::Option_group* group) const {
    if (!group) return false;
    for (const auto& opt : group->get_options()) {
        if (opt && opt->count() > 0) return true;
    }
    for (const auto& sub : group->get_subcommands()) {
        for (const auto& opt : sub->get_options()) {
            if (opt && opt->count() > 0) return true;
        }
    }
    return false;
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
            if (std::string(param.cliFlag) == "--psf_file_paths" || std::string(param.cliFlag) == "--multiple_psf_config_paths" || std::string(param.cliFlag) == "--psf_config_path"){
                if (!psfPathGroup) {
                    psfPathGroup = group->add_option_group("PSF Path", "PSF file path options (at least one required for CLI mode)");
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


PSFGenerationRequest CLIFrontend::generatePSFRequest(std::shared_ptr<SetupConfigPSF> setupConfig){
    PSFGenerationRequest request(setupConfig, loggingCallback, progressVisualization);
    return request;
}

DeconvolutionRequest CLIFrontend::generateDeconvRequest(std::shared_ptr<SetupConfig> setupConfigCopy, std::shared_ptr<DeconvolutionConfig> deconvConfigCopy) {
    DeconvolutionRequest request(setupConfigCopy, deconvConfigCopy, loggingCallback, progressVisualization);
    if (!inlinePsfConfigs.empty()) {
        request.setInlinePSFConfigs(inlinePsfConfigs);
    }
    return request;
}
