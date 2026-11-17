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
#include <dolphin/deconvolution/AlgorithmNames.h>
#include <dolphin/deconvolution/DeconvolutionAlgorithmFactory.h>

CLIFrontend::CLIFrontend(Dolphin* dolphin, int argc, char** argv)
    : IFrontend(dolphin){
        this->argc = argc;
        this->argv = argv;
        cliAlgorithmName = prescanAlgorithmName(argc, argv);
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
        spdlog::error("Required parameter(s) missing:");
        for (const auto& p : missingParams) {
            spdlog::error("  - {}", p);
        }
        spdlog::info("Provide parameters via CLI or use -c/--config, -s/--setup_config, -p/--psf_configs for JSON config files");
        spdlog::info("{}", psfCLI->help());
        return false;
    }

    if (!bundle.hasPSF) {
        spdlog::error("No PSF config provided — use -p/--psf_configs or inline psf_configs in JSON");
        spdlog::info("{}", psfCLI->help());
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
            spdlog::info("Setup config loaded from JSON — CLI setup args ignored");
        }
        merged.setupConfig = jsonBundle.setupConfig;
    } else {
        merged.setupConfig = cliBundle.setupConfig;
    }
    merged.hasSetup = true;

    if (jsonBundle.hasDeconv) {
        if (cliBundle.hasDeconv) {
            spdlog::info("Deconvolution config loaded from JSON — CLI deconvolution args ignored");
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
            spdlog::info("Setup config loaded from JSON — CLI setup args ignored");
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
    std::vector<std::string> missingParams;
    if (bundle.hasDeconv && bundle.deconvConfig) {
        missingParams = checkRequired(*bundle.deconvConfig);
    }
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
    // cliAlgorithmName was resolved in the constructor's prescan: a config file
    // (-c/-d) wins over --algorithm_name, matching mergeBundles(). The options
    // below bind to the config class of that algorithm.
    addAlgorithmSelectionOption();
    cliBundle.deconvConfig = DeconvolutionAlgorithmFactory::getInstance()
                                 .createConfigForAlgorithm(cliAlgorithmName);
    deconvCliGroup = deconvolutionCLI->add_option_group("Deconvolution CLI", "Deconvolution commandline options");
    addParameters(*cliBundle.deconvConfig, deconvCliGroup);
    cliBundle.hasDeconv = true;
}

std::string CLIFrontend::prescanAlgorithmName(int argc, char** argv) {
    // The deconvolution options bind to the config class of the selected
    // algorithm, but the CLI is built before anything is parsed. Peek directly at
    // argv so the option tree already matches the selection. A config file (-c/-d)
    // wins over --algorithm_name, matching mergeBundles().
    const std::string algorithmFlag = "--algorithm_name";
    const std::vector<std::string> configFlags = {"-d", "--deconv_config", "-c", "--config"};

    std::string algorithmFromCLI;
    std::string algorithmFromFile;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasNext = i + 1 < argc;

        if (arg == algorithmFlag && hasNext) {
            algorithmFromCLI = argv[i + 1];
        }
        else if (arg.rfind(algorithmFlag + "=", 0) == 0) {
            algorithmFromCLI = arg.substr(algorithmFlag.size() + 1);
        }

        for (const auto& flag : configFlags) {
            const bool matches = (arg == flag && hasNext) || arg.rfind(flag + "=", 0) == 0;
            if (!matches) continue;
            const std::string path = (arg == flag) ? argv[i + 1] : arg.substr(flag.size() + 1);
            algorithmFromFile = peekAlgorithmName(path);
            break;
        }
    }

    if (!algorithmFromFile.empty()) return algorithmFromFile;
    if (!algorithmFromCLI.empty()) return algorithmFromCLI;
    return DeconvolutionAlgorithms::kRichardsonLucy;
}

std::string CLIFrontend::peekAlgorithmName(const std::string& path) {
    try {
        json jsonData = Config::loadJSONFile(path);
        const json& data = jsonData.contains("deconvolution_config")
                               ? jsonData.at("deconvolution_config")
                               : jsonData;
        if (data.is_object() && data.contains("algorithm_name")) {
            return data.at("algorithm_name").get<std::string>();
        }
    }
    catch (const std::exception& e) {
        // The real load reports the problem; fall back to the default here.
        spdlog::debug("Could not read algorithm_name from {}: {}", path, e.what());
    }
    return {};
}

void CLIFrontend::addAlgorithmSelectionOption() {
    // Bound to cliAlgorithmName (not to a config member): the config class is
    // created after parsing, from the selected name (see generateDeconvRequest).
    // Group names may not contain spaces (CLI11 construction rule).
    auto* group = deconvolutionCLI->add_option_group("Algorithm", "Algorithm selection");
    group->add_option("--algorithm_name", cliAlgorithmName,
                      "Deconvolution algorithm; determines which options are available")
        ->check(CLI::IsMember(DeconvolutionAlgorithms::names()))
        ->configurable(false);
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
        else if (std::string(param.cliFlag) == "--algorithm_name") {
            // Already offered by addAlgorithmSelectionOption(); it decides the
            // config class, so it cannot be bound to a config member.
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
                auto opt = group->add_option(param.cliFlag, value, param.cliDesc);
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

void CLIFrontend::loadDeconvConfigFromJSON(const json& jsonData, std::shared_ptr<DeconvolutionConfig>& config) {
    if (!jsonData.contains("deconvolution_config")) {
        return;
    }
    try {
        // Creates the config class matching the JSON's algorithm_name.
        config = DeconvolutionConfig::createFromJSON(jsonData["deconvolution_config"]);
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to parse deconvolution config: ") + e.what());
    }
}

std::vector<std::shared_ptr<PSFConfig>> CLIFrontend::loadPSFConfigsFromJSON(const json& jsonData) {
    std::vector<std::shared_ptr<PSFConfig>> configs;
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();

    if (jsonData.contains("psf_configs")) {
        const auto& psfField = jsonData["psf_configs"];
        if (psfField.is_array()) {
            for (const auto& psfJson : psfField) {
                if (!psfJson.is_object()) {
                    throw std::runtime_error(
                        "Each entry in 'psf_configs' array must be a JSON object with 'model_name'. "
                        "Found: " + psfJson.dump() + " (type: " + std::string(psfField.type_name()) + ")"
                    );
                }
                configs.push_back(factory.createConfig(psfJson));
            }
        } else if (psfField.is_object()) {
            configs.push_back(factory.createConfig(psfField));
        } else {
            throw std::runtime_error(
                "'psf_configs' must be a JSON array of PSF config objects or a single PSF config object. "
                "Found type: " + std::string(psfField.type_name()) + " (value: " + psfField.dump() + ")"
            );
        }
    } else if (jsonData.contains("psf_config")) {
        const auto& psfField = jsonData["psf_config"];
        if (!psfField.is_object()) {
            throw std::runtime_error(
                "'psf_config' must be a JSON object with 'model_name'. "
                "Found type: " + std::string(psfField.type_name()) + " (value: " + psfField.dump() + ")"
            );
        }
        configs.push_back(factory.createConfig(psfField));
    } else if (jsonData.contains("model_name")) {
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

void CLIFrontend::loadDeconvConfigFromFile(const std::string& path, std::shared_ptr<DeconvolutionConfig>& config) {
    json jsonData = Config::loadJSONFile(path);
    try {
        config = DeconvolutionConfig::createFromJSON(jsonData);
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to parse deconvolution config file {}: {}", path, e.what());
        throw;
    }
}

std::vector<std::shared_ptr<PSFConfig>> CLIFrontend::loadPSFConfigsFromFile(const std::string& path) {
    json jsonData = Config::loadJSONFile(path);
    auto configs = loadPSFConfigsFromJSON(jsonData);
    if (configs.empty()) {
        throw std::runtime_error(
            "No PSF config found in file: " + path + "\n"
            "Expected one of:\n"
            "  - 'psf_configs' as a JSON array of PSF config objects (each with 'model_name')\n"
            "  - 'psf_configs' as a single PSF config object (with 'model_name')\n"
            "  - 'psf_config' as a single PSF config object (with 'model_name')\n"
            "  - A root-level 'model_name' field"
        );
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
    if (bundle.hasPSF) {
        setupConfig->psfConfigs = bundle.psfConfigs;
    }
    std::shared_ptr<DeconvolutionConfig> deconvConfig = bundle.deconvConfig
        ? bundle.deconvConfig
        : DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm(cliAlgorithmName);
    DeconvolutionRequest request(setupConfig, std::move(deconvConfig), loggingCallback, progressVisualization);
    return request;
}
