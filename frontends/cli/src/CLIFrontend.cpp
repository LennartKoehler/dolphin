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
#include <sys/stat.h>
#include <cstring>
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
    catch (const CLI::ParseError& e) {
        // CLI11 throws ParseError (and subclasses like RequiredError) for missing required options,
        // validation failures, etc. Print the error and return false so the caller knows parsing failed.
        spdlog::error("{}", e.what());
        spdlog::info("{}", app.help());
        return false;
    }
    catch (const std::exception& e) {
        spdlog::error("{}", e.what());
        spdlog::info("{}", app.help());
        return false;
    }
}



void CLIFrontend::run() {
    // 1. Define ALL subcommands and their options FIRST
    psfgenerator();      // Define PSF options
    deconvolution();     // Define deconvolution options (but don't parse yet)

    // 2. Parse once to determine which subcommand was used
    bool success = parseCLI();
    if (!success) {
        return;
    }
    setupConfig.printValues();

    // 3. Handle based on which subcommand was selected
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
        spdlog::error("No subcommand selected");
        spdlog::info("{}", app.help());
    }
}

void CLIFrontend::psfgenerator() {
    CLI::Option_group* psf_group = psfCLI->add_option_group("PSF Options", "PSF generation options");
    psf_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");

    psfconfigGroup = psf_group;  // Add configGroup as member variable
    psfcli_group = psfCLI->add_option_group("CLI", "PSF Commandline options");

    addParameters(psfConfig, psfcli_group);
    // Set up exclusions
    if (psfconfigGroup) {
        psfcli_group->excludes(psfconfigGroup);
        psfconfigGroup->excludes(psfcli_group);
    }

}

void CLIFrontend::deconvolution() {
    // Define deconvolution options (but don't parse here)
    readCLISetupConfigPath();
    readSetupConfigParameters();
    readCLIParametersDeconvolution();
}

// New helper methods
bool CLIFrontend::handlePSFGeneration() {
    // overwrite the default of which params are required in cli.
    // the default is for deconvolutionconfig

    std::vector<std::string> missingParams = checkRequired(psfConfig);
    if (!missingParams.empty()) {
        spdlog::error("Required parameter(s) missing:");
        for (const auto& p : missingParams) {
            spdlog::error("  - {}", p);
        }
        spdlog::info("{}", psfCLI->help());
        return false;
    }

    return true;
}

bool CLIFrontend::readDeconvolutionFromConfigFile() {

    // Handle configuration loading from file
    // Either a config file is provided (--config) or individual CLI parameters are used (mutual exclusion)
    if (!setupConfigPath.empty()) {
        try {
            setupConfig = SetupConfig::createFromJSONFile(setupConfigPath);

            deconvolutionConfig = DeconvolutionConfig::createFromJSONFile(setupConfigPath);

            spdlog::info("Configuration loaded from: {}", setupConfigPath);
        } catch (const std::exception& e) {
            spdlog::error("{}", e.what());
            return false;
        }
    }
    return true;
}

bool CLIFrontend::readPSFFromConfigFile() {

    // Handle configuration loading from file
    // Either a config file is provided (--config) or individual CLI parameters are used (mutual exclusion)
    if (!setupConfigPath.empty()) {
        try {
            psfConfig = SetupConfigPSF::createFromJSONFile(setupConfigPath);

            spdlog::info("Configuration loaded from: {}", setupConfigPath);
        } catch (const std::exception& e) {
            spdlog::error("{}", e.what());
            return false;
        }
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
    cli_group = deconvolutionCLI->add_option_group("CLI", "Deconvolution Commandline options");

    addParameters(setupConfig, cli_group);
    // Set up exclusions
    if (configGroup) {
        cli_group->excludes(configGroup);
        configGroup->excludes(cli_group);
    }
}

void CLIFrontend::readCLIParametersDeconvolution() {
    // Use visitParams to iterate through all deconvolution parameters and create CLI options
    addParameters(deconvolutionConfig, cli_group);
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
    configGroup = config_group;  // Add configGroup as member variable
}




void CLIFrontend::addParameters(Config& config, CLI::Option_group* group){

    config.visitParams([this, group]<typename T>(T& value, ConfigParameter& param){

        if constexpr (std::is_same_v<T, std::array<int, 3>>){
            // Skip: std::array<int,3> not directly supported as a CLI option
            return;
        }
        else {
            if constexpr (std::is_same_v<T, bool>){
                auto opt = group->add_flag(param.cliFlag, value, param.cliDesc);
                opt->configurable(false);  // Avoid cross-subcommand name collision checks in CLI11
                return;
            }
            if (std::string(param.cliFlag) == "--psf_file_paths" || std::string(param.cliFlag) == "--multiple_psf_config_paths" || std::string(param.cliFlag) == "--psf_config_path"){
                static CLI::Option_group* group=app.add_option_group("subgroup");
                group->excludes(configGroup);
                configGroup->excludes(group);
                group->add_flag(param.cliFlag, value, param.cliDesc)->ignore_case();
                group->require_option(1);
                return;
            }
            auto opt = group->add_option(param.cliFlag, value, param.cliDesc);
            opt->configurable(false);  // Avoid cross-subcommand name collision checks in CLI11
            // NOTE: Don't use opt->required() here — CLI11 throws on the FIRST missing
            // required option, preventing us from reporting ALL missing parameters at once.
            // Instead, required parameters are validated manually in handleDeconvolution().
        }
    });
}


std::vector<std::string> CLIFrontend::checkRequired(Config& config) const {
    // Validate that ALL required parameters are present.
    // This is necessary because:
    //   1. We don't use CLI11's ->required() (it throws on the first missing option,
    //      preventing us from reporting all missing parameters at once).
    //   2. When --config is used, the CLI group is excluded, bypassing CLI11's
    //      required() enforcement entirely.
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
        // Numeric types (int, float) have default values and can't be "empty",
        // so they are not marked cliRequired in practice. If that changes,
        // add checks here.
    };

    config.visitParams(checkRequired);
    return missingParams;
}


void progressVisualization(std::atomic<float>& current, float max){
    // Calculate progress

    float barWidth = 50;
    int pos = static_cast<int>((current * barWidth) / max);
    int progress = static_cast<int>((current * 100) / max);
    // Print progress bar
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
    return request;
}
