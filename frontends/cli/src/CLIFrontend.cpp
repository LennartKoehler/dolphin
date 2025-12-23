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
#include <sys/stat.h>
#include <type_traits>
#include <cstring>
#include "Dolphin.h"

CLIFrontend::CLIFrontend(Dolphin* dolphin, int argc, char** argv)
    : IFrontend(dolphin){
        this->argc = argc;
        this->argv = argv;
        psfCLI = app.add_subcommand("psfgenerator", "Generate PSF file");
        deconvolutionCLI = app.add_subcommand("deconvolution", "Run deconvolution");
    }


bool CLIFrontend::parseCLI(){
    try{
        CLI11_PARSE(app, argc, argv);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
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
    
    // 3. Handle based on which subcommand was selected
    if (*psfCLI) {
        handlePSFGeneration();
        PSFGenerationRequest request = generatePSFRequest(setupConfig.psfConfigPath);
        dolphin->generatePSF(request);
    }
    else if (*deconvolutionCLI) {
        handleDeconvolution();
        DeconvolutionRequest request = generateDeconvRequest(std::make_shared<SetupConfig>(setupConfig));
        dolphin->deconvolve(request);
    }
    else {
        std::cerr << "[ERROR] No subcommand selected" << std::endl;
        std::cout << app.help() << std::endl;
    }
}

void CLIFrontend::psfgenerator() {
    // Define PSF generator options
    CLI::Option_group* psf_group = psfCLI->add_option_group("PSF Options", "PSF generation options"); // TESTVALUE uncomment
    psf_group->add_option("-p", setupConfig.psfConfigPath, "Input PSF Config file")->required();
    psf_group->add_option("-d", setupConfig.outputDir, "Output directory"); // TODO change
}

void CLIFrontend::deconvolution() {
    // Define deconvolution options (but don't parse here)
    readCLISetupConfigPath();
    readSetupConfigParameters();
    // readCLIParametersPSF();
    readCLIParametersDeconvolution();

}

// New helper methods
void CLIFrontend::handlePSFGeneration() {
    std::cout << "[INFO] PSF generation mode selected" << std::endl;
    // PSF-specific processing
    if (setupConfig.psfConfigPath.empty()) {
        std::cerr << "[ERROR] PSF config path is required for PSF generation" << std::endl;
        return;
    }
}

void CLIFrontend::handleDeconvolution() {
    std::cout << "[INFO] Deconvolution mode selected" << std::endl;
    
    // Handle configuration loading
    if (!setupConfigPath.empty()) {
        try {
            setupConfig = SetupConfig::createFromJSONFile(setupConfigPath);
            std::cout << "[INFO] Configuration loaded from: " << setupConfigPath << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << std::endl;
            return;
        }
    } else {
        // CLI was used, copy deconvolution config
        setupConfig.deconvolutionConfig = std::make_shared<DeconvolutionConfig>(deconvolutionConfig);
    }
}




void CLIFrontend::readSetupConfigParameters() {
    cli_group = deconvolutionCLI->add_option_group("CLI", "Commandline options");
   
    setupConfig.visitParams([this]<typename T>(T& value, ConfigParameter& param){
        if (param.type == ParameterType::Bool){
            cli_group->add_flag(param.cliFlag, value, param.cliDesc);
        }
        else{
            auto opt = cli_group->add_option(param.cliFlag, value, param.cliDesc);
                // Mark as required if specified
            if (param.cliRequired) {
                opt->required();
            }
            
            // Apply positive number check for parameters with min values >= 0
            if (param.hasRange && param.minVal >= 0.0) {
                opt->check(CLI::PositiveNumber);
            }
        }
    });
    // // Remove ->required() - CLI11 will only check this if deconvolution subcommand is used
    // cli_group->add_option("-i,--image", setupConfig.imagePath, "Input image Path")->required();
    
    // // Optional parameters
    // cli_group->add_option("-d", setupConfig.outputDir, "Output directory");
    // cli_group->add_option("--backend", setupConfig.backend, "Type of Backend ('cuda'/'cpu')");
    // cli_group->add_flag("--savepsf", setupConfig.savePsf, "Save used PSF");
    // cli_group->add_flag("--time", setupConfig.time, "Show duration active");
    // cli_group->add_flag("--seperate", setupConfig.sep, "Save as TIF directory, each layer as single file");
    // cli_group->add_flag("--info", setupConfig.printInfo, "Prints info about input Image");
    // cli_group->add_flag("--showExampleLayers", setupConfig.showExampleLayers, "Shows a layer of loaded image and PSF)");
    // cli_group->add_flag("--saveSubimages", setupConfig.saveSubimages, "Saves subimages seperate as file");
    
    // Set up exclusions
    if (configGroup) {
        cli_group->excludes(configGroup);
        configGroup->excludes(cli_group);
    }
}

void CLIFrontend::readCLIParametersDeconvolution() {
    // Use visitParams to iterate through all deconvolution parameters and create CLI options
    deconvolutionConfig.visitParams([this]<typename T>(T& value, ConfigParameter& param) {
        if (param.type == ParameterType::Bool){
            cli_group->add_flag(param.cliFlag, value, param.cliDesc);
        }
        else{
            auto opt = cli_group->add_option(param.cliFlag, value, param.cliDesc);
                // Mark as required if specified
            if (param.cliRequired) {
                opt->required();
            }
            
            // Apply positive number check for parameters with min values >= 0
            if (param.hasRange && param.minVal >= 0.0) {
                opt->check(CLI::PositiveNumber);
            }
        }
    });
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
    
    // DON'T exclude here yet - cli_group doesn't exist
    // Store the group for later exclusion
    configGroup = config_group;  // Add configGroup as member variable
}






PSFGenerationRequest CLIFrontend::generatePSFRequest(const std::string& configPath){
    PSFGenerationRequest request(configPath);
    
    // Set CLI-specific options
    request.save_result = true;  // CLI typically wants to save results
    request.output_path = setupConfig.outputDir;
    return request;    
}

DeconvolutionRequest CLIFrontend::generateDeconvRequest(std::shared_ptr<SetupConfig> setupConfigCopy) {
    // Create request with setup config
    DeconvolutionRequest request(setupConfigCopy);
    
    // Set CLI-specific options from parsed arguments
    request.save_separate = setupConfigCopy->sep;
    request.save_subimages = setupConfigCopy->saveSubimages;
    request.show_example = setupConfigCopy->showExampleLayers;
    request.print_info = setupConfigCopy->printInfo;
    request.output_path = setupConfigCopy->outputDir;
    return request;
}