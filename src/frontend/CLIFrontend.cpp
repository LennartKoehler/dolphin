#include "frontend/CLIFrontend.h"
#include <sys/stat.h>
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
        dolphin->generatePSF(setupConfig.psfConfigPath);
    }
    else if (*deconvolutionCLI) {
        handleDeconvolution();
        dolphin->deconvolve(std::make_shared<SetupConfig>(setupConfig));
    }
    else {
        std::cerr << "[ERROR] No subcommand selected" << std::endl;
        std::cout << app.help() << std::endl;
    }
}

void CLIFrontend::psfgenerator() {
    // Define PSF generator options
    CLI::Option_group* psf_group = psfCLI->add_option_group("PSF Options", "PSF generation options");
    psf_group->add_option("-p", setupConfig.psfConfigPath, "Input PSF Config file")->required();
}

void CLIFrontend::deconvolution() {
    // Define deconvolution options (but don't parse here)
    readCLISetupConfigPath();    
    readCLIParameters();         
    readCLIParametersPSF();      
    readCLIParametersDeconvolution(); 
    // Remove parseCLI() call from here!
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




void CLIFrontend::readCLIParameters() {
    cli_group = deconvolutionCLI->add_option_group("CLI", "Commandline options");
    
    // Remove ->required() - CLI11 will only check this if deconvolution subcommand is used
    cli_group->add_option("-i,--image", setupConfig.imagePath, "Input image Path")->required();
    
    // Optional parameters
    cli_group->add_option("--gpu", setupConfig.gpu, "Type of GPU API ('cuda'/'none')");
    cli_group->add_flag("--savepsf", setupConfig.savePsf, "Save used PSF");
    cli_group->add_flag("--time", setupConfig.time, "Show duration active");
    cli_group->add_flag("--seperate", setupConfig.sep, "Save as TIF directory, each layer as single file");
    cli_group->add_flag("--info", setupConfig.printInfo, "Prints info about input Image");
    cli_group->add_flag("--showExampleLayers", setupConfig.showExampleLayers, "Shows a layer of loaded image and PSF)");
    cli_group->add_flag("--saveSubimages", setupConfig.saveSubimages, "Saves subimages seperate as file");
    
    // Set up exclusions
    if (configGroup) {
        cli_group->excludes(configGroup);
        configGroup->excludes(cli_group);
    }
}

void CLIFrontend::readCLIParametersDeconvolution() {
    // Remove ->required() - CLI11 handles this automatically for subcommands
    cli_group->add_option("-a,--algorithm", deconvolutionConfig.algorithmName, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    
    // Rest of options...
    cli_group->add_option("--epsilon", deconvolutionConfig.epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", deconvolutionConfig.iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", deconvolutionConfig.lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");
    cli_group->add_option("--borderType", deconvolutionConfig.borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", deconvolutionConfig.psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--subimageSize", deconvolutionConfig.subimageSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);
    cli_group->add_flag("--grid", deconvolutionConfig.grid, "Image divided into sub-image cubes (grid)");
}

void CLIFrontend::readCLISetupConfigPath() {
    CLI::Option_group *config_group = deconvolutionCLI->add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", setupConfigPath, "Path to configuration file");
    
    // DON'T exclude here yet - cli_group doesn't exist
    // Store the group for later exclusion
    configGroup = config_group;  // Add configGroup as member variable
}


void CLIFrontend::readCLIParametersPSF(){

    cli_group->add_option("-p,--psf", setupConfig.psfFilePath, "Input PSF path(s) or 'synthetic'");
    cli_group->add_option("--psfDirectory", setupConfig.psfDirPath, "Input PSF path(s) or 'synthetic'");
    cli_group->add_option("--psfConfig", setupConfig.psfConfigPath, "Input PSF Config file");


}

