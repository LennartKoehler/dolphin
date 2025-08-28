#include "frontend/CLIFrontend.h"
#include <sys/stat.h>


CLIFrontend::CLIFrontend(ConfigManager* config, int argc, char** argv)
    : IFrontend(config){
        init(argc, argv);
    }

void CLIFrontend::init(int argc, char** argv){
    this->argc = argc;
    this->argv = argv;
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

void CLIFrontend::run(){
    bool error = parseCLI();
    readCLIParameters();
    readCLIParametersPSF();
    
    config->setCuda();
    
    handleCLIConfigs();

    if (!configFilePath.empty()) {
        config->handleJSONConfigs(configFilePath);
    }
}


void CLIFrontend::readCLIParametersPSF(){
    cli_group->add_option("-p,--psf", psfPaths, "Input PSF path(s) or 'synthetic'")
    ->required()
    ->expected(-1) // Allows multiple values
    ->check([](const std::string& path) {
        if (path == "synthetic") {
            return std::string(); // "synthetic" is valid
        }
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            return "Path does not exist: " + path;
        }
        if (!(info.st_mode & S_IFDIR) && !(info.st_mode & S_IFREG)) {
            return "Path is neither a file nor a directory: " + path;
        }
        return std::string(); // Valid
    });
    // Define a group for configuration file
    CLI::Option_group *config_group = app.add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", configFilePath, "Path to configuration file")->required();


    // Exclude CLI arguments if configuration file is set
    cli_group->excludes(config_group);
    config_group->excludes(cli_group);
}

void CLIFrontend::readCLIParameters(){
    cli_group = app.add_option_group("CLI", "Commandline options");
    cli_group->add_option("-i,--image", config->image_path, "Input image Path")->required();
    // Example: ./deconvtool -p psf1.json psf2.json

    cli_group->add_option("-a,--algorithm", config->algorithmName, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    cli_group->add_option("--epsilon", config->epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", config->iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", config->lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", config->borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", config->psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--subimageSize", config->subimageSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);

    cli_group->add_option("--gpu", config->gpu, "Type of GPU API ('cuda'/'none')");

    cli_group->add_flag("--savepsf", config->savePsf, "Save used PSF");
    cli_group->add_flag("--time", config->time, "Show duration active");
    cli_group->add_flag("--grid", config->grid, "Image divided into sub-image cubes (grid)");
    cli_group->add_flag("--seperate", config->sep, "Save as TIF directory, each layer as single file");
    cli_group->add_flag("--info", config->printInfo, "Prints info about input Image");
    cli_group->add_flag("--showExampleLayers", config->showExampleLayers, "Shows a layer of loaded image and PSF)");
    cli_group->add_flag("--saveSubimages", config->saveSubimages, "Saves subimages seperate as file");



}

void CLIFrontend::handleCLIConfigs() {
    for (const auto& path : psfPaths) {
        config->processSinglePSFPath(path);
    }
}


