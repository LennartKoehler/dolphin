#include "frontend/CLIFrontend.h"
#include <sys/stat.h>


CLIFrontend::CLIFrontend(SetupConfig* config, int argc, char** argv)
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
    readCLIParametersDeconvolution();
}


void CLIFrontend::readCLIParametersPSF(){
    cli_group->add_option("-p,--psf", setupConfig.psfFilePath, "Input PSF path(s) or 'synthetic'");
    cli_group->add_option("--psfDirectory", setupConfig.psfDirPath, "Input PSF path(s) or 'synthetic'");
    cli_group->add_option("-c,--config", setupConfig.psfConfigPath, "Path to configuration file")->required();

    // CLI::Option_group *config_group = app.add_option_group("Config", "Configuration file");


    // // Exclude CLI arguments if configuration file is set
    // cli_group->excludes(config_group);
    // config_group->excludes(cli_group);
}

void CLIFrontend::readCLIParameters(){
    cli_group = app.add_option_group("CLI", "Commandline options");
    cli_group->add_option("-i,--image", setupConfig.imagePath, "Input image Path")->required();


    cli_group->add_option("--gpu", setupConfig.gpu, "Type of GPU API ('cuda'/'none')");

    cli_group->add_flag("--savepsf", setupConfig.savePsf, "Save used PSF");
    cli_group->add_flag("--time", setupConfig.time, "Show duration active");
    cli_group->add_flag("--seperate", setupConfig.sep, "Save as TIF directory, each layer as single file");
    cli_group->add_flag("--info", setupConfig.printInfo, "Prints info about input Image");
    cli_group->add_flag("--showExampleLayers", setupConfig.showExampleLayers, "Shows a layer of loaded image and PSF)");
    cli_group->add_flag("--saveSubimages", setupConfig.saveSubimages, "Saves subimages seperate as file");


}

void CLIFrontend::readCLIParametersDeconvolution(){

    cli_group->add_option("-a,--algorithm", deconvolutionConfig.algorithmName, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    cli_group->add_option("--epsilon", deconvolutionConfig.epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", deconvolutionConfig.iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", deconvolutionConfig.lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", deconvolutionConfig.borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", deconvolutionConfig.psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--subimageSize", deconvolutionConfig.subimageSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);
    cli_group->add_flag("--grid", deconvolutionConfig.grid, "Image divided into sub-image cubes (grid)");
}


