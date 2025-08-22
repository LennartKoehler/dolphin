#include "ConfigManager.h"
#include <sys/stat.h>







void ConfigManager::setCuda(){
#ifdef CUDA_AVAILABLE
    if(gpu == "") {
        gpu = "cuda";
        std::cout << "[INFO] CUDA activated, to deactivated use --gpu none (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }else if(gpu != "cuda") {
        std::cout << "[WARNING] --gpu set to "<< gpu <<". CUDA is available, but not activated. Use --gpu cuda (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }
#endif
}

bool ConfigManager::handleInput(int argc, char** argv){
    try{
        setCLIOptions();
        CLI11_PARSE(app, argc, argv);
        setCuda();
        handleCLIConfigs();

        if (!config_file_path.empty()) {
            handleJSONConfigs(config_file_path);
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return EXIT_FAILURE;
    }

}

void ConfigManager::handleJSONConfigs(const std::string& configPath) {
    this->config = loadJSONFile(configPath);
    image_path = extractImagePath(this->config);
    processPSFPaths();
    extractAlgorithmParameters();
    extractOptionalParameters();
}

json ConfigManager::loadJSONFile(const std::string& filePath) const {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open configuration file: " + filePath);
    }
    
    std::cout << "[STATUS] " << filePath << " successfully read" << std::endl;

    json jsonFile;
    file >> jsonFile;
    return jsonFile;
}

std::string ConfigManager::extractImagePath(const json& config) const {
    return config["image_path"].get<std::string>();
}

void ConfigManager::processPSFPaths() {
    if (!config.contains("psf_path")) {
        return;
    }
    
    if (config["psf_path"].is_string()) {
        std::string psf_path = config["psf_path"].get<std::string>();
        processSinglePSFPath(psf_path);
    } else if (config["psf_path"].is_array()) {
        processPSFPathArray();
    } else {
        throw std::runtime_error("Field 'psf_path' has invalid format.");
    }
}

void ConfigManager::processSinglePSFPath(const std::string& psf_path) {
    
    if (isJSONFile(psf_path)) {
        json configJson = loadJSONFile(psf_path);
        if (configJson.contains("path") && configJson["path"].get<std::string>() != "") {
            psfPaths.push_back(configJson["path"].get<std::string>());
        }
        else{
            psfJSON.push_back(configJson);
        }
    } else {
        // if (!psf_path.empty()) { // LK why do i need this? only add psf_paths if theyre not empty?
        psfPaths.push_back(psf_path);
        // }
    }
}

void ConfigManager::processPSFPathArray() {
    for (const auto& element : config["psf_path"]) {
        if (element.is_string()) {
            std::string elementStr = element.get<std::string>();
            processSinglePSFPath(elementStr);
        }
    }
}

bool ConfigManager::isJSONFile(const std::string& path) {
    return path.substr(path.find_last_of(".") + 1) == "json";
}

void ConfigManager::handleCLIConfigs() {
    for (const auto& path : psfPathsCLI) {
        processSinglePSFPath(path);
    }
}


void ConfigManager::extractAlgorithmParameters() {
    algorithmName = config["algorithm"].get<std::string>();
    epsilon = config["epsilon"].get<double>();
    iterations = config["iterations"].get<int>();
    lambda = config["lambda"].get<double>();
    psfSafetyBorder = config["psfSafetyBorder"].get<int>();
    subimageSize = config["subimageSize"].get<int>();
    borderType = config["borderType"].get<int>();
}

void ConfigManager::extractOptionalParameters() {
    sep = config["seperate"].get<bool>();
    time = config["time"].get<bool>();
    savePsf = config["savePsf"].get<bool>();
    showExampleLayers = config["showExampleLayers"].get<bool>();
    printInfo = config["info"].get<bool>();
    grid = config["grid"].get<bool>();
    
    if (config.contains("saveSubimages")) {
        saveSubimages = config["saveSubimages"].get<bool>();
    }
    if (config.contains("gpu")) {
        gpu = config["gpu"].get<std::string>();
    }
}


void ConfigManager::setCLIOptions(){
    cli_group = app.add_option_group("CLI", "Commandline options");
    cli_group->add_option("-i,--image", image_path, "Input image Path")->required();
    // Example: ./deconvtool -p psf1.json psf2.json
    cli_group->add_option("-p,--psf", psfPathsCLI, "Input PSF path(s) or 'synthetic'")
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
    cli_group->add_option("-a,--algorithm", algorithmName, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    cli_group->add_option("--epsilon", epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--subimageSize", subimageSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);

    cli_group->add_option("--gpu", gpu, "Type of GPU API ('cuda'/'none')");

    cli_group->add_flag("--savepsf", savePsf, "Save used PSF");
    cli_group->add_flag("--time", time, "Show duration active");
    cli_group->add_flag("--grid", grid, "Image divided into sub-image cubes (grid)");
    cli_group->add_flag("--seperate", sep, "Save as TIF directory, each layer as single file");
    cli_group->add_flag("--info", printInfo, "Prints info about input Image");
    cli_group->add_flag("--showExampleLayers", showExampleLayers, "Shows a layer of loaded image and PSF)");
    cli_group->add_flag("--saveSubimages", saveSubimages, "Saves subimages seperate as file");

    // Define a group for configuration file
    CLI::Option_group *config_group = app.add_option_group("Config", "Configuration file");
    config_group->add_option("-c,--config", config_file_path, "Path to configuration file")->required();

    // Exclude CLI arguments if configuration file is set
    cli_group->excludes(config_group);
    config_group->excludes(cli_group);

}

