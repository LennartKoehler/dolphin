#include "Dolphin.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "ConvolutionAlgorithm.h"
#include "DeconvolutionAlgorithm.h"

#include <sys/stat.h>
#ifdef CUDA_AVAILABLE
#include "../lib/cube/include/CUBE.h"
#endif





Dolphin::Dolphin(){

}

Dolphin::~Dolphin(){

}

bool Dolphin::init(int argc, char** argv){
    std::cout << "[Start DeconvTool]" << std::endl;
    PSFPackage psfPackage;
    try{
        setCLIOptions();
        CLI11_PARSE(app, argc, argv);
        setCuda();
        handleConfigs();
        psfPackage = initPSF();
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    psfs = psfPackage.psfs;
    psfCubeVec = psfPackage.psfCubeVec;
    psfLayerVec = psfPackage.psfLayerVec;

    inputHyperstack = initHyperstack();
    algorithm = algorithmFactory(algorithmName);
    return 0;

}

std::unique_ptr<BaseAlgorithm> Dolphin::algorithmFactory(const std::string& algorithmName){
    if (algorithmName == "convolve"){
        return std::make_unique<ConvolutionAlgorithm>();
    }
    else{
        return initDeconvolution(psfCubeVec, psfLayerVec);
    }
}

void Dolphin::setCLIOptions(){
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

    cli_group->add_flag("--savepsf", time, "Save used PSF");
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

void Dolphin::setCuda(){
#ifdef CUDA_AVAILABLE
    if(gpu == "") {
        gpu = "cuda";
        std::cout << "[INFO] CUDA activated, to deactivated use --gpu none (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }else if(gpu != "cuda") {
        std::cout << "[WARNING] --gpu set to "<< gpu <<". CUDA is available, but not activated. Use --gpu cuda (CPU parallelism is deactivated for deconvtoolcuda)" << std::endl;
    }
#endif
}

void Dolphin::handleConfigs(){
    if (!config_file_path.empty()) {
        // Read configuration file
        std::ifstream config_file(config_file_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Failed to open configuration file: " + config_file_path);
        }

        std::cout<< "[STATUS] " << config_file_path << " successfully read" << std::endl;
        config_file >> config;
        // Values from configuration file passed to arguments
        image_path = config["image_path"].get<std::string>();
        if (config.contains("psf_path")) {
            if (config["psf_path"].is_string()) {
                psf_path = config["psf_path"].get<std::string>();
                if (psf_path.substr(psf_path.find_last_of(".") + 1) == "json") {
                    //psfconfig objekt machen
                    PSFConfig psf_config;
                    if( psf_config.loadFromJSON(psf_path)) {
                        psf_config.printValues();
                    }else {
                        throw std::runtime_error("psf_foncig.loadFromJSON failed");
                    }
                    psfConfigs.push_back(psf_config);

                } else {
                    if(psf_path != "" || !psf_path.empty()) {
                        psfPaths.push_back(psf_path);
                    }
                }
            } else if (config["psf_path"].is_array()) {
                for (const auto& element : config["psf_path"]) { // Range-based for loop
                    if (element.is_string()) {
                        //entweder config
                        std::string elementStr = element.get<std::string>();
                        // Überprüfe die Dateiendung
                        if (elementStr.substr(elementStr.find_last_of(".") + 1) == "json") {
                            //psfconfig objekt machen
                            PSFConfig psf_config;
                            if( psf_config.loadFromJSON(elementStr)) {
                                psf_config.printValues();
                            }else {
                                throw std::runtime_error("psf_config.loagFromJSON failed");
                            }
                            psfConfigs.push_back(psf_config);

                        } else {
                            psfPaths.push_back(elementStr);
                        }
                    }
                }
            } else {
                throw std::runtime_error("Field 'psf_path' does not exist.");
                
            }

        }
        // Required in configuration file
        algorithmName = config["algorithm"].get<std::string>();
        epsilon = config["epsilon"].get<double>();
        iterations = config["iterations"].get<int>();
        lambda = config["lambda"].get<double>();
        psfSafetyBorder = config["psfSafetyBorder"].get<int>();
        subimageSize = config["subimageSize"].get<int>();
        borderType = config["borderType"].get<int>();
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
    }else {
        for(auto path: psfPathsCLI) {
            if (path.substr(path.find_last_of(".") + 1) == "json") {
                PSFConfig psf_config;
                if( psf_config.loadFromJSON(path)) {
                    psf_config.printValues();
                }else {
                    throw std::runtime_error("psf_config.loadFromJSON failed");
                    
                }
                psfConfigs.push_back(psf_config);

            } else {
                psfPaths.push_back(path);
            }
        }
    }

}

PSFPackage Dolphin::initPSF(){
    std::vector<PSF> psfs;

    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;

    for (const auto& psfPath : psfPaths) {
        PSF psftmp;
        if (psfPath.substr(psfPath.find_last_of(".") + 1) == "tif" || psfPath.substr(psfPath.find_last_of(".") + 1) == "tiff" || psfPath.substr(psfPath.find_last_of(".") + 1) == "ometif") {
            psftmp.readFromTifFile(psfPath.c_str());
        } else {
            psftmp.readFromTifDir(psfPath.c_str());
        }
        psfs.push_back(psftmp);
        psfCubeVec.push_back({});
        psfLayerVec.push_back({});
    }
    for (auto& psfConfig : psfConfigs) {
        psfCubeVec.push_back(psfConfig.psfCubes);
        psfLayerVec.push_back(psfConfig.psfLayers);
        if(psfConfig.psfPath != "") {
            PSF psftmp;
            psftmp.readFromTifFile(psfConfig.psfPath.c_str());
            psfs.push_back(psftmp);
            psfConfig.x = psftmp.image.slices[0].cols;
            psfConfig.y = psftmp.image.slices[0].rows;
            psfConfig.z = psftmp.image.slices.size();
        }else if(psfConfig.psfModel == "gauss") {
            double sigmax = psfConfig.sigmax;
            double sigmay = psfConfig.sigmay;
            double sigmaz = psfConfig.sigmaz;
            int x = psfConfig.x;
            int y = psfConfig.y;
            int z = psfConfig.z;

            PSFGenerator<GaussianPSFGeneratorAlgorithm, double&, double&, double&, int&, int&, int&> gaussianGenerator(sigmax, sigmay, sigmaz, x, y, z); // LK the psfgenerator is pretty useles as it is bound to the generator algorithm, should be used more like a factory
            // create PSFGenerator which simply takes the psfconfig, has multiple algorithms, and runs the specified algorithm to generate a psf
            PSF psftmp;
            psftmp = gaussianGenerator.generate();
            psfs.push_back(psftmp);
        }else{
            throw std::runtime_error("No correct PSF model ('gauss'/...)");
            
        }
    }

    int firstPsfX = psfs[0].image.slices[0].cols;
    int firstPsfY = psfs[0].image.slices[0].rows;
    int firstPsfZ = psfs[0].image.slices.size();
    for (int i = 0; i < psfs.size(); i++) {
        if(firstPsfX != psfs[i].image.slices[0].cols || firstPsfY != psfs[i].image.slices[0].rows || firstPsfZ != psfs[i].image.slices.size()) {
            throw std::runtime_error("PSF sizes do not match");
            std::cout << firstPsfX << " " << firstPsfY << " " << firstPsfZ << " " << psfs[i].image.slices[0].cols << " " << psfs[i].image.slices[0].rows << " "<<psfs[i].image.slices.size()<<std::endl;
            
        }
    }
    std::cout << "[INFO] " << psfs.size() << " PSF(s) loaded" << std::endl;
    if (savePsf) {
        for (int i = 0; i < psfs.size(); i++) {
            psfs[i].saveAsTifFile("../result/psf_"+std::to_string(i)+".tif");
        }
    }
    return PSFPackage{psfCubeVec, psfLayerVec, psfs};
}

Hyperstack Dolphin::initHyperstack(){
    Hyperstack hyperstack;
    if (image_path.substr(image_path.find_last_of(".") + 1) == "tif" || image_path.substr(image_path.find_last_of(".") + 1) == "tiff" || image_path.substr(image_path.find_last_of(".") + 1) == "ometif") {
        hyperstack.readFromTifFile(image_path.c_str());
    } else {
        std::cout << "[INFO] No file ending .tif, pretending image is DIR" << std::endl;
        hyperstack.readFromTifDir(image_path.c_str());
    }

    if (printInfo) {
        hyperstack.printMetadata();
    }
    if (showExampleLayers) {
        hyperstack.showChannel(0);
    }
    hyperstack.saveAsTifFile("../result/input_hyperstack.tif");
    hyperstack.saveAsTifDir("../result/input_hyperstack");
    return hyperstack;
}

std::unique_ptr<BaseAlgorithm> Dolphin::initDeconvolution(const std::vector<std::vector<int>>& psfCubeVec, const std::vector<std::vector<int>>& psfLayerVec){


    DeconvolutionConfig deconvConfig;
    deconvConfig.psfCubeVec = psfCubeVec;
    deconvConfig.psfLayerVec = psfLayerVec;

    deconvConfig.iterations = iterations;
    deconvConfig.epsilon = epsilon;
    deconvConfig.grid = grid;
    deconvConfig.lambda = lambda;
    deconvConfig.borderType = borderType;
    deconvConfig.psfSafetyBorder = psfSafetyBorder;
    deconvConfig.cubeSize = subimageSize;
    deconvConfig.gpu = gpu;

    deconvConfig.time = time;
    deconvConfig.saveSubimages = saveSubimages;

    return deconvolutionAlgorithmFactory(algorithmName, deconvConfig);
}

void Dolphin::run(){
    assert(algorithm && "deconvolutionAlgorithm must be initialized");
    assert(!psfs.empty() && "No PSFs loaded");
    assert(inputHyperstack.isValid() && "Input hyperstack not loaded");

    Hyperstack result = algorithm->run(inputHyperstack, psfs);
    // Hyperstack deconvHyperstack = deconvolutionAlgorithm->deconvolve(inputHyperstack, psfs);


    // TODO save function, maybe create more general writer class
    if (showExampleLayers) {
        result.showChannel(0);
    }

    result.saveAsTifFile("../result/deconv.tif");
    if(sep){
        result.saveAsTifDir("../result/deconv");
    }
    // result = inverseAlgorithm.deconvolve(hyperstack, psfs);
    //###PROGRAMM END###//
    std::cout << "[End DeconvTool]" << std::endl;
}