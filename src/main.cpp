#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"
#include <sys/stat.h>
#ifdef CUDA_AVAILABLE
#include "../lib/cube/include/CUBE.h"
#endif

#include "HyperstackImage.h"
#include "PSF.h"

#include "PSFGenerator.h"
#include "PSFConfig.h"
#include "GaussianPSFGeneratorAlgorithm.h"
#include "SimpleGaussianPSFGeneratorAlgorithm.h"
#include "BornWolfModel.h"

#include "DeconvolutionAlgorithm.h"
#include "InverseFilterDeconvolutionAlgorithm.h"
#include "RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "RLDeconvolutionAlgorithm.h"
#include "RLTVDeconvolutionAlgorithm.h"
#include "RLADDeconvolutionAlgorithm.h"

using json = nlohmann::json;


int main(int argc, char** argv) {
    std::cout << "[Start DeconvTool]" << std::endl;

    // Arguments
    std::string image_path;
    std::string psf_path;
    std::string algorithm;
    int iterations = 10; //RL and RLTV
    double lambda = 0.01; //RIF and RLTV
    double epsilon = 1e-6; // complex divison
    bool time = false; //show time
    bool sep = false; //save layer separate (TIF dir)
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    bool grid = true; //do grid processing
    int subimageSize = 0; //sub-image size (edge)
    int psfSafetyBorder = 10; //padding around PSF
    int borderType = cv::BORDER_REFLECT; //extension type of image

    PSFConfig psfconfig_1;
    PSFConfig psfconfig_2;
    std::vector<PSFConfig> psfConfigs;
    std::vector<std::string> psfPaths;
    std::vector<std::string> psfPathsCLI;

    std::string gpu = "";

    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    // Define a group for CLI arguments
    CLI::Option_group *cli_group = app.add_option_group("CLI", "Commandline options");

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
    cli_group->add_option("-a,--algorithm", algorithm, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
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

    // Define a group for configuration file
    CLI::Option_group *config_group = app.add_option_group("Config", "Configuration file");
    std::string config_file_path;
    config_group->add_option("-c,--config", config_file_path, "Path to configuration file")->required();

    // Exclude CLI arguments if configuration file is set
    cli_group->excludes(config_group);
    config_group->excludes(cli_group);

    CLI11_PARSE(app, argc, argv);

    json config;

    if (!config_file_path.empty()) {
        // Read configuration file
        std::ifstream config_file(config_file_path);
        if (!config_file.is_open()) {
            std::cerr << "[ERROR] failed opening of configuration file:" << config_file_path << std::endl;
            return EXIT_FAILURE;
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
                        std::cerr << "[ERROR] psf_config.loadFromJSON failed" << std::endl;
                        return EXIT_FAILURE;
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
                                std::cerr << "[ERROR] psf_config.loadFromJSON failed" << std::endl;
                                return EXIT_FAILURE;
                            }
                            psfConfigs.push_back(psf_config);

                        } else {
                            psfPaths.push_back(elementStr);
                        }
                    }
                }
            } else {
                std::cerr << "[ERROR] Field 'psf_path' does not exist." << std::endl;
                return EXIT_FAILURE;
            }

        }
        // Required in configuration file
        algorithm = config["algorithm"].get<std::string>();
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
                    std::cerr << "[ERROR] psf_config.loadFromJSON failed" << std::endl;
                    return EXIT_FAILURE;
                }
                psfConfigs.push_back(psf_config);

            } else {
                psfPaths.push_back(path);
            }
        }
    }



    //###PROGRAMM START###//
    PSF psf;
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

            PSFGenerator<GaussianPSFGeneratorAlgorithm, double&, double&, double&, int&, int&, int&> gaussianGenerator(sigmax, sigmay, sigmaz, x, y, z);

            PSF psftmp;
            psftmp = gaussianGenerator.generate();
            psfs.push_back(psftmp);
        }else{
            std::cerr << "[ERROR] No correct PSF model ('gauss'/...)" << std::endl;
            return EXIT_FAILURE;
        }
    }

    int firstPsfX = psfs[0].image.slices[0].cols;
    int firstPsfY = psfs[0].image.slices[0].rows;
    int firstPsfZ = psfs[0].image.slices.size();
    for (int i = 0; i < psfs.size(); i++) {
        if(firstPsfX != psfs[i].image.slices[0].cols || firstPsfY != psfs[i].image.slices[0].rows || firstPsfZ != psfs[i].image.slices.size()) {
            std::cerr << "[ERROR] PSF sizes do not match" << std::endl;
            std::cout << firstPsfX << " " << firstPsfY << " " << firstPsfZ << " " << psfs[i].image.slices[0].cols << " " << psfs[i].image.slices[0].rows << " "<<psfs[i].image.slices.size()<<std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "[INFO] " << psfs.size() << " PSF(s) loaded" << std::endl;

    Hyperstack hyperstack;
    if (image_path.substr(image_path.find_last_of(".") + 1) == "tif" || image_path.substr(image_path.find_last_of(".") + 1) == "tiff" || image_path.substr(image_path.find_last_of(".") + 1) == "ometif") {
        hyperstack.readFromTifFile(image_path.c_str());
    } else {
        std::cout << "[INFO] No file ending .tif, pretending image is DIR" << std::endl;
        hyperstack.readFromTifDir(image_path.c_str());
    }

    if (savePsf) {
        for (int i = 0; i < psfs.size(); i++) {
            psfs[i].saveAsTifFile("../result/psf_"+std::to_string(i)+".tif");
        }
    }
    if (printInfo) {
        hyperstack.printMetadata();
    }
    if (showExampleLayers) {
        hyperstack.showChannel(0);
    }
    hyperstack.saveAsTifFile("../result/input_hyperstack.tif");
    hyperstack.saveAsTifDir("../result/input_hyperstack");

    DeconvolutionConfig deconvConfig;
    deconvConfig.iterations = iterations;
    deconvConfig.epsilon = epsilon;
    deconvConfig.grid = grid;
    deconvConfig.lambda = lambda;
    deconvConfig.borderType = borderType;
    deconvConfig.psfSafetyBorder = psfSafetyBorder;
    deconvConfig.cubeSize = subimageSize;
    deconvConfig.gpu = gpu;
    deconvConfig.psfCubeVec = psfCubeVec;
    deconvConfig.psfLayerVec = psfLayerVec;


    Hyperstack deconvHyperstack;

    // Starttime
    auto start = std::chrono::high_resolution_clock::now();

    if (algorithm == "inverse") {
        DeconvolutionAlgorithm<InverseFilterDeconvolutionAlgorithm> inverseAlgorithm(deconvConfig);
        deconvHyperstack = inverseAlgorithm.deconvolve(hyperstack, psfs);
    } else if (algorithm == "rl") {
        DeconvolutionAlgorithm<RLDeconvolutionAlgorithm> rlAlgorithm(deconvConfig);
        deconvHyperstack = rlAlgorithm.deconvolve(hyperstack, psfs);
    }else if (algorithm == "rltv") {
        DeconvolutionAlgorithm<RLTVDeconvolutionAlgorithm> rltvAlgorithm(deconvConfig);
        deconvHyperstack = rltvAlgorithm.deconvolve(hyperstack, psfs);
    } else if (algorithm == "rif") {
        DeconvolutionAlgorithm<RegularizedInverseFilterDeconvolutionAlgorithm> rifAlgorithm(deconvConfig);
        deconvHyperstack = rifAlgorithm.deconvolve(hyperstack, psfs);
        //TODO
    } else if (algorithm == "convolve") {
        deconvHyperstack = hyperstack.convolve(psf);
    } else {
        std::cerr << "[ERROR] Please choose a --algorithm: InverseFilter[inverse], RegularizedInverseFilter[rif], RichardsonLucy[rl], RichardsonLucyTotalVariation[rltv]" << std::endl;
        return EXIT_FAILURE;
    }

    if (time) {
        // Endtime
        auto end = std::chrono::high_resolution_clock::now();
        // Calculation of the duration of the programm
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "[INFO] Algorithm runtime: " << duration.count() << " ms" << std::endl;
    }
    if (showExampleLayers) {
        deconvHyperstack.showChannel(0);
    }

    deconvHyperstack.saveAsTifFile("../result/deconv.tif");
    if(sep){
        deconvHyperstack.saveAsTifDir("../result/deconv");
    }

    //###PROGRAMM END###//
    std::cout << "[End DeconvTool]" << std::endl;
    return EXIT_SUCCESS;
}

