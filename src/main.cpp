#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"

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
    std::string psf_path_2;
    bool secondPSF = false; //use second PSF
    std::string algorithm;
    std::string psfModel = "gauss";
    std::string dataFormatImage; //FILE or DIR
    std::string dataFormatPSF; //FILE or DIR
    int iterations = 10; //RL and RLTV
    double lambda = 0.01; //RIF and RLTV
    double sigmax = 5.0; //synthetic PSF
    double sigmay = 5.0; //synthetic PSF
    double sigmaz = 5.0; //synthetic PSF
    int psfx = 20; //synthetic PSF width
    int psfy = 20; //synthetic PSF heigth
    int psfz = 30; //synthetic PSF depth/layers
    double epsilon = 1e-6; // complex divison
    bool time = false; //show time
    bool sep = false; //save layer separate (TIF dir)
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    bool grid = true; //do grid processing
    int cubeSize = 0; //sub-image size (edge)
    int psfSafetyBorder = 10; //padding around PSF
    int borderType = cv::BORDER_REFLECT; //extension type of image

    double sigmax_2 = 10.0; //synthetic second PSF
    double sigmay_2 = 10.0; //synthetic second PSF
    double sigmaz_2 = 15.0; //synthetic second PSF
    int psfx_2 = 20; //synthetic PSF width
    int psfy_2 = 20; //synthetic PSF heigth
    int psfz_2 = 30; //synthetic PSF depth/layers
    std::string psfModel_2 = "gauss";
    bool psf_1_is_config = false;
    bool psf_2_is_config = false;

    std::vector<int> secondpsflayers; //sub-image layers for secondPSF
    std::vector<int> secondpsfcubes; //sub-images for secondPSF


    //TODO CLI daten einlesen
    PSFConfig psfconfig_1;
    PSFConfig psfconfig_2;

    std::string gpu = "";


    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    // Define a group for CLI arguments
    CLI::Option_group *cli_group = app.add_option_group("CLI", "Commandline options");

    cli_group->add_option("-i,--image", image_path, "Input image Path")->required();
    cli_group->add_option("-p,--psf", psf_path, "Input PSF path or 'synthetic'")->required();
    cli_group->add_option("--psf2", psf_path_2, "Input second PSF path or 'synthetic'");
    cli_group->add_option("--psfmodel", psfModel, "Model of synthetic PSF ['gauss'] ('gauss')");
    cli_group->add_option("-a,--algorithm", algorithm, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    cli_group->add_option("--dataFormatImage", dataFormatImage, "Data format of Image ('FILE'/'DIR')")->required();
    cli_group->add_option("--dataFormatPSF", dataFormatPSF, "Data format of PSF ('FILE'/'DIR')")->required();

    cli_group->add_option("--psfx", psfx, "PSF width for synthetic PSF [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfy", psfy, "PSF heigth for synthetic PSF [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfz", psfz, "PSF depth for synthetic PSF [30]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmax", sigmax, "SigmaX for synthetic PSF [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmay", sigmay, "SigmaY for synthetic PSF [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmaz", sigmaz, "SigmaZ for synthetic PSF [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmax_2", sigmax_2, "SigmaX for second synthetic PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmay_2", sigmay_2, "SigmaY for second synthetic PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmaz_2", sigmaz_2, "SigmaZ for second synthetic PSF [15]")->check(CLI::PositiveNumber);

    cli_group->add_option("--epsilon", epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--cubeSize", cubeSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);

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
        std::cout<< "[STATUS] " << config_file_path << " successfully read" << std::endl;
        if (!config_file.is_open()) {
            std::cerr << "[ERROR] failed opening of configuration file:" << config_file_path << std::endl;
            return EXIT_FAILURE;
        }
        config_file >> config;
        // Values from configuration file passed to arguments
        image_path = config["image_path"].get<std::string>();
        psf_path = config["psf_path"].get<std::string>();
    }

    // Required in configuration file
    algorithm = config["algorithm"].get<std::string>();
    dataFormatImage = config["dataFormatImage"].get<std::string>();
    dataFormatPSF = config["dataFormatPSF"].get<std::string>();
    epsilon = config["epsilon"].get<double>();
    iterations = config["iterations"].get<int>();
    lambda = config["lambda"].get<double>();
    psfSafetyBorder = config["psfSafetyBorder"].get<int>();
    cubeSize = config["cubeSize"].get<int>();
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
    if (psf_path.substr(psf_path.find_last_of(".") + 1) == "json") {
        psf_1_is_config = true;

        if( psfconfig_1.loadFromJSON(psf_path)) {
            psfconfig_1.printValues();
        }else {
            return EXIT_FAILURE;
        }

    }
    if (config.contains("psf_path_2")) {
        psf_path_2 = config["psf_path_2"].get<std::string>();
        if(psf_path_2 != "none") {
            if (psf_path_2.substr(psf_path.find_last_of(".") + 1) == "json") {
                psf_2_is_config = true;
                if( psfconfig_2.loadFromJSON(psf_path_2)) {
                    psfconfig_2.printValues();
                }else {
                    return EXIT_FAILURE;
                }
            }
            secondPSF = true;
        }
    }

    //###PROGRAMM START###//
    PSF psf;
    PSF psf_2;
    std::vector<PSF> psfs;

    if (dataFormatPSF == "DIR") {
        psf.readFromTifDir(psf_path.c_str());
    } else if (dataFormatPSF == "FILE") {
        if (psf_1_is_config == true) {
            if(psfModel == "gauss") {
                PSFGenerator<GaussianPSFGeneratorAlgorithm, double &, double &, double &, int &, int &, int &> gaussianGenerator(psfconfig_1.sigmax, psfconfig_1.sigmay, psfconfig_1.sigmaz, psfconfig_1.x, psfconfig_1.y, psfconfig_1.z);
                psf = gaussianGenerator.generate();
            }else{
                std::cerr << "[ERROR] No correct PSF model ('gauss'/...)" << std::endl;
                return EXIT_FAILURE;
            }
        }else {
            psf.readFromTifFile(psf_path.c_str());
        }
    } else {
        std::cerr << "[ERROR] No correct dataformat for PSF - choose DIR or FILE" << std::endl;
        return EXIT_FAILURE;
    }
    psfs.push_back(psf);

    if (secondPSF) {
        if (dataFormatPSF == "DIR") {
            psf_2.readFromTifDir(psf_path_2.c_str());
        } else if (dataFormatPSF == "FILE") {
            if(psf_2_is_config == true){
                if(psfModel_2 == "gauss"){
                    std::cout << "[INFO] Generated second PSF" << std::endl;
                    PSFGenerator<GaussianPSFGeneratorAlgorithm, double &, double &, double &, int &, int &, int &> gaussianGenerator(psfconfig_2.sigmax, psfconfig_2.sigmay, psfconfig_2.sigmaz, psfconfig_2.x, psfconfig_2.y, psfconfig_2.z);
                    psf_2 = gaussianGenerator.generate();
                }else{
                    std::cerr << "[ERROR] No correct PSF model ('gauss'/...)" << std::endl;
                    return EXIT_FAILURE;
                }
                if((psf_2.image.slices.size() != psf.image.slices.size()) || (psf_2.image.slices[0].cols != psf.image.slices[0].cols) || (psf_2.image.slices[0].rows != psf.image.slices[0].rows)){
                    std::cerr << "[ERROR] Dimensions of both PSFs are not equal" << std::endl;
                    return EXIT_FAILURE;
                }
            }else {
                psf_2.readFromTifFile(psf_path_2.c_str());
            }
        } else {
            std::cerr << "[ERROR] No correct dataformat for PSF - choose DIR or FILE" << std::endl;
            return EXIT_FAILURE;
        }
        if((psf_2.image.slices.size() != psf.image.slices.size()) || (psf_2.image.slices[0].cols != psf.image.slices[0].cols) || (psf_2.image.slices[0].rows != psf.image.slices[0].rows)){
            std::cerr << "[ERROR] Dimensions of both PSFs are not equal" << std::endl;
            return EXIT_FAILURE;
        }
        psfs.push_back(psf_2);
    }
    std::cout << "[INFO] " << psfs.size() << " PSF(s) loaded" << std::endl;

    Hyperstack hyperstack;
    if(dataFormatImage == "FILE"){
        hyperstack.readFromTifFile(image_path.c_str());
    }else if(dataFormatImage == "DIR"){
        hyperstack.readFromTifDir(image_path.c_str());
    } else {
        std::cerr << "[ERROR] No correct dataformat for Image - choose DIR or FILE" << std::endl;
        return EXIT_FAILURE;
    }
    if (savePsf) {
        psf.saveAsTifFile("../result/psf.tif");
        if(secondPSF) {
            psf_2.saveAsTifFile("../result/psf_2.tif");
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
    deconvConfig.cubeSize = cubeSize;
    deconvConfig.secondpsflayers = psfconfig_2.psfLayers;
    deconvConfig.secondpsfcubes = psfconfig_2.psfCubes;
    deconvConfig.secondPSF = secondPSF;
    deconvConfig.gpu = gpu;

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
        std::cout << "[INFO] Algorithm duration: " << duration.count() << " ms" << std::endl;
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
