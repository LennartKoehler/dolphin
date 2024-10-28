#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"

#include "HyperstackImage.h"
#include "PSF.h"

#include "PSFGenerator.h"
#include "GaussianPSFGeneratorAlgorithm.h"
#include "SimpleGaussianPSFGeneratorAlgorithm.h"
#include "BornWolfModel.h"

#include "DeconvolutionAlgorithm.h"
#include "InverseFilterDeconvolutionAlgorithm.h"
#include "RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "RLDeconvolutionAlgorithm.h"
#include "RLTVDeconvolutionAlgorithm.h"

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
    std::vector<int> secondpsflayers; //sub-image layers for secondPSF
    std::vector<int> secondpsfcubes; //sub-images for secondPSF


    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    // Define a group for CLI arguments
    CLI::Option_group *cli_group = app.add_option_group("CLI", "Commandline options");

    cli_group->add_option("-i,--image", image_path, "Input image Path")->required();
    cli_group->add_option("-p,--psf", psf_path, "Input PSF path or 'synthetic'")->required();
    cli_group->add_option("--psf2", psf_path_2, "Input second PSF path or 'synthetic'");
    cli_group->add_option("--psftype", psfModel, "Type of synthetic PSF ('gauss')");
    cli_group->add_option("-a,--algorithm", algorithm, "Algorithm selection ('rl'/'rltv'/'rif'/'inverse')")->required();
    cli_group->add_option("--dataFormatImage", dataFormatImage, "Data format of Image ('FILE'/'DIR')")->required();
    cli_group->add_option("--dataFormatPSF", dataFormatPSF, "Data format of PSF ('FILE'/'DIR')")->required();

    cli_group->add_option("--psfx", psfx, "PSF width for synthetic PSF  [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfy", psfy, "PSF heigth for synthetic PSF  [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfz", psfz, "PSF depth for synthetic PSF  [30]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmax", sigmax, "SigmaX for synthetic PSF [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmay", sigmay, "SigmaY for synthetic PSF  [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmaz", sigmaz, "SigmaZ for synthetic PSF  [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmax_2", sigmax_2, "SigmaX for second synthetic PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmay_2", sigmay_2, "SigmaY for second synthetic PSF  [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmaz_2", sigmaz_2, "SigmaZ for second synthetic PSF  [15]")->check(CLI::PositiveNumber);

    cli_group->add_option("--epsilon", epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(CLI::PositiveNumber);
    cli_group->add_option("--lambda", lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", borderType, "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--cubeSize", cubeSize, "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(CLI::PositiveNumber);

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
        if(psf_path == "synthetic"){
            if(!config.contains("psfmodel")){
                std::cout << "[WARNING] No PSF model specified, using default 'gauss'" << std::endl;
            }else{
                psfModel = config["psfmodel"].get<std::string>();
            }
        }
        if (config.contains("psf_path_2")) {
            psf_path_2 = config["psf_path_2"].get<std::string>();
            if(psf_path_2 == "synthetic"){
                if(!config.contains("psfmodel")){
                    std::cout << "[WARNING] No second PSF model specified, using default 'gauss'" << std::endl;
                }else{
                    psfModel = config["psfmodel"].get<std::string>();
                }
            }
            if (!(config.contains("sigmax_2") || config.contains("sigmay_2") || config.contains("sigmaz_2") && psf_path_2 == "synthetic")) {
                std::cerr << "[ERROR] Incomplete sigma values for second psf" << std::endl;
                return EXIT_FAILURE;
            } else{
                sigmax_2 = config.value("sigmax_2", 0.0); // sigmax_2 ist vom Typ double
                sigmay_2 = config.value("sigmay_2", 0.0); // sigmay_2 ist vom Typ double
                sigmaz_2 = config.value("sigmaz_2", 0.0); // sigmaz_2 ist vom Typ double
                if(sigmax_2 == 0.0){
                    std::cout << "[WARNING] sigmaX value 0" << std::endl;
                }
                if(sigmay_2 == 0.0){
                    std::cout << "[WARNING] sigmaY value 0" << std::endl;
                }
                if(sigmaz_2 == 0.0){
                    std::cout << "[WARNING] sigmaZ value 0" << std::endl;
                }
            }
            // Check, if "secondpsflayers" in config
            if (config.contains("secondpsflayers")) {
                secondpsflayers = config["secondpsflayers"].get<std::vector<int>>();

                // Check values
                std::cout << "[STATUS] secondpsflayers: ";
                for (const int& layer : secondpsflayers) {
                    std::cout << layer << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "[WARNING] 'secondpsflayers' not found in the configuration file. Using default values." << std::endl;
                secondpsflayers = {};
            }
            // Cehck, if "secondpsflayers" in config
            if (config.contains("secondpsfcubes")) {
                secondpsfcubes = config["secondpsfcubes"].get<std::vector<int>>();

                // Check values
                std::cout << "[STATUS] secondpsfcubes: ";
                for (const int& cube : secondpsfcubes) {
                    std::cout << cube << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "[WARNING] 'secondpsfcubes' not found in the configuration file. Using default values." << std::endl;
                secondpsfcubes = {};
            }
        }
        secondPSF = true;
    }
    if(secondpsflayers.empty() && secondpsfcubes.empty()){
        secondPSF = false;
    }
    // Required in configuration file
    algorithm = config["algorithm"].get<std::string>();
    dataFormatImage = config["dataFormatImage"].get<std::string>();
    dataFormatPSF = config["dataFormatPSF"].get<std::string>();
    sigmax = config["sigmax"].get<double>();
    sigmay = config["sigmay"].get<double>();
    sigmaz = config["sigmaz"].get<double>();
    psfx = config["psfx"].get<int>();
    psfy = config["psfy"].get<int>();
    psfz = config["psfz"].get<int>();
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

    //###PROGRAMM START###//
    PSF psf;
    PSF psf_2;
    std::vector<PSF> psfs;
    if (psf_path == "synthetic") {
        if(psfModel == "gauss") {
            PSFGenerator<GaussianPSFGeneratorAlgorithm, double &, double &, double &, int &, int &, int &> gaussianGenerator(
                    sigmax, sigmay, sigmaz, psfx, psfy, psfz);
            psf = gaussianGenerator.generate();
            psfs.push_back(psf);
        }else{
            std::cerr << "[ERROR] No correct PSF model ('gauss'/...)" << std::endl;
            return EXIT_FAILURE;
        }
    } else {
        if (dataFormatPSF == "DIR") {
            psf.readFromTifDir(psf_path.c_str());
        } else if (dataFormatPSF == "FILE") {
            psf.readFromTifFile(psf_path.c_str());
        } else {
            std::cerr << "[ERROR] No correct dataformat for PSF - choose DIR or FILE" << std::endl;
            return EXIT_FAILURE;
        }
        psfs.push_back(psf);
    }
    if (config.contains("psf_path_2")) {
        if(psf_path_2 == "synthetic"){
            if(psfModel == "gauss"){
                std::cout << "[INFO] Generated second PSF" << std::endl;
                PSFGenerator<GaussianPSFGeneratorAlgorithm, double &, double &, double &, int &, int &, int &> gaussianGenerator(
                        sigmax_2, sigmay_2, sigmaz_2, psfx, psfy, psfz);
                psf_2 = gaussianGenerator.generate();
                psfs.push_back(psf_2);
            }else{
                std::cerr << "[ERROR] No correct PSF model ('gauss'/...)" << std::endl;
                return EXIT_FAILURE;
            }
            if((psf_2.image.slices.size() != psf.image.slices.size()) || (psf_2.image.slices[0].cols != psf.image.slices[0].cols) || (psf_2.image.slices[0].rows != psf.image.slices[0].rows)){
                std::cerr << "[ERROR] Dimensions of both PSFs are not equal" << std::endl;
                return EXIT_FAILURE;
            }
        }else {
            if (dataFormatPSF == "DIR") {
                psf_2.readFromTifDir(psf_path_2.c_str());
            } else if (dataFormatPSF == "FILE") {
                psf_2.readFromTifFile(psf_path_2.c_str());
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
    deconvConfig.secondpsflayers = secondpsflayers;
    deconvConfig.secondpsfcubes = secondpsfcubes;
    deconvConfig.secondPSF = secondPSF;

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
