#pragma once
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"


using json = nlohmann::json;

class ConfigManager{
public:
    bool handleInput(int argc, char** argv);


    // Arguments
    std::string image_path;
    std::string psf_path;
    std::string config_file_path;
    std::string algorithmName;
    int iterations = 10; //RL and RLTV
    double lambda = 0.01; //RIF and RLTV
    double epsilon = 1e-6; // complex divison
    bool time = false; //show time
    bool sep = false; //save layer separate (TIF dir)
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    bool grid = false; //do grid processing
    int subimageSize = 0; //sub-image size (edge)
    int psfSafetyBorder = 10; //padding around PSF
    int borderType = 2; // = cv::BORDER_REFLECT extension type of image
    bool saveSubimages = false;
    std::string gpu = "";

    std::vector<std::string> psfPaths;
    std::vector<json> psfJSON;

private:
    std::vector<std::string> psfPathsCLI;
    json config;
    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    CLI::Option_group* cli_group;

    void handleJSONConfigs(const std::string& configPath);
    void setCLIOptions();

    json loadJSONFile(const std::string& path) const;
    std::string extractImagePath(const json& file) const;
    void processPSFPaths();
    void processSinglePSFPath(const std::string& path);
    void processPSFPathArray();
    bool isJSONFile(const std::string& path);
    void addPSFConfigFromJSON(const json& config);
    void extractAlgorithmParameters();
    void extractOptionalParameters();
    void handleCLIConfigs();
    void setCuda();
};