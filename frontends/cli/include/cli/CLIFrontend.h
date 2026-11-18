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
#pragma once
#include "CLI/CLI.hpp"

#include "dolphin/frontend/IFrontend.h"
#include "dolphin/SetupConfig.h"

#include <dolphin/deconvolution/AlgorithmNames.h>
#include <dolphin/deconvolution/DeconvolutionConfig.h>
#include <dolphin/ServiceAbstractions.h>
#include <dolphin/psf/PSFGeneratorFactory.h>


struct ConfigBundle {
    SetupConfig setupConfig;
    // Deconvolution configs are polymorphic (one class per algorithm), so the
    // bundle holds whichever concrete config was created for the run.
    std::shared_ptr<DeconvolutionConfig> deconvConfig;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;

    bool hasSetup = false;
    bool hasDeconv = false;
    bool hasPSF = false;
};

struct PSFConfigBundle {
    SetupConfigPSF setupConfig;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;

    bool hasSetup = false;
    bool hasPSF = false;
};


class CLIFrontend : public IFrontend{
public:
    CLIFrontend(Dolphin* dolphin, int argc, char** argv);
    void run() override;


protected:
    CLI::App app{"Dolphin"};
    CLI::App* deconvolutionCLI = nullptr;
    CLI::App* psfCLI = nullptr;

    CLI::Option_group* setupCliGroup = nullptr;
    CLI::Option_group* deconvCliGroup = nullptr;

    CLI::Option_group* psfcli_group = nullptr;
    CLI::Option_group* psfconfigGroup = nullptr;
    CLI::Option_group* psfPathGroup = nullptr;

    ConfigBundle jsonBundle;
    ConfigBundle cliBundle;

    PSFConfigBundle jsonPsfBundle;
    PSFConfigBundle cliPsfBundle;

    int argc;
    char** argv;
    std::string configPath;

    // Set by --algorithm_name; decides the config class and its CLI options.
    std::string cliAlgorithmName = DeconvolutionAlgorithms::kRichardsonLucy;

    std::string setupConfigPath;
    std::string deconvConfigPath;

    std::vector<std::string> psfConfigPaths;


    void deconvolution();
    void psfgenerator();

    bool parseCLI();
    void readCLISetupConfigPath();
    void readSetupConfigParameters();
    void readCLIParametersDeconvolution();
    void addAlgorithmSelectionOption();
    static std::string prescanAlgorithmName(int argc, char** argv);
    static std::string peekAlgorithmName(const std::string& path);

    void loadJSONBundle(const std::string& path);
    static ConfigBundle mergeBundles(const ConfigBundle& jsonBundle, const ConfigBundle& cliBundle);

    void loadPSFJSONBundle(const std::string& path);
    static PSFConfigBundle mergePSFBundles(const PSFConfigBundle& jsonBundle, const PSFConfigBundle& cliBundle);

    bool handlePSFGeneration(const PSFConfigBundle& bundle);
    bool handleDeconvolution(const ConfigBundle& bundle);


    std::vector<std::string> checkRequired(Config& config) const ;
    void addParameters(Config& config, CLI::Option_group* group);

    static std::shared_ptr<PSFConfig> loadPSFConfigFromPath(const std::string& path);

    static bool loadSetupConfigFromJSON(const json& jsonData, SetupConfigPSF& config);
    static void loadDeconvConfigFromJSON(const json& jsonData, std::shared_ptr<DeconvolutionConfig>& config);
    static std::vector<std::shared_ptr<PSFConfig>> loadPSFConfigsFromJSON(const json& jsonData);

    static void loadSetupConfigFromFile(const std::string& path, SetupConfigPSF& config);
    static void loadDeconvConfigFromFile(const std::string& path, std::shared_ptr<DeconvolutionConfig>& config);
    static std::vector<std::shared_ptr<PSFConfig>> loadPSFConfigsFromFile(const std::string& path);

    PSFGenerationRequest generatePSFRequest(const PSFConfigBundle& bundle);
    DeconvolutionRequest generateDeconvRequest(const ConfigBundle& bundle);
};
