#pragma once
#include "CLI/CLI.hpp"

#include "dolphin/frontend/IFrontend.h"
#include "dolphin/SetupConfig.h"

#include <dolphin/deconvolution/DeconvolutionConfig.h>
#include <dolphin/ServiceAbstractions.h>
#include <dolphin/psf/PSFGeneratorFactory.h>


struct ConfigBundle {
    SetupConfig setupConfig;
    DeconvolutionConfig deconvConfig;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;

    bool hasSetup = false;
    bool hasDeconv = false;
    bool hasPSF = false;
};


class CLIFrontend : public IFrontend{
public:
    CLIFrontend(Dolphin* dolphin, int argc, char** argv);
    void run() override;


private:
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

    SetupConfigPSF psfConfig;

    int argc;
    char** argv;
    std::string setupConfigPath;


    void deconvolution();
    void psfgenerator();

    bool parseCLI();
    void readCLISetupConfigPath();
    void readSetupConfigParameters();
    void readCLIParametersDeconvolution();

    void loadJSONBundle(const std::string& path);
    static ConfigBundle mergeBundles(const ConfigBundle& jsonBundle, const ConfigBundle& cliBundle);

    bool readPSFFromConfigFile();

    bool handlePSFGeneration();
    bool handleDeconvolution(const ConfigBundle& bundle);


    std::vector<std::string> checkRequired(Config& config) const ;
    void addParameters(Config& config, CLI::Option_group* group);

    PSFGenerationRequest generatePSFRequest(std::shared_ptr<SetupConfigPSF> setupConfig);
    DeconvolutionRequest generateDeconvRequest(const ConfigBundle& bundle);
};
