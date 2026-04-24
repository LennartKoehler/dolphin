#pragma once
#include "CLI/CLI.hpp"

#include "dolphin/frontend/IFrontend.h"
#include "dolphin/SetupConfig.h"

#include <dolphin/deconvolution/DeconvolutionConfig.h>
#include <dolphin/ServiceAbstractions.h>


class CLIFrontend : public IFrontend{
public:
    CLIFrontend(Dolphin* dolphin, int argc, char** argv);
    void run() override;


private:
    CLI::App app{"Dolphin"};
    CLI::App* deconvolutionCLI = nullptr;
    CLI::App* psfCLI = nullptr;

    CLI::Option_group* cli_group;
    CLI::Option_group* configGroup;

    CLI::Option_group* psfcli_group;
    CLI::Option_group* psfconfigGroup;

    SetupConfigPSF psfConfig;
    SetupConfig setupConfig;
    DeconvolutionConfig deconvolutionConfig;

    int argc;
    char** argv;
    std::string setupConfigPath;


    void deconvolution();
    void psfgenerator();

    bool parseCLI();
    void readCLISetupConfigPath();
    void readSetupConfigParameters();
    void readCLIParametersDeconvolution();
    bool readDeconvolutionFromConfigFile();

    bool readPSFFromConfigFile();

    bool handlePSFGeneration();
    bool handleDeconvolution();


    std::vector<std::string> checkRequired(Config& config) const ;
    void addParameters(Config& config, CLI::Option_group* group);

    PSFGenerationRequest generatePSFRequest(std::shared_ptr<SetupConfigPSF> setupConfig);
    DeconvolutionRequest generateDeconvRequest(std::shared_ptr<SetupConfig> setupConfig, std::shared_ptr<DeconvolutionConfig> deconvConfig);
};
