#pragma once
#include "../lib/CLI/CLI11.hpp"

#include "IFrontend.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "SetupConfig.h"
#include "ServiceAbstractions.h"
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

    SetupConfig setupConfig;
    DeconvolutionConfig deconvolutionConfig;
    int argc;
    char** argv;
    std::string setupConfigPath = "";


    void deconvolution();
    void psfgenerator();

    bool parseCLI();
    void readCLISetupConfigPath();
    void readCLIParameters();
    void readCLIParametersPSF();
    void readCLIParametersDeconvolution();
    void handlePSFGeneration();
    void handleDeconvolution();

    PSFGenerationRequest generatePSFRequest(const std::string& psfconfigpath);
    DeconvolutionRequest generateDeconvRequest(std::shared_ptr<SetupConfig> setupConfig);
};