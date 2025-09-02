#pragma once
#include "../lib/CLI/CLI11.hpp"

#include "IFrontend.h"
#include "DeconvolutionConfig.h"

class CLIFrontend : public IFrontend{
public:
    CLIFrontend(SetupConfig* config, int argc, char** argv);
    void init(int argc, char** argv);
    void run() override;


private:
    CLI::App app{"Dolphin"};
    CLI::App* deconvolutionCLI = nullptr;
    CLI::App* psfCLI = nullptr;

    CLI::Option_group* cli_group;
    CLI::Option_group* configGroup;


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
};