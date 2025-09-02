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
    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    CLI::Option_group* cli_group;
    SetupConfig setupConfig;
    DeconvolutionConfig deconvolutionConfig;
    int argc;
    char** argv;

    bool parseCLI();
    void readCLIParameters();
    void readCLIParametersPSF();
    void readCLIParametersDeconvolution();
};