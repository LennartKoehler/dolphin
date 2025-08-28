#pragma once
#include "../lib/CLI/CLI11.hpp"

#include "IFrontend.h"

class CLIFrontend : public IFrontend{
public:
    CLIFrontend(ConfigManager* config, int argc, char** argv);
    void init(int argc, char** argv);
    void run() override;


private:
    CLI::App app{"deconvtool - Deconvolution of Microscopy Images"};
    CLI::Option_group* cli_group;
    std::vector<std::string> psfPaths;
    std::string configFilePath;
    int argc;
    char** argv;

    bool parseCLI();
    void readCLIParameters();
    void readCLIParametersPSF();
    void handleCLIConfigs();
};