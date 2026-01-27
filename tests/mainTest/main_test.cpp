#include <iostream>
#include "Dolphin.h"
#include "frontend/SetupConfig.h"



void runWithConfig(std::string configpath){
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    SetupConfig config = SetupConfig::createFromJSONFile(configpath); 

    DeconvolutionRequest request(std::make_shared<SetupConfig>(config));
    
    dolphin->deconvolve(request);
}


int main(int argc, char** argv) {
    std::cout << "=== DOLPHIN Test ===" << std::endl;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json>" << std::endl;
        return 1;
    }
    std::string configPath = argv[1];


    // Run the tests
    runWithConfig(configPath);
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}