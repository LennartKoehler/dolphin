#include <iostream>
#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"



void runWithConfig(std::string configpath){
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    try{
        SetupConfig config = SetupConfig::createFromJSONFile(configpath);
        DeconvolutionConfig deconvConfig = DeconvolutionConfig::createFromJSONFile(configpath);
        DeconvolutionRequest request(std::make_shared<SetupConfig>(config), std::make_shared<DeconvolutionConfig>(deconvConfig));
        dolphin->deconvolve(request);
    }
    catch(const std::exception& e){
        std::cout << e.what() << std::endl;
        return;
    }


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
