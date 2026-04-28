#include <iostream>
#include "dolphin/Dolphin.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"


void progressVisualization(std::atomic<float>& current, float max){
    // Calculate progress

    float barWidth = 50;
    int pos = static_cast<int>((current * barWidth) / max);
    int progress = static_cast<int>((current * 100) / max);
    // Print progress bar
    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] "
      << std::setw(3)
      << progress << "%";
    std::cout.flush();

    if(current >= max){
        std::cout <<std::endl;
    }
}

void runWithConfig(std::string configpath){
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    try{
        SetupConfig config = SetupConfig::createFromJSONFile(configpath);
        DeconvolutionConfig deconvConfig = DeconvolutionConfig::createFromJSONFile(configpath);
        DeconvolutionRequest request(std::make_shared<SetupConfig>(config), std::make_shared<DeconvolutionConfig>(deconvConfig), progressVisualization);
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
