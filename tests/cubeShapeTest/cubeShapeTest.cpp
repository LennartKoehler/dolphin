#include <iostream>
#include "dolphin/Dolphin.h"
#include "dolphin/frontend/SetupConfig.h"
#include "dolphinbackend/CuboidShape.h"

#include <chrono>


void runWithConfig(const SetupConfig& config){
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    


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

    SetupConfig config = SetupConfig::createFromJSONFile(configPath); 

    int base = 64;
    config.cubePadding = std::array<int, 3>{base, base, base};

    std::vector<std::array<int, 3>> cubeSizes{
        std::array<int, 3>{3 * base, 3 * base, 3 * base},
        std::array<int, 3>{4 * base, 4* base -1, 4 * base},
        std::array<int, 3>{8 * base, 8 * base, 8 * base},
    };
    for (auto cubeSize : cubeSizes){

        config.cubeSize = cubeSize;
        auto start = std::chrono::high_resolution_clock::now();
        runWithConfig(config);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        std::cout << "\nDuration for cubeSize " << CuboidShape{config.cubeSize}.print() << " was " << duration <<" seconds" << std::endl;
    }
    // Run the tests
    
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}