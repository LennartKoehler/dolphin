#include <iostream>
#include "Dolphin.h"
#include "frontend/SetupConfig.h"



void runLabeledImageTest(){
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    SetupConfig config = SetupConfig::createFromJSONFile("labeledImage/labeled_image_test.json"); 

    DeconvolutionRequest request(std::make_shared<SetupConfig>(config));
    
    dolphin->deconvolve(request);
}


int main() {
    std::cout << "=== DOLPHIN Labeled Image Test ===" << std::endl;
    
    // Run the tests
    runLabeledImageTest();
    
    std::cout << "\n=== Test completed ===" << std::endl;
    return 0;
}