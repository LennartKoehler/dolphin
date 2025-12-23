#include "Dolphin.h"
#include "GUIFrontend.h"

int main(int argc, char** argv) {
    std::cout << "[Start Dolphin GUI]" << std::endl;
    
    // Initialize Dolphin
    Dolphin* dolphin = new Dolphin();
    dolphin->init();
    
    // Create GUI frontend and run
    GUIFrontend* frontend = new GUIFrontend(dolphin);
    frontend->run();
    
    // Cleanup
    delete frontend;
    delete dolphin;
    
    std::cout << "[End Dolphin GUI]" << std::endl;
    return 0;
}


