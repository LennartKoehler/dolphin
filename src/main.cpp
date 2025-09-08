#include "Dolphin.h"
#include "frontend/gui/GUIFrontend.h"
#include "frontend/CLIFrontend.h"

int main(int argc, char** argv) {
    std::cout << "[Start Dolphin]" << std::endl;
    Dolphin* dolphin = new Dolphin;
    dolphin->init();
    IFrontend* frontend;

    if (argc > 1){
        frontend = new CLIFrontend{dolphin, argc, argv};
    }
    else{
        frontend = new GUIFrontend{dolphin};
    }
    frontend->run();
    std::cout << "[End Dolphin]" << std::endl;

}


