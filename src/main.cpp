#include "Dolphin.h"
#include "frontend/gui/GUIFrontend.h"
#include "frontend/CLIFrontend.h"

int main(int argc, char** argv) {
    std::cout << "[Start Dolphin]" << std::endl;
    SetupConfig* setupConfig= new SetupConfig();
    Dolphin dolphin{};
    IFrontend* frontend;

    if (argc > 1){
        frontend = new CLIFrontend{setupConfig, argc, argv};
        frontend->run();
        dolphin.init(setupConfig);
        dolphin.run();
    }
    else{
        frontend = new GUIFrontend{setupConfig, dolphin};
        frontend->run();
    }
    
    std::cout << "[End Dolphin]" << std::endl;

}


