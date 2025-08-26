#include "Dolphin.h"
#include "frontend/gui/GUIFrontend.h"
#include "frontend/CLIFrontend.h"

int main(int argc, char** argv) {
    std::cout << "[Start DeconvTool]" << std::endl;
    bool cli = false;
    ConfigManager* configManager= new ConfigManager();
    Dolphin dolphin{};
    IFrontend* frontend;

    if (cli){
        frontend = new CLIFrontend{configManager, argc, argv};
        dolphin.init(configManager);
        dolphin.run();
    }
    else{
        frontend = new GUIFrontend{configManager, dolphin};
        frontend->run();
    }
    

}


