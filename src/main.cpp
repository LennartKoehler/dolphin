/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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


