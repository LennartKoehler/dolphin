#include "Dolphin.h"

int main(int argc, char** argv) {
    
    Dolphin* dolphin = new Dolphin();
    
    dolphin->init(argc, argv);
    dolphin->run();
}


