#include "Dolphin.h"

int main(int argc, char** argv) {
    
    Dolphin* dolphin = new Dolphin();
    
    int status = dolphin->init(argc, argv);
    if (status != 0){return status;}
    dolphin->run();
}


