#include "GibsonLanniPSFGenerator.h"

std::string GibsonLanniPSFConfig::getName(){
    return this->psfModelName;
}

bool GibsonLanniPSFConfig::loadFromJSON(const json& jsonData){
    ti0 = readParameter<double>(jsonData, "ti0");
    // read all the necessary parameters....
    return true;
}

void GibsonLanniPSFConfig::printValues(){
    std::cout << "ti0" << ti0 << std::endl;// TODO maybe upgrade to c++23 and use std::print()
}