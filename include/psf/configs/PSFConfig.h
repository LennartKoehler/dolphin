#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "Config.h"

class PSFConfig : public Config{
public:
    PSFConfig();
    virtual ~PSFConfig(){};
    PSFConfig(const PSFConfig& other);
    virtual std::string getName() const = 0;
    static std::shared_ptr<PSFConfig> createFromJSON(const json& jsonData);

    bool compareDim(const PSFConfig &other);


    std::string psfModelName;
    std::string ID;

    int sizeX = 20;
    int sizeY = 20;
    int sizeZ = 10;
    float NA = 1.0;
    float resLateral_nm = 200;
    float resAxial_nm = 200;

protected:
    void registerAllParameters();


};

