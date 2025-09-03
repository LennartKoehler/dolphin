#pragma once

#include <string>
#include <vector>
#include <iostream>

#include "Config.h"

class PSFConfig : public Config{
public:
    PSFConfig() = default;
    virtual ~PSFConfig(){};
    PSFConfig(const PSFConfig& other);
    virtual bool loadFromJSON(const json& jsonData);
    virtual void printValues() = 0;
    virtual std::string getName() const = 0;
    virtual bool loadFromJSONSpecific(const json& jsonData) = 0; // factory method
    static std::shared_ptr<PSFConfig> createFromJSON(const json& jsonData);

    bool compareDim(const PSFConfig &other);


    int sizeX = 20;
    int sizeY = 20;
    int sizeZ = 10;
    double NA = 1.0;
    double resLateral_nm = 200;
    double resAxial_nm = 200;
    std::vector<int> psfCubeVec;
    std::vector<int> psfLayerVec;


};

