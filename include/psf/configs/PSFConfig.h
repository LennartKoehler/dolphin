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


    int sizeX;
    int sizeY;
    int sizeZ;
    double NA;
    double resLateral_nm;
    double resAxial_nm;
    std::vector<int> psfCubeVec;
    std::vector<int> psfLayerVec;


};

