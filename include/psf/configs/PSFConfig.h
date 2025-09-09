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


    int sizeX = 20;
    int sizeY = 20;
    int sizeZ = 10;
    double NA = 1.0;
    double resLateral_nm = 200;
    double resAxial_nm = 200;
    std::string psfModelName;

private:
    virtual void registerAllParameters() override;
};

