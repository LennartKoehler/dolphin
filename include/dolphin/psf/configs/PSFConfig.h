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

