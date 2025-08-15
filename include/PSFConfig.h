#pragma once

#include <string>
#include <vector>
#include <iostream>
#include "../lib/nlohmann/json.hpp"

using json = nlohmann::json;

class PSFConfig {
public:
    PSFConfig() = default;
    virtual ~PSFConfig(){};
    virtual bool loadFromJSON(const json& jsonData) = 0;
    virtual void printValues() = 0;
    virtual std::string getName() = 0;
    
    template<typename T>
    T readParameter(const json& jsonData, std::string fieldName){
        if (jsonData.contains(fieldName)) {
            return jsonData.at(fieldName).get<T>();
        } else {
            throw std::runtime_error("[ERROR] Missing required parameter: " + fieldName);
        }
    }
    bool compareDim(const PSFConfig &other) {
        if(this->sizeX != other.sizeX || this->sizeY != other.sizeY || this->sizeZ != other.sizeZ) {
            std::cerr << "[ERROR] All PSFs have to be the same size" << std::endl;
            return false;
        }
        return true;
    }


    int sizeX;
    int sizeY;
    int sizeZ;

};

