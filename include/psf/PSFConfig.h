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
    PSFConfig(const PSFConfig& other);
    virtual bool loadFromJSON(const json& jsonData) = 0;
    virtual bool loadFromJSONBase(const json& jsonData);
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

    template<typename T>
    void readParameterOptional(const json& jsonData, std::string fieldName, T& field){
        if (jsonData.contains(fieldName)) {
            field = jsonData.at(fieldName).get<T>();
        }
    }


    bool compareDim(const PSFConfig &other);


    int sizeX;
    int sizeY;
    int sizeZ;
    double NA;
    double resLateral_nm;
    double resAxial_nm;
};

