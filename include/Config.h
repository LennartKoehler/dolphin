#pragma once

#include "../lib/nlohmann/json.hpp"
#include <fstream>
using json = nlohmann::json;

// object representation of the json configs
class Config{
public:
    virtual bool loadFromJSON(const json& jsonData) = 0;


protected:
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

    static json loadJSONFile(const std::string& filePath) {
        std::ifstream configFile(filePath);
        if (!configFile.is_open()) {
            throw std::runtime_error("Could not open config file: " + filePath);
        }
        
        json jsonData;
        configFile >> jsonData;
        return jsonData;
    }

};