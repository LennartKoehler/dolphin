#pragma once

#include "../lib/nlohmann/json.hpp"
using json = nlohmann::json;

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

};