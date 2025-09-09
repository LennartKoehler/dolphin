#pragma once

#include "../lib/nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;


struct ReadWriteHelper {
    std::string jsonTag;
    std::function<void(const json&)> reader;
    std::function<void(ordered_json&)> writer;
};

class Config{
public:
    Config() = default;

    // IMPORTANT: in any derived class always run the copy constructor of parent so that at the end this is run:
    Config(const Config& other){
        parameters.clear();
    }


    virtual bool loadFromJSON(const json& jsonData) {
        for (auto& param : parameters) {
            try {
                param.reader(jsonData);
            } catch (const std::exception& e) {
                std::cerr << "Failed to load " << param.jsonTag << ": " << e.what() << std::endl;
                return false;
            }
        }
        return true;
    }

    virtual ordered_json writeToJSON() {
        ordered_json result;
        for (auto& param : parameters) {
            param.writer(result);
        }
        return result;
    }

    virtual void printValues(){
        json temp = writeToJSON(); // does this work?
        temp.dump();
    }

protected:
    virtual void registerAllParameters() = 0;



    template<typename T>
    void registerParameter(const std::string& jsonTag, T& field, bool isOptional) {
        ReadWriteHelper param;
        param.jsonTag = jsonTag;
        
        // Reader lambda
        param.reader = [&field, jsonTag, isOptional](const json& jsonData) {
            if (jsonData.contains(jsonTag)) {
                field = jsonData.at(jsonTag).get<T>();
            }
            else if(!isOptional){
                throw std::runtime_error("[ERROR] Missing required parameter: " + jsonTag);
            }
        };
        
        // Writer lambda
        param.writer = [&field, jsonTag](ordered_json& jsonData) {
            jsonData[jsonTag] = field;
        };
        
        parameters.push_back(std::move(param));
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



    std::vector<ReadWriteHelper> parameters;

};