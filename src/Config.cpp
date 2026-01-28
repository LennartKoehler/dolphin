#include "dolphin/Config.h"

#include <spdlog/spdlog.h>

bool Config::loadFromJSON(const json& jsonData){
    bool success = true;
    try{
        // First, collect all valid json tags from registered parameters
        std::unordered_set<std::string> validTags;
        for (const auto& param : parameters) {
            if (param.jsonTag) {
                validTags.insert(param.jsonTag);
            }
        }
        
        // Check for unknown keys in JSON
        std::vector<std::string> unknownKeys;
        for (auto it = jsonData.begin(); it != jsonData.end(); ++it) {
            if (validTags.find(it.key()) == validTags.end()) {
                unknownKeys.push_back(it.key());
            }
        }
        
        // Report unknown keys
        if (!unknownKeys.empty()) {
            for (const auto& key : unknownKeys) {
                spdlog::get("config")->warn("({}) Unknown key in config: {}", this->getName(), key);
            }

        }
        
        visitParams([&jsonData]<typename T>(T& value, ConfigParameter& param) {
            if (jsonData.contains(param.jsonTag)) {
                value = jsonData.at(param.jsonTag).get<T>();
            }
        });
    }
    catch (const std::exception& e){
        spdlog::get("config")->info("({}) Error in reading config from json: {}", this->getName(), e.what());
        success = false;
    }

    return success;
}

json Config::writeToJSON(){
    json jsonData;
    visitParams([&jsonData]<typename T>(T& value, ConfigParameter& param) {
        if (!param.value)
            return;
        jsonData[param.jsonTag] = value;

    });

    return jsonData;
}

void Config::printValues(){
    visitParams([this]<typename T>(T& value, ConfigParameter& param){
        spdlog::get("config")->info("({}) {}: {}", this->getName(), param.name, value);
    });
}

json Config::loadJSONFile(const std::string& filePath) {
    std::ifstream configFile(filePath);
    if (!configFile.is_open()) {
        throw std::runtime_error("Could not open config file: " + filePath);
    }
    
    json jsonData;
    configFile >> jsonData;
    return jsonData;
}