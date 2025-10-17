#pragma once

#include "../lib/nlohmann/json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

enum class ParameterType{
 Float, Int, String, VectorInt, Bool, VectorString, FilePath, RangeMap, DeconvolutionConfig
};

class ParameterIDGenerator {
public:
    static int getNextID() {
        static int currentID = 0;
        return ++currentID;
    }
};



struct ConfigParameter{
    ParameterType type;
    void* value;
    const char* name;
    bool optional;
    const char* jsonTag;
    const char* cliFlag;
    const char* cliDesc;
    bool cliRequired;
    bool hasRange;
    double minVal;
    double maxVal;
    void* selection;
    int ID = ParameterIDGenerator::getNextID();

};


// The idea of everything inheriting from this config, is that the child registers all its parameters with everything there is to it
// Each parameter should directly provide everything there ever is to know about it
// Then whereever its used, like in the frontend, they can be accessed through the visitParams. Whichs provides an option for specialtypes with SpecialVisistor
// This way I should only have to edit the config i want to change (like add a parameter) and all frontends and whatever directly know what to do with it
// and the parameter can directly be used in the code (unless its a new ParameterType). Although a nice idea this makes the use of visitParams a bit weird
// and not typesafe (void*)

class Config{
    using ParamVisitor = std::function<void(ConfigParameter)>;

public:
    Config() = default;


    virtual bool loadFromJSON(const json& jsonData){
        bool success = true;
        try{
            visitParams([&jsonData]<typename T>(T& value, ConfigParameter& param) {
                if (jsonData.contains(param.jsonTag)) {
                    value = jsonData.at(param.jsonTag).get<T>();
                }
            });
        }
        catch (const std::exception){
            std::cout << "Error in reading config from json" << std::endl;
            success = false;
        }

        return success;
    }
    virtual json writeToJSON(){
        json jsonData;
        visitParams([&jsonData]<typename T>(T& value, ConfigParameter& param) {
            if (!param.value)
                return;
            jsonData[param.jsonTag] = value;

        });

        return jsonData;
    }

    void printValues(){
        visitParams([]<typename T>(T& value, ConfigParameter& param){
            std::cout << param.name << ": " << value << std::endl;
        });
    }

    template<typename Visitor>
    void visitParams(Visitor&& visitor){
        visitParams(std::forward<Visitor>(visitor), [](ConfigParameter&){return false;});
    }

    template<typename Visitor>
    void visitParams(Visitor&& visitor, std::function<bool(ConfigParameter&)> specialVisitor){
        for (auto& param: parameters){
            visitParamValue(param, std::forward<Visitor>(visitor), specialVisitor);
        }
    };

protected:



    static json loadJSONFile(const std::string& filePath) {
        std::ifstream configFile(filePath);
        if (!configFile.is_open()) {
            throw std::runtime_error("Could not open config file: " + filePath);
        }
        
        json jsonData;
        configFile >> jsonData;
        return jsonData;
    }


    template<typename Visitor>
    bool visitParamValue(ConfigParameter& param, Visitor&& visitor, std::function<bool(ConfigParameter&)> specialVisitor) {
        bool handled = specialVisitor(param);
        if (!handled){

            switch (param.type) {
                case ParameterType::Int:
                    visitor.template operator()<int>(*reinterpret_cast<int*>(param.value), param);
                    break;
                case ParameterType::Float:
                    visitor.template operator()<float>(*reinterpret_cast<float*>(param.value), param);
                    break;
                case ParameterType::Bool:
                    visitor.template operator()<bool>(*reinterpret_cast<bool*>(param.value), param);
                    break;
                case ParameterType::String:
                case ParameterType::FilePath:
                case ParameterType::VectorString:
                    visitor.template operator()<std::string>(*reinterpret_cast<std::string*>(param.value), param);
                    break;

            }
        }
        return handled;
    }

    std::vector<ConfigParameter> parameters;
    std::vector<ConfigParameter>& getParameters() { return parameters;};
};