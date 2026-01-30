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

#include "nlohmann/json.hpp"
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <array>

using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

enum class ParameterType{
 Float, Int, String, VectorInt, Bool, VectorString, FilePath, RangeMap, DeconvolutionConfig, IntArray3
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
    size_t size = 0;
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
    Config(){
    }

    virtual std::string getName() const = 0;


    virtual bool loadFromJSON(const json& jsonData);
    virtual json writeToJSON() const ;

    void printValues() const;

    bool logUnvalidParameters(const json& jsonData) const ;


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



    static json loadJSONFile(const std::string& filePath);

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
                // case ParameterType::VectorInt:
                //     auto* data = static_cast<int*>(param.value);
                //     std::vector<int> vec(data, data + param.size);
                //     visitor.template operator()<std::vector<int>>(vec, param);
                //     break;
                
                case ParameterType::IntArray3:
                    // int* intdata = static_cast<int*>(param.value);
                    visitor.template operator()<std::array<int, 3>>(*reinterpret_cast<std::array<int, 3>*>(param.value), param);
                    break;
            }
        }
        return handled;
    }

    std::vector<ConfigParameter> parameters;
    std::vector<ConfigParameter>& getParameters() { return parameters;};


};
