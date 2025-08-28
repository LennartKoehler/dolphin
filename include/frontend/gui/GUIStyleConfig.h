#pragma once
#include <string>
#include <unordered_map>



enum class ParameterType{
 Double, Int, String, VectorInt, Bool
};



struct ParameterDescription {
    std::string name;
    ParameterType type;
    void* ptr;
    double minVal, maxVal;
};


class GUIStyleConfig{
public:
    GUIStyleConfig() = default;
    virtual ~GUIStyleConfig(){}

    virtual void drawParameter(const ParameterDescription& param) = 0;

};