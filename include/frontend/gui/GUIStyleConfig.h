#pragma once
#include <string>
#include <unordered_map>



enum class ParameterType{
 Double, Int, String, VectorInt, Bool, VectorString, FilePath
};


class ParameterIDGenerator {
public:
    static int getNextID() {
        static int currentID = 0;
        return ++currentID;
    }
};

struct ParameterDescription {
    std::string name;
    ParameterType type;
    void* ptr;
    double minVal, maxVal;
    int ID = ParameterIDGenerator::getNextID();
    
};


class GUIStyleConfig{
public:
    GUIStyleConfig() = default;
    virtual ~GUIStyleConfig(){}

    virtual void drawParameter(const ParameterDescription& param) = 0;

};