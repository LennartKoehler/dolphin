#pragma once

#include <memory>
#include <vector>
#include "GUIStyleConfig.h"

class UIConfig{
public:
    void showParameters(std::shared_ptr<GUIStyleConfig> style);
protected:
    std::vector<ParameterDescription> parameters;

};