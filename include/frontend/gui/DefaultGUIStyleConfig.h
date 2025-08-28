#pragma once
#include "GUIStyleConfig.h"
#include <memory>
#include <unordered_map>
#include "frontend/gui/imguiWidget.h"



class DefaultGUIStyleConfig : public GUIStyleConfig{
public:
    DefaultGUIStyleConfig();

    void drawParameter(const ParameterDescription& param) override;

private:
    void registerDisplays();
    std::unordered_map<ParameterType, std::unique_ptr<imguiWidget>> styleGuide;
};