#pragma once
#include "GUIStyleConfig.h"
#include <memory>
#include <unordered_map>
#include "frontend/gui/imguiWidget.h"
#include <functional>


class DefaultGUIStyleConfig : public GUIStyleConfig{
public:
    DefaultGUIStyleConfig();

    void drawParameter(const ParameterDescription& param) override;

private:
    void registerDisplays();
    std::unordered_map<ParameterType, std::function<std::unique_ptr<imguiWidget>()>> widgetFactory;
    mutable std::unordered_map<int, std::unique_ptr<imguiWidget>> widgetCache;

};