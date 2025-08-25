#include "DefaultGUIStyleConfig.h"
#include "imguiDisplayer.h"
#include <memory>

DefaultGUIStyleConfig::DefaultGUIStyleConfig(){
    registerDisplays();
}

void DefaultGUIStyleConfig::drawParameter(const ParameterDescription& param){
    if (auto it = styleGuide.find(param.type); it != styleGuide.end()) {
        it->second->display(param);
    } else {
        ImGui::Text("No widget for type!");
    }
}


void DefaultGUIStyleConfig::registerDisplays(){
    styleGuide[ParameterType::Int] = std::make_unique<imguiSliderInt>();
    styleGuide[ParameterType::String] = std::make_unique<imguiInputString>();
    styleGuide[ParameterType::Double] = std::make_unique<imguiSliderDouble>();
}