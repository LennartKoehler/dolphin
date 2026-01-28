#pragma once
#include <string>
#include <unordered_map>
#include <imgui.h>
#include "dolphin/Config.h"

template <typename T>
struct SelectionHelper{
    T* field;
    std::vector<T>* selection;
};







class GUIStyleConfig{
public:
    GUIStyleConfig() = default;
    virtual ~GUIStyleConfig(){}

    virtual void drawParameter(const ConfigParameter& param) = 0;

};

static void showConfigParameters(Config& config, std::shared_ptr<GUIStyleConfig> style){
    ImGui::PushItemWidth(350.0f);

    config.visitParams(
        [style]<typename T>(T& value, ConfigParameter& param){},
        [style](ConfigParameter& param){

            style->drawParameter(param);
            return true;
        });

    ImGui::PopItemWidth();
}

