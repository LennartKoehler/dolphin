#pragma once
#include <string>
#include <unordered_map>
#include <imgui.h>
#include "Config.h"

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
            if (param.type == ParameterType::VectorString){
                SelectionHelper<std::string>* helper = new SelectionHelper<std::string>();
                helper->field = reinterpret_cast<std::string*>(param.value);
                helper->selection = reinterpret_cast<std::vector<std::string>*>(param.selection);
                param.value = reinterpret_cast<void*>(helper);
                style->drawParameter(param);
            }
            else{
                style->drawParameter(param);
            }
            return true;
        });

    ImGui::PopItemWidth();
}

