#include "frontend/gui/UIConfig.h"
#include "imgui.h"

void UIConfig::showParameters(std::shared_ptr<GUIStyleConfig> style){
    ImGui::PushItemWidth(350.0f);

    for (const auto parameter : parameters){
        style->drawParameter(parameter);
    }
    ImGui::PopItemWidth();
}