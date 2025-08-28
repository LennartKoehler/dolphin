#include "frontend/gui/UIConfig.h"


void UIConfig::showParameters(std::shared_ptr<GUIStyleConfig> style){
    for (const auto parameter : parameters){
        style->drawParameter(parameter);
    }
}