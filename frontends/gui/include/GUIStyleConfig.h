/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

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
    virtual ~GUIStyleConfig() = default;

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

