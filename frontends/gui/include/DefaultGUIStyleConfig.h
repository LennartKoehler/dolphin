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
#include "GUIStyleConfig.h"
#include <memory>
#include <unordered_map>
#include "imguiWidget.h"
#include <functional>


class DefaultGUIStyleConfig : public GUIStyleConfig{
public:
    DefaultGUIStyleConfig();

    void drawParameter(const ConfigParameter& param) override;

private:
    void registerDisplays();
    std::unordered_map<ParameterType, std::function<std::unique_ptr<imguiWidget>()>> widgetFactory;
    mutable std::unordered_map<int, std::unique_ptr<imguiWidget>> widgetCache;

};