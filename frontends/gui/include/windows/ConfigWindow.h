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
#include "windows/Window.h"
#include "dolphin/Config.h"


class ConfigBase {
public:
    ConfigBase(std::shared_ptr<Config> config) : config(config) {}

protected:
    std::shared_ptr<Config> config;
    
};

class ConfigContent : public Content, public ConfigBase {
public:
    ConfigContent(std::string name, std::shared_ptr<Config> config)
        : Content(name), ConfigBase(config) {}

protected:
    void content() override {
        showConfigParameters(*config, style);

    }
};

class ConfigWindow : public Window, public ConfigBase {
public:
    ConfigWindow(int width, int height, std::string name, std::shared_ptr<Config> config)
        : Window(width, height, name), ConfigBase(config) {}

protected:
    void content() override {
        showConfigParameters(*config, style);

    }
};