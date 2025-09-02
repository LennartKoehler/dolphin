#pragma once
#include "frontend/gui/windows/Window.h"
#include "frontend/gui/UIConfig.h"


class ConfigBase {
public:
    ConfigBase(std::shared_ptr<UIConfig> config) : config(config) {}

protected:
    std::shared_ptr<UIConfig> config;
    
};

class ConfigContent : public Content, public ConfigBase {
public:
    ConfigContent(std::string name, std::shared_ptr<UIConfig> config)
        : Content(name), ConfigBase(config) {}

protected:
    void content() override {
        config->showParameters(style);

    }
};

class ConfigWindow : public Window, public ConfigBase {
public:
    ConfigWindow(int width, int height, std::string name, std::shared_ptr<UIConfig> config)
        : Window(width, height, name), ConfigBase(config) {}

protected:
    void content() override {
        config->showParameters(style);

    }
};