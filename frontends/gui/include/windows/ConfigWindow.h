#pragma once
#include "windows/Window.h"
#include "Config.h"


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