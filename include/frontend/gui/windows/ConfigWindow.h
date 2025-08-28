#pragma once
#include "frontend/gui/windows/Window.h"
#include "frontend/gui/UIConfig.h"


class ConfigWindow : public Window{
public:
    ConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config);
    // void show() override;
    

protected:
    std::shared_ptr<UIConfig> config;
    virtual void content() override;

};