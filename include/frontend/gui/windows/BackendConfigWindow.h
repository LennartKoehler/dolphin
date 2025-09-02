#pragma once

#include "ConfigWindow.h"

class BackendConfigWindow : public ConfigWindow{
public:
    BackendConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config);
    virtual void content() override;

private:
    std::shared_ptr<Window> plotWindow;
    GUIFrontend* guiFrontend;
};