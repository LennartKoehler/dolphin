#pragma once

#include "ConfigWindow.h"

class PSFConfigWindow : public ConfigWindow{
public:
    PSFConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config);
    virtual void content() override;

private:
    std::shared_ptr<Window> plotWindow;
};