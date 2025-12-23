#pragma once
#include "BackendConfigWindow.h"

class PSFMainWindow : public Window{
public:
    PSFMainWindow(GUIFrontend* frontend, int width, int height, std::string name);
    virtual void content() override;

    std::string psfPath;

private:
    bool showPSFWindow = false;
    GUIFrontend* guiFrontend;
};

class DeconvolutionMainWindow : public Window{
public:
    DeconvolutionMainWindow(GUIFrontend* frontend, int width, int height, std::string name);
    virtual void show() override;


private:
    GUIFrontend* guiFrontend;
};