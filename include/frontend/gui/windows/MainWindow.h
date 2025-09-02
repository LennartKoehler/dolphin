#pragma once

#include "Window.h"


class MainWindow : public Window{
public:
    MainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name);

    // void show() override;

private:
    void content() override;
    void startWindow() override;
    void endWindow() override;
    GUIFrontend* guiFrontend;
};