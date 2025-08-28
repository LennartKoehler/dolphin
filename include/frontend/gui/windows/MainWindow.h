#pragma once

#include "Window.h"


class MainWindow : public Window{
public:
    MainWindow(GUIFrontend* frontend, int width, int height, std::string name);

    // void show() override;

private:
    void addConfigButtons();
    void content() override;
    void startWindow() override;
    void endWindow() override;
};