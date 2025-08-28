#include "frontend/gui/windows/ConfigWindow.h"


ConfigWindow::ConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config)
    : Window(guiFrontend, width, height, name),
    config(config){}


void ConfigWindow::content(){
    config->showParameters(style);
}

