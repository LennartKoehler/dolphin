#include "frontend/gui/windows/FunctionContent.h"
#include "frontend/gui/GUIFrontend.h"

#include "frontend/gui/windows/BackendConfigWindow.h"


#include "imgui.h"


BackendConfigWindow::BackendConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<Config> config)
    : ConfigWindow(width, height, name, config),
    guiFrontend(guiFrontend){
    }

void BackendConfigWindow::content(){
    showConfigParameters(*config, style);

}