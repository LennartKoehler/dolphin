#include "frontend/gui/windows/BackendConfigWindow.h"
#include "frontend/gui/windows/FunctionContent.h"
#include "frontend/gui/GUIFrontend.h"

#include "frontend/gui/windows/BackendConfigWindow.h"
#include "frontend/gui/uipsf/UIConfigPSFGaussian.h"
#include "frontend/gui/uipsf/UIConfigPSFGibsonLanni.h"

#include "imgui.h"


BackendConfigWindow::BackendConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config)
    : ConfigWindow(width, height, name, config),
    guiFrontend(guiFrontend){
    }

void BackendConfigWindow::content(){
    config->showParameters(style);

}