#include "frontend/gui/windows/PSFConfigWindow.h"
#include "frontend/gui/windows/FunctionContent.h"
#include "frontend/gui/GUIFrontend.h"
#include "imgui.h"


PSFConfigWindow::PSFConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<UIConfig> config)
    : ConfigWindow(guiFrontend, width, height, name, config){
        std::shared_ptr<FunctionContent> plot = std::make_shared<FunctionContent>("PSF", [guiFrontend](){guiFrontend->generatePSF();});
        plot->activate();
        plotWindow = std::make_shared<Window>(guiFrontend, 4000, 7000, "PSF Plot");
        plotWindow->addChild(plot);
        addChild(plotWindow);
    }

void PSFConfigWindow::content(){
    config->showParameters(style);
    if (ImGui::Button("Generate PSF")){
        plotWindow->activate();
    }
}