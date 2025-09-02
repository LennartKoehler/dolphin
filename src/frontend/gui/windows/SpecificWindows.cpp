#include "frontend/gui/windows/SpecificWindows.h"
#include "frontend/gui/windows/FunctionContent.h"
#include "frontend/gui/GUIFrontend.h"

#include "frontend/gui/uipsf/UIConfigPSFGaussian.h"
#include "frontend/gui/uipsf/UIConfigPSFGibsonLanni.h"
#include "frontend/gui/UIDeconvolutionConfig.h"
#include "frontend/gui/UISetupConfig.h"

#include "imgui.h"

PSFMainWindow::PSFMainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name)
    : guiFrontend(guiFrontend),
    Window(width, height, name){


        std::shared_ptr<UIConfigPSFGaussian> gauss = std::make_shared<UIConfigPSFGaussian>();
        std::shared_ptr<ConfigWindow> gausswindow = std::make_shared<ConfigWindow>(width, height, "Gaussian PSF", gauss);

        std::shared_ptr<UIConfigPSFGibsonLanni> gibsonlanni = std::make_shared<UIConfigPSFGibsonLanni>();
        std::shared_ptr<ConfigWindow> gibsonlanniwindow = std::make_shared<ConfigWindow>(width, height, "GibsonLanni PSF", gibsonlanni);


        std::shared_ptr<FunctionContent> plot = std::make_shared<FunctionContent>("PSF", [guiFrontend](){guiFrontend->generatePSF();});
        plot->activate();
        std::shared_ptr<Window> plotWindow = std::make_shared<Window>(4000, 7000, "PSF Plot");
        plotWindow->addChild(plot);
        addChild(plotWindow);
    }

void PSFMainWindow::content(){
    Window::content();
    // if (ImGui::Button("Generate PSF")){
    //     plotWindow->activate();
    // }
}


DeconvolutionMainWindow::DeconvolutionMainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name)
    : guiFrontend(guiFrontend),
    Window(width, height, name){
        std::shared_ptr<UIDeconvolutionConfig> deconv = std::make_shared<UIDeconvolutionConfig>();
        std::shared_ptr<ConfigContent> deconvwindow = std::make_shared<ConfigContent>("Deconvolution Config", deconv);
        deconvwindow->activate();

        std::shared_ptr<UISetupConfig> setup = std::make_shared<UISetupConfig>();
        std::shared_ptr<ConfigContent> setupwindow = std::make_shared<ConfigContent>("Setup Config", setup);
        setupwindow->activate();

        addChild(deconvwindow);
        addChild(setupwindow);
    }

void DeconvolutionMainWindow::show(){
    if (active){
        startWindow();
        if (ImGui::BeginTable("Split", 2, ImGuiTableFlags_Resizable)) 
        {
            ImGui::TableNextColumn();
            
            ImGui::Text("Main Setup");
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
            getChild("Setup Config")->show();
        
            ImGui::TableNextColumn();

            ImGui::Text("Deconvolution Parameters");
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
            getChild("Deconvolution Config")->show();


            ImGui::EndTable();
        }
        endWindow();
    }
}

        // std::shared_ptr<UIConfigPSFGaussian> gaussian = std::make_shared<UIConfigPSFGaussian>();
        // std::shared_ptr<BackendConfigWindow> gaussianwindow = std::make_shared<BackendConfigWindow>(this, (int)(1280 * guiFrontend->mainScale), (int)(800 * guiFrontend->mainScale), "Gaussian PSF Config", gaussian);
        
        // std::shared_ptr<UIConfigPSFGibsonLanni> gibsonLanni = std::make_shared<UIConfigPSFGibsonLanni>();
        // std::shared_ptr<BackendConfigWindow> gibsonwindow = std::make_shared<BackendConfigWindow>(this, (int)(1280 * guiFrontend->mainScale), (int)(800 * guiFrontend->mainScale), "Gibson Lanni PSF Config", gibsonLanni);

        // std::shared_ptr<ButtonContent> psfButton = std::make_shared<ButtonContent>("run", [guiFrontend](){guiFrontend->generatePSF();});
        // psfButton->activate();

        // std::shared_ptr<FunctionContent> plot = std::make_shared<FunctionContent>("PSF", [guiFrontend](){guiFrontend->generatePSF();});
        // plot->activate();
        // plotWindow = std::make_shared<Window>(4000, 7000, "PSF Plot");
        // plotWindow->addChild(plot);
        // addChild(plotWindow);