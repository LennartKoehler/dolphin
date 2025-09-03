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

        addChild(gausswindow);
        addChild(gibsonlanniwindow);

        // Create separate buttons for each PSF type
        std::shared_ptr<ButtonContent> gaussButton = std::make_shared<ButtonContent>("Generate Gaussian PSF", 
            [guiFrontend, gauss]() {
                auto config = gauss->getConfig();
                guiFrontend->generatePSF(config);
            });
        gaussButton->activate();

        std::shared_ptr<ButtonContent> gibsonButton = std::make_shared<ButtonContent>("Generate Gibson-Lanni PSF", 
            [guiFrontend, gibsonlanni]() {
                auto config = gibsonlanni->getConfig();
                guiFrontend->generatePSF(config);
            });
        gibsonButton->activate();

        gausswindow->addChild(gaussButton);
        gibsonlanniwindow->addChild(gibsonButton);
    }

void PSFMainWindow::content(){
    ImGui::Text("Select the PSF Model");
    if (ImGui::Button("Gaussian PSF")){
        getChild("Gaussian PSF")->activate();
    }
    if (ImGui::Button("GibsonLanni PSF")){
        getChild("GibsonLanni PSF")->activate();
    }
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

        std::shared_ptr<PSFMainWindow> psfconfig = std::make_shared<PSFMainWindow>(guiFrontend, width, height, "Generate PSF");
        addChild(psfconfig);
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
            getChild("Generate PSF")->show();

            // PSF:
            ImGui::Text("Select the PSF Model");
            if (ImGui::Button("Generate PSF")){
                getChild("Generate PSF")->activate();
            }
            ImGui::TableNextColumn();

            ImGui::Text("Deconvolution Parameters");
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
            getChild("Deconvolution Config")->show();


            ImGui::EndTable();
        }
        endWindow();
    }
}

        // std::shared_ptr<UIConfigPSFGibsonLanni> gaussian = std::make_shared<UIConfigPSFGaussian>();
        // std::shared_ptr<BackendibsonLanniWindow> gaussianwindow = std::make_shared<BackendConfigWindow>(this, (int)(1280 * guiFrontend->mainScale), (int)(800 * guiFrontend->mainScale), "Gaussian PSF Config", gaussian);
        
        // std::shared_ptr<UIConfigPSFGibsonLanni> gibsonLanni = std::make_shared<UIConfigPSFGibsonLanni>();
        // std::shared_ptr<BackendConfigWindow> gibsonwindow = std::make_shared<BackendConfigWindow>(this, (int)(1280 * guiFrontend->mainScale), (int)(800 * guiFrontend->mainScale), "Gibson Lanni PSF Config", gibsonLanni);

        // std::shared_ptr<ButtonContent> psfButton = std::make_shared<ButtonContent>("run", [guiFrontend](){guiFrontend->generatePSF();});
        // psfButton->activate();

        // std::shared_ptr<FunctionContent> plot = std::make_shared<FunctionContent>("PSF", [guiFrontend](){guiFrontend->generatePSF();});
        // plot->activate();
        // plotWindow = std::make_shared<Window>(4000, 7000, "PSF Plot");
        // plotWindow->addChild(plot);
        // addChild(plotWindow);