#include "frontend/gui/windows/SpecificWindows.h"
#include "frontend/gui/windows/FunctionContent.h"
#include "frontend/gui/GUIFrontend.h"

#include "frontend/gui/uipsf/UIConfigPSFGaussian.h"
#include "frontend/gui/uipsf/UIConfigPSFGibsonLanni.h"
#include "frontend/gui/UIDeconvolutionConfig.h"
#include "frontend/gui/UISetupConfig.h"

#include "imgui.h"


// big mess :)
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
            [this, guiFrontend, gauss, gausswindow]() {
                auto config = gauss->getConfig();
                std::unique_ptr<PSFGenerationResult> result = guiFrontend->generatePSF(config);
                this->psfPath = result->generated_path;
                gausswindow->deactivate();
                showPSFWindow = true;
            });
        gaussButton->activate();

        std::shared_ptr<ButtonContent> gibsonButton = std::make_shared<ButtonContent>("Generate Gibson-Lanni PSF", 
            [this, guiFrontend, gibsonlanni, gibsonlanniwindow]() {
                auto config = gibsonlanni->getConfig();
                std::unique_ptr<PSFGenerationResult> result = guiFrontend->generatePSF(config);
                this->psfPath = result->generated_path;
                gibsonlanniwindow->deactivate();
                showPSFWindow = true;
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

    if (showPSFWindow){
        ImGui::Begin("PSF saved under", &showPSFWindow);
        
        // Create a buffer for the path (InputText needs char array)
        static char pathBuffer[512];
        std::strncpy(pathBuffer, psfPath.c_str(), sizeof(pathBuffer) - 1);
        pathBuffer[sizeof(pathBuffer) - 1] = '\0';
        
        // Read-only input text that user can select and copy
        ImGui::InputText("Path", pathBuffer, sizeof(pathBuffer), 
                        ImGuiInputTextFlags_ReadOnly);
        
        // Optional: Add a copy button
        ImGui::SameLine();
        if (ImGui::Button("Copy")) {
            ImGui::SetClipboardText(psfPath.c_str());
        }
        
        ImGui::End();
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

        std::shared_ptr<PSFMainWindow> psfconfigwindow = std::make_shared<PSFMainWindow>(guiFrontend, width, height, "Generate PSF");

        std::shared_ptr<ButtonContent> startDeconvolutionButton = std::make_shared<ButtonContent>("Start Deconvolution", 
            [guiFrontend, setup, deconv, psfconfigwindow]() {
                auto setupconfig = setup->getConfig();
                auto deconvconfig = deconv->getConfig();
                if (setupconfig->psfFilePath.empty() && !psfconfigwindow->psfPath.empty()){
                    setupconfig->psfFilePath = psfconfigwindow->psfPath;
                }
                setupconfig->deconvolutionConfig = deconvconfig;
                guiFrontend->deconvolve(setupconfig);
            });
        startDeconvolutionButton->activate();

        addChild(startDeconvolutionButton);
        addChild(psfconfigwindow);
        addChild(deconvwindow);
        addChild(setupwindow);
    }

void DeconvolutionMainWindow::show(){
    if (active){
        startWindow();
        if (ImGui::BeginTable("Split", 2, ImGuiTableFlags_Resizable)) 
        {

            //Setup
            ImGui::TableNextColumn();
            
            ImGui::Text("Main Setup");
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
            getChild("Setup Config")->show();
            getChild("Generate PSF")->show();

            // PSF
            if (ImGui::Button("Generate PSF")){
                getChild("Generate PSF")->activate();
            }
            ImGui::SameLine();
            ImGui::Text("Select the PSF Model");


            // Deconvolution
            ImGui::TableNextColumn();

            ImGui::Text("Deconvolution Parameters");
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
            getChild("Deconvolution Config")->show();


            ImGui::EndTable();
            ImGui::Dummy(ImVec2(0.0f, 20.0f));

            //center
            float windowWidth = ImGui::GetWindowWidth();
            float buttonWidth = 200.0f; // Estimate button width
            ImGui::SetCursorPosX((windowWidth - buttonWidth) * 0.5f);
            getChild("Start Deconvolution")->show();
        }
        endWindow();
    }
}

