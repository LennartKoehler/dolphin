/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "frontend/gui/windows/MainWindow.h"
#include "imgui.h"
#include <string>
#include <stdexcept>
#include "frontend/gui/GUIFrontend.h"
#include "frontend/gui/windows/SpecificWindows.h"


MainWindow::MainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name)
    : Window(width, height, name),
    guiFrontend(guiFrontend)
    {
    std::shared_ptr<DeconvolutionMainWindow> deconvmain = std::make_shared<DeconvolutionMainWindow>(guiFrontend, width, height, "Deconvolution");
    std::shared_ptr<PSFMainWindow> psfmain = std::make_shared<PSFMainWindow>(guiFrontend, width, height, "PSF Generator");
    addChild(deconvmain);
    addChild(psfmain);

}

void MainWindow::startWindow(){
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0, 0));                        // top-left corner
    ImGui::SetNextWindowSize(io.DisplaySize);                      // full display size

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDecoration   // no title bar, resize, or close
                                  | ImGuiWindowFlags_NoMove
                                  | ImGuiWindowFlags_NoBringToFrontOnFocus
                                  | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);      // remove rounding
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);    // remove border

    ImGui::Begin("Main Window", nullptr, window_flags);





}

void MainWindow::endWindow(){
    ImGui::PopStyleVar(2);
    ImGui::End();
}

void MainWindow::content(){
    ImGui::Text("Deconvolution with Optimized Local PSFs for High-speed Image recoNstruction (DOLPHIN)");
    ImGui::Text("is a C++ command-line tool designed for deconvolution of microscopy images.");
    ImGui::Text("It supports multiple deconvolution algorithms and allows the generation and use of synthetic Point Spread Functions (PSF).");
    ImGui::Text("The tool is intended for users familiar with image processing and deconvolution techniques.");
    ImGui::TextLink("https://github.com/LennartKoehler/dolphin");


    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    if(ImGui::Button("Deconvolution")){
        getChild("Deconvolution")->activate();
    }
    if(ImGui::Button("PSF Generator")){
        getChild("PSF Generator")->activate();
    }
}


