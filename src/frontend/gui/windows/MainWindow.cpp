#include "frontend/gui/windows/MainWindow.h"
#include "imgui.h"
#include <string>
#include <stdexcept>
#include "frontend/gui/GUIFrontend.h"



MainWindow::MainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name)
    : Window(guiFrontend, width, height, name)
    {

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
    addConfigButtons();
}

void MainWindow::addConfigButtons(){

    for (auto it : children){
        if (it->getName().find("Config") != std::string::npos){
            if (ImGui::Button(it->getName().c_str())){
                it->activate();
            }
        }
    }
}

