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

#pragma once
#include "dolphin/Dolphin.h"
#include "dolphin/frontend/IFrontend.h"
#include "GUIStyleConfig.h"


class GLFWwindow;
class MainWindow;

class GUIFrontend : public IFrontend{
public:
    GUIFrontend(Dolphin* dolphin);
    void run() override;

    std::unique_ptr<PSFGenerationResult> generatePSF(std::shared_ptr<PSFConfig> config);
    std::future<std::unique_ptr<DeconvolutionResult>> deconvolve(std::shared_ptr<SetupConfig> config, std::shared_ptr<DeconvolutionConfig> deconvConfig);
 
    double mainScale;

private:
    int init();
    void initWindows();
    void update();
    void cleanup();
    void render();

    std::string getExecutableDirectory();

    GLFWwindow* glfwWindow;
    int width = 2500;
    int height = 1400;
    std::shared_ptr<GUIStyleConfig> style;
    std::shared_ptr<MainWindow> mainWindow;
    std::string output_path_;
};