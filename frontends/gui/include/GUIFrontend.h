#pragma once
#include "dolphin/Dolphin.h"
#include "dolphin/frontend/IFrontend.h"
#include "GUIStyleConfig.h"


class GLFWwindow;
class Window;
class MainWindow;

class GUIFrontend : public IFrontend{
public:
    GUIFrontend(Dolphin* dolphin);
    void run() override;

    std::unique_ptr<PSFGenerationResult> generatePSF(std::shared_ptr<PSFConfig> config);
    std::future<std::unique_ptr<DeconvolutionResult>> deconvolve(std::shared_ptr<SetupConfig> config);
 
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