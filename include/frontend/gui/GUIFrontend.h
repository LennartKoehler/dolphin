#pragma once
#include "Dolphin.h"
#include "frontend/IFrontend.h"
#include "frontend/gui/GUIStyleConfig.h"


class GLFWwindow;
class Window;
class MainWindow;

class GUIFrontend : public IFrontend{
public:
    GUIFrontend(SetupConfig* config, Dolphin& dolphin);
    void run() override;

    void runDolphin();
    void initDolphin();
    void generatePSF();
private:
    int init();
    void initWindows();
    void update();
    void cleanup();
    void render();

    // void showActiveWindows();
    // void addWindow(std::string windowName, std::shared_ptr<Window> window);
    // std::shared_ptr<Window> getWindow(std::string windowName);

    GLFWwindow* glfwWindow;
    int width = 2500;
    int height = 1400;
    double mainScale;
    std::shared_ptr<GUIStyleConfig> style;
    std::shared_ptr<MainWindow> mainWindow;
    // std::unordered_map<std::string, std::shared_ptr<Window>> configWindows;
    Dolphin& dolphin; // does it need to know about dlophin?
};