#pragma once
#include <unordered_map>
#include <string>
#include <functional>
#include "frontend/gui/DefaultGUIStyleConfig.h"
class GLFWwindow;
class GUIFrontend;

class Content{
public:
    Content(std::string name);
    virtual ~Content() = default;
    virtual void show();
    virtual void content() = 0;
    bool isActive();
    void activate();
    void deactivate();
    std::string getName();

protected:
    bool active = false;
    std::string name;

};

class Window : public Content{
public:
    Window(GUIFrontend* guiFrontend, int width, int height, std::string name);
    virtual ~Window(){}
    virtual void show() override;
    void addChild(std::shared_ptr<Content> window);
    void setStyle(std::shared_ptr<GUIStyleConfig> style);

protected:
    int width;
    int height;
    GUIFrontend* guiFrontend;
    std::shared_ptr<GUIStyleConfig> style = std::make_shared<DefaultGUIStyleConfig>();
    virtual void content();
    virtual void startWindow();
    virtual void endWindow();

    void deactivateAllChildren();
    void showChildren();
    std::vector<std::shared_ptr<Content>> children;


};


