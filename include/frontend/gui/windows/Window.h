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
    bool isActive();
    void activate();
    void deactivate();
    std::string getName();
    void addChild(std::shared_ptr<Content> window);
    void setStyle(std::shared_ptr<GUIStyleConfig> style);

protected:
    std::shared_ptr<Content> getChild(const std::string& name);
    virtual void content() = 0;
    void setParent(Content* parent);
    void deactivateAllChildren();
    void showChildren();
    std::unordered_map<std::string, std::shared_ptr<Content>> children;
    Content* parent;
    bool active = false;
    std::string name;
    GUIFrontend* guiFrontend;
    std::shared_ptr<GUIStyleConfig> style = std::make_shared<DefaultGUIStyleConfig>();


};

class Window : public Content{
public:
    Window(int width, int height, std::string name);
    virtual ~Window(){}
    virtual void show() override;

protected:
    int width;
    int height;
    virtual void content();
    virtual void startWindow();
    virtual void endWindow();




};


