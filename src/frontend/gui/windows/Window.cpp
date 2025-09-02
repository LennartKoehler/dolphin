#include "frontend/gui/windows/MainWindow.h"
#include "imgui.h"
#include <string>
#include <stdexcept>
#include "frontend/gui/GUIFrontend.h"
#include <GLFW/glfw3.h>

Content::Content(std::string name)
    :name(name){}

bool Content::isActive(){
    return active;
}

void Content::deactivate(){
    active = false;
}

void Content::activate(){
    active = true;
}
std::string Content::getName(){
    return name;
}

void Content::show(){
    if (active){
        content();
    }

}

void Content::setParent(Content* parent){
    this->parent = parent;
}
void Content::showChildren(){
    for (auto it : children){
        it.second->show();
    }
}

void Content::deactivateAllChildren(){
    for (auto it : children){
        it.second->deactivate();
    }
}

std::shared_ptr<Content> Content::getChild(const std::string& name){
    auto it = children.find(name);
    if (it == children.end()){
        throw std::runtime_error("Child with name '" + name + "' not found");
    }
    return it->second;
}

void Content::addChild(std::shared_ptr<Content> content){
    children[content->getName()] = content;
    content->setParent(this);
}

Window::Window(int width, int height, std::string name)
    : width(width),
    height(height),
    Content(name)
    {

}

void Window::content(){}


void Window::show(){
    if (active){
        startWindow();
        content();
        showChildren();
        endWindow();
    }

}

void Window::startWindow(){
    ImGui::Begin(name.c_str(), &active);
}

void Window::endWindow(){
    ImGui::End();
}





// CompositeWindow::CompositeWindow(GUIFrontend* frontend, int width, int height, std::string name)
//     : Window(frontend, width, height, name)  // CompositeWindow calls Window constructor
// {
//     // CompositeWindow-specific initialization
// }

// void CompositeWindow::startWindow(){
//     ImGui::Begin(name.c_str(), &active, ImGuiWindowFlags_AlwaysAutoResize);
//     content();
//     ImGui::End();
// }

// void CompositeWindow::show(){
//     startWindow();
//     showChildren();
// }

// void CompositeWindow::content(){
// }
