#pragma once

#include "GUIStyleConfig.h"
#include <vector>
#include "imgui.h"
#include <imfilebrowser.h>

//basically my own api for common imgui calls
class imguiWidget{
public:
    imguiWidget() = default;
    virtual ~imguiWidget(){}
    virtual void display(const ConfigParameter& p) = 0;
    void operator() (const ConfigParameter& p);

};


class imguiSliderDouble : public imguiWidget{
public:
    void display(const ConfigParameter& p) override;
};

class imguiSliderInt : public imguiWidget {
public:
    void display(const ConfigParameter& p) override;
};

class imguiInputInt: public imguiWidget{
public:
    void display(const ConfigParameter& p) override;
};

class imguiInputFloat: public imguiWidget{
public:
    void display(const ConfigParameter& p) override;
};

class imguiInputString : public imguiWidget {
public:
    void display(const ConfigParameter& p) override;
};

class imguiCheckbox : public imguiWidget {
public:
    void display(const ConfigParameter& p) override;
};

class imguiVectorInt : public imguiWidget {
public:
    void display(const ConfigParameter& p) override;

private:
    std::vector<int>* values = nullptr;
    int newValue = 0;  // Value to be added
    
    void displayElement(int* val, int index);
    void addElementWidget();
    void removeElement(int index);
};

class imguiStringSelection : public imguiWidget {
public:    
    void display(const ConfigParameter& p) override;

private:
    std::vector<std::string> options;
    int currentSelection = 0;
};


class imguiFileExplorer : public imguiWidget{
public:
    // imguiFileExplorer() : fileDialog(ImGuiFileBrowserFlags_OpenFile) {}
    void display(const ConfigParameter& p) override;

private:
    ImGui::FileBrowser fileDialog;
    std::string buttonName;
    std::string selected;
};