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

#include "GUIStyleConfig.h"
#include <vector>
#include "imgui.h"
#include <imfilebrowser.h>

//basically my own api for common imgui calls
class imguiWidget{
public:
    imguiWidget() = default;
    virtual ~imguiWidget() = default;
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