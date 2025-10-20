#include "frontend/gui/imguiWidget.h"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <algorithm>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers


void imguiWidget::operator() (const ConfigParameter& param){
    display(param);
}


void imguiSliderDouble::display(const ConfigParameter& p){
    ImGui::SliderScalar(
        p.name,
        ImGuiDataType_Double,
        static_cast<double*>(p.value),
        &p.minVal,
        &p.maxVal
    );
}

void imguiSliderInt::display(const ConfigParameter& p) {
    ImGui::SliderInt(
        p.name,
        static_cast<int*>(p.value),
        static_cast<int>(p.minVal),
        static_cast<int>(p.maxVal)
    );
}

void imguiInputInt::display(const ConfigParameter& p){
    ImGui::InputInt(p.name, static_cast<int*>(p.value));
}

void imguiInputFloat::display(const ConfigParameter& p){
    ImGui::InputFloat(p.name, static_cast<float*>(p.value));
}



void imguiInputString::display(const ConfigParameter& p) {
    auto strPtr = static_cast<std::string*>(p.value);
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%s", strPtr->c_str());

    if (ImGui::InputText(p.name, buffer, sizeof(buffer))) {
        *strPtr = buffer; // copy back to std::string if edited
    }
}

void imguiCheckbox::display(const ConfigParameter& p) {
    ImGui::Checkbox(
        p.name,
        static_cast<bool*>(p.value)
    );
}


void imguiVectorInt::display(const ConfigParameter& p){
    values = static_cast<std::vector<int>*>(p.value);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    ImGui::Text("%s:", p.name);
    // ImGui::Separator();
    
    ImGui::PushID(p.name);
    // Display existing elements
    for (size_t i = 0; i < values->size(); ++i) {
        int id = static_cast<int>(i);
        ImGui::PushID(id);  // Unique ID for each element
        
        displayElement(&(*values)[i], id);
        
        ImGui::SameLine();
        
        // Remove button for each element
        if (ImGui::Button("X")) {
            removeElement(id);
        }
        
        ImGui::PopID();
    }
    
    // Add new element widget
    addElementWidget();
    ImGui::PopID();
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

}

void imguiVectorInt::displayElement(int* val, int index) {
    char label[32];
    std::snprintf(label, sizeof(label), "##element_%d", index);
    ImGui::InputInt(label, val);
}

void imguiVectorInt::addElementWidget() {
    ImGui::Separator();
    
    // Input for new value
    ImGui::InputInt("New Value", &newValue);
    
    ImGui::SameLine();
    
    // Add button
    if (ImGui::Button("Add")) {
        if (values) {
            values->push_back(newValue);
            newValue = 0;  // Reset input
        }
    }
    
    ImGui::SameLine();
    
    // Clear all button
    if (ImGui::Button("Clear All")) {
        if (values) {
            values->clear();
        }
    }
}

void imguiVectorInt::removeElement(int index) {
    if (values && index >= 0 && index < static_cast<int>(values->size())) {
        values->erase(values->begin() + index);
    }
}

void imguiStringSelection::display(const ConfigParameter& p) {

    std::string* field = static_cast<std::string*>(p.value);
    options = *static_cast<std::vector<std::string>*>(p.selection);

    // Find current selection index
    auto it = std::find(options.begin(), options.end(), *field);
    if (it != options.end()) {
        currentSelection = static_cast<int>(std::distance(options.begin(), it));
    } else {
        currentSelection = 0; // Default to first option if not found
    }
    
    // Create array of const char* for ImGui::Combo
    std::vector<const char*> items;
    for (const auto& option : options) {
        items.push_back(option.c_str());
    }
    
    // Display combo box
    if (ImGui::Combo(p.name, &currentSelection, items.data(), static_cast<int>(items.size()))) {
        // Update the string when selection changes
        if (currentSelection >= 0 && currentSelection < static_cast<int>(options.size())) {
            *field = (options)[currentSelection];
        }
    }
}


void imguiFileExplorer::display(const ConfigParameter& p){
    fileDialog.SetTitle(p.name);
    ImGui::PushID(p.name);
    
    if (selected.empty()){
        buttonName = "Open...";
    }
    else{
        buttonName = selected;
    }
    // Get the current item width and use it for button
    float currentItemWidth = ImGui::CalcItemWidth();
    ImVec2 buttonSize(currentItemWidth, 0.0f);
    if(ImGui::Button(buttonName.c_str(), buttonSize))
        fileDialog.Open();
    fileDialog.Display();
    ImGui::SameLine();
    ImGui::Text("%s", p.name);
    
    if(fileDialog.HasSelected())
    {
        selected = fileDialog.GetSelected().string();
        *static_cast<std::string*>(p.value) = selected;
        // std::cout << "Selected filename" <<  << std::endl;
        fileDialog.ClearSelected();
    }
    ImGui::PopID();
}