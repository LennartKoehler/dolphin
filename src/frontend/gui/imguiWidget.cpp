#include "frontend/gui/imguiWidget.h"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

void imguiWidget::operator() (const ParameterDescription& param){
    display(param);
}


void imguiSliderDouble::display(const ParameterDescription& p){
    ImGui::InputDouble(
        p.name.c_str(),
        static_cast<double*>(p.ptr)
        // &p.minVal,
        // &p.maxVal
    );
}

void imguiSliderInt::display(const ParameterDescription& p) {
    ImGui::InputInt(
        p.name.c_str(),
        static_cast<int*>(p.ptr)
        // static_cast<int>(p.minVal),
        // static_cast<int>(p.maxVal)
    );
}

void imguiInputString::display(const ParameterDescription& p) {
    auto strPtr = static_cast<std::string*>(p.ptr);
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%s", strPtr->c_str());

    if (ImGui::InputText(p.name.c_str(), buffer, sizeof(buffer))) {
        *strPtr = buffer; // copy back to std::string if edited
    }
}

void imguiCheckbox::display(const ParameterDescription& p) {
    ImGui::Checkbox(
        p.name.c_str(),
        static_cast<bool*>(p.ptr)
    );
}


void imguiVectorInt::display(const ParameterDescription& p){
    values = static_cast<std::vector<int>*>(p.ptr);
    
    ImGui::Text("%s:", p.name.c_str());
    ImGui::Separator();
    
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