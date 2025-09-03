#include "frontend/gui/DefaultGUIStyleConfig.h"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers


DefaultGUIStyleConfig::DefaultGUIStyleConfig(){
    registerDisplays();
}

void DefaultGUIStyleConfig::drawParameter(const ParameterDescription& param){

    if (auto it = styleGuide.find(param.type); it != styleGuide.end()) {
        it->second->display(param);
    } else {
        ImGui::Text("No widget for type!");
    }
}


void DefaultGUIStyleConfig::registerDisplays(){
    styleGuide[ParameterType::Int] = std::make_unique<imguiSliderInt>();
    styleGuide[ParameterType::String] = std::make_unique<imguiInputString>();
    styleGuide[ParameterType::Double] = std::make_unique<imguiSliderDouble>();
    styleGuide[ParameterType::Bool] = std::make_unique<imguiCheckbox>();
    styleGuide[ParameterType::VectorInt] = std::make_unique<imguiVectorInt>();
    styleGuide[ParameterType::VectorString] = std::make_unique<imguiStringSelection>();


}