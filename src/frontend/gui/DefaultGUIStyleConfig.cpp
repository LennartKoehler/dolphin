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

void DefaultGUIStyleConfig::drawParameter(const ConfigParameter& param){
    // Create a unique key for this parameter
    int cacheKey = param.ID;
    
    // Check if we already have a widget for this parameter
    if (widgetCache.find(cacheKey) == widgetCache.end()) {
        // Create new widget instance for this parameter
        if (auto it = widgetFactory.find(param.type); it != widgetFactory.end()) {
            widgetCache[cacheKey] = it->second();
        }
    }
    
    // Use the cached widget
    if (auto& widget = widgetCache[cacheKey]) {
        widget->display(param);
    }
}

void DefaultGUIStyleConfig::registerDisplays(){
    widgetFactory[ParameterType::Int] = []() { return std::make_unique<imguiInputInt>(); };
    widgetFactory[ParameterType::String] = []() { return std::make_unique<imguiInputString>(); };
    widgetFactory[ParameterType::Float] = []() { return std::make_unique<imguiInputFloat>(); };
    widgetFactory[ParameterType::Bool] = []() { return std::make_unique<imguiCheckbox>(); };
    widgetFactory[ParameterType::VectorInt] = []() { return std::make_unique<imguiVectorInt>(); };
    widgetFactory[ParameterType::VectorString] = []() { return std::make_unique<imguiStringSelection>(); };
    widgetFactory[ParameterType::FilePath] = []() { return std::make_unique<imguiFileExplorer>(); };
}