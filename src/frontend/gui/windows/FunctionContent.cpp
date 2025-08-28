#include "frontend/gui/windows/FunctionContent.h"
#include "imgui.h"


FunctionContent::FunctionContent(std::string name, function func)
    :callback(func),
    Content(name){}

void FunctionContent::content(){
    callback();
}

void FunctionContent::setCallback(function func){
    this->callback = func;
}


ButtonContent::ButtonContent(std::string name, function func)
    : FunctionContent(name, func){}

void ButtonContent::content(){
        if (ImGui::Button(name.c_str())){
            callback();
        }
}


