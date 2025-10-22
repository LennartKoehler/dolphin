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


