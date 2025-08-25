#include "gui/imguiDisplayer.h"


void imguiDisplayer::operator() (const ParameterDescription& param){
    display(param);
}


void imguiSliderDouble::display(const ParameterDescription& p){
    ImGui::SliderScalar(
        p.name.c_str(),
        ImGuiDataType_Double,
        p.ptr,
        &p.minVal,
        &p.maxVal
    );
}

void imguiSliderInt::display(const ParameterDescription& p) {
    ImGui::SliderInt(
        p.name.c_str(),
        static_cast<int*>(p.ptr),
        static_cast<int>(p.minVal),
        static_cast<int>(p.maxVal)
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