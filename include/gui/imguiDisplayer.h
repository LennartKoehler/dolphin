#pragma once

#include "GUIStyleConfig.h"
#include "gui/GUI.h"


#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers


//basically my own api for common imgui calls
class imguiDisplayer{
public:
    imguiDisplayer() = default;
    virtual ~imguiDisplayer();
    virtual void display(const ParameterDescription& p);
    void operator() (const ParameterDescription& p);

};


class imguiSliderDouble : public imguiDisplayer{
public:
    void display(const ParameterDescription& p) override;
};

class imguiSliderInt : public imguiDisplayer {
public:
    void display(const ParameterDescription& p) override;
};

class imguiInputString : public imguiDisplayer {
public:
    void display(const ParameterDescription& p) override;
};