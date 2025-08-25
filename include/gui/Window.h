#pragma once
class GLFWwindow;



class Window{
public:
    Window(GLFWwindow* window);
    void show();
    GLFWwindow* getGLFW();
private:
    GLFWwindow* window;
    bool active = false;
};