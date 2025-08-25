#pragma once

class Dolphin;
class Window;
class GUI{
public:
    GUI();
    int init();
    void update();
    void cleanup();
    void render();
    void showActiveWindow();

private:
    Window* activeWindow;
    Dolphin* dolphin;
};