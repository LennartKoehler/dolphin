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

#pragma once

#include "Window.h"


class MainWindow : public Window{
public:
    MainWindow(GUIFrontend* guiFrontend, int width, int height, std::string name);

private:
    void content() override;
    void startWindow() override;
    void endWindow() override;
    GUIFrontend* guiFrontend;
};