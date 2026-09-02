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
#include "BackendConfigWindow.h"

class PSFMainWindow : public Window{
public:
    PSFMainWindow(GUIFrontend* frontend, int width, int height, std::string name);
    void content() override;

    std::string psfPath;

private:
    bool showPSFWindow = false;
    GUIFrontend* guiFrontend;
};

class DeconvolutionMainWindow : public Window{
public:
    DeconvolutionMainWindow(GUIFrontend* frontend, int width, int height, std::string name);
    void show() override;


private:
    GUIFrontend* guiFrontend;
};