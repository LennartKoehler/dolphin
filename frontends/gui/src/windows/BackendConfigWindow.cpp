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

#include "windows/FunctionContent.h"
#include "GUIFrontend.h"

#include "windows/BackendConfigWindow.h"


#include "imgui.h"


BackendConfigWindow::BackendConfigWindow(GUIFrontend* guiFrontend, int width, int height, std::string name,  std::shared_ptr<Config> config)
    : ConfigWindow(width, height, name, config),
    guiFrontend(guiFrontend){
    }

void BackendConfigWindow::content(){
    showConfigParameters(*config, style);

}