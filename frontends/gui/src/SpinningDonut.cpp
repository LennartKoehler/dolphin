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

#include "implot3d.h"
#include "implot3d_internal.h"
#include <cmath>
#include <array>
#include <numbers>


void SpinningDonut(){
    ImPlot3D::CreateContext();
    float thickness = 4;
    float radius = 10;
    int resolution = 30;
    int slowdown = 20;
    const static int nPoints = 10000;
    static float xs[nPoints], ys[nPoints], zs[nPoints];
    for (int i = 0; i < nPoints; i++) {
        double angle = 2*M_PI * i/nPoints;
        
        // Generate basic donut shape
        double x = std::cos(i / resolution) * thickness * std::cos(angle) + (radius * cos(angle));
        double y = std::sin(i / resolution) * thickness;
        double z = ((radius * sin(angle)) + std::cos(i / resolution) * thickness * std::sin(angle));
        
        // Time-based rotation angles (different speeds for each axis)
        double time = (float)ImGui::GetTime() / slowdown;
        double angleX = 2*M_PI * time * 0.7;        // Rotate around X-axis
        double angleY = 2*M_PI * time * 1.0;        // Rotate around Y-axis  
        double angleZ = 2*M_PI * time * 0.5;        // Rotate around Z-axis
        
        // Rotation around X-axis
        double cosX = std::cos(angleX), sinX = std::sin(angleX);
        double y1 = y * cosX - z * sinX;
        double z1 = y * sinX + z * cosX;
        
        // Rotation around Y-axis
        double cosY = std::cos(angleY), sinY = std::sin(angleY);
        double x2 = x * cosY + z1 * sinY;
        double z2 = -x * sinY + z1 * cosY;
        
        // Rotation around Z-axis
        double cosZ = std::cos(angleZ), sinZ = std::sin(angleZ);
        double x3 = x2 * cosZ - y1 * sinZ;
        double y3 = x2 * sinZ + y1 * cosZ;
        
        xs[i] = x3;
        ys[i] = y3;
        zs[i] = z2;
    }


    if (ImPlot3D::BeginPlot("Line Plots")) {
        ImPlot3D::SetupAxes("x", "y", "z");
        ImPlot3D::SetupAxisLimits(ImAxis3D_X, -13, 13);  // X axis: -10 to 10
        ImPlot3D::SetupAxisLimits(ImAxis3D_Y, -13, 13);    // Y axis: -5 to 5
        ImPlot3D::SetupAxisLimits(ImAxis3D_Z, -13, 13);   // Z axis: 0 to 100
    
        ImPlot3D::PlotLine("f(x)", xs, ys, zs, nPoints);
        ImPlot3D::EndPlot();
    }
}


