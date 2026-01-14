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
#include "NumericalIntegrator.h"

class SimpsonIntegrator : public NumericalIntegrator {
public:
    SimpsonIntegrator(int maxIterations = 10000);
    
    double integrate(std::function<double(double)> func, 
                    double a, double b, 
                    double tolerance = defaultTolerance, int accuracy = defaultAccuracy) override;
    
    double integrateComplex(std::function<std::array<double, 2>(double)> func,
                          double a, double b,
                          double tolerance = defaultTolerance, int accuracy = defaultAccuracy) override;

    int getK(int accuracy);
private:
    int maxIterations;
    static constexpr double defaultTolerance = 1e-8;
    static constexpr int defaultAccuracy = 3;
    
    // for imaginary, one could implement with template for more generic
    //should K or accuracy be passed here?
    double adaptiveSimpson(std::function<std::array<double, 2>(double)> func, 
                     double a, double b, 
                     double tolerance = defaultTolerance, int accuracy = defaultAccuracy);

};