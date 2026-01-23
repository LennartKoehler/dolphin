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
#include <functional>
// #include <array>

class NumericalIntegrator {
public:
    virtual ~NumericalIntegrator() = default;
    
    // For scalar functions: f(x) -> double
    virtual double integrate(std::function<double(double)> func, 
                           double a, double b, 
                           double tolerance = 1e-6, int accuracy = 1) = 0;
    
    // For complex_t functions: f(x) -> {real, imag}
    virtual double integrateComplex(std::function<std::array<double, 2>(double)> func,
                                  double a, double b,
                                  double tolerance = 1e-6, int accuracy = 1) = 0;
    
    // Get integration statistics
    virtual int getEvaluationCount() const { return evaluationCount; }
    virtual int getIterationCount() const { return iterationCount; }

protected:
    int evaluationCount = 0;
    int iterationCount = 0;
};



