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