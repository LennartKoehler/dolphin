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
    
    // For complex functions: f(x) -> {real, imag}
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



