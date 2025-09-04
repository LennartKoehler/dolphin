
#include "SimpsonIntegrator.h"
#include <stdexcept>


SimpsonIntegrator::SimpsonIntegrator(int maxIterations)
    : maxIterations(maxIterations) {
}

int SimpsonIntegrator::getK(int accuracy){
    int K;
    if (accuracy == 0) K = 4;
    else if (accuracy == 1) K = 5;
    else if (accuracy == 2) K = 6;
    else K = 3;
    return K;
}



double SimpsonIntegrator::integrateComplex(std::function<std::array<double, 2>(double)> func,
    double a, double b,
    double tolerance, int accuracy) {
        return adaptiveSimpson(func, a, b, tolerance, accuracy);
}

double SimpsonIntegrator::integrate(std::function<double(double)> func, 
    double a, double b, 
    double tolerance, int accuracy){
        throw std::runtime_error("SimpsonIntegrator::integrate() not implemented - use integrateComplex() instead");

}

double SimpsonIntegrator::adaptiveSimpson(std::function<std::array<double, 2>(double)> func,
    double a, double b,
    double tolerance, int accuracy){
  
        int K = getK(accuracy);
        int N = 2;
        int k = 0;
        int iteration = 1;
        double del = (b - a) / 2.0;
        double curDifference = tolerance;

        std::array<double, 2> sumOddIndex = {0.0, 0.0};
        std::array<double, 2> sumEvenIndex = {0.0, 0.0};
        std::array<double, 2> valueX0 = {0.0, 0.0};
        std::array<double, 2> valueXn = {0.0, 0.0};
        std::array<double, 2> value = {0.0, 0.0};

        double rho = (b - a) / 2.0;
        sumOddIndex = func(rho);
        valueX0 = func(a);
        valueXn = func(b);

        double realSum = valueX0[0] + 2.0 * sumEvenIndex[0] + 4.0 * sumOddIndex[0] + valueXn[0];
        double imagSum = valueX0[1] + 2.0 * sumEvenIndex[1] + 4.0 * sumOddIndex[1] + valueXn[1];

        double curI = (realSum * realSum + imagSum * imagSum) * del * del;
        double prevI = curI;

        // LK compute more precise integrals by increasing the number of interpolation points
        while (k < K && iteration < maxIterations) {
            iteration++;
            N *= 2;
            del /= 2;
            sumEvenIndex[0] += sumOddIndex[0];
            sumEvenIndex[1] += sumOddIndex[1];
            sumOddIndex[0] = sumOddIndex[1] = 0.0;

            for (int n = 1; n < N; n += 2) {
                rho = n * del;
                value = func(rho);
                sumOddIndex[0] += value[0];
                sumOddIndex[1] += value[1];
            }

            realSum = valueX0[0] + 2.0 * sumEvenIndex[0] + 4.0 * sumOddIndex[0] + valueXn[0];
            imagSum = valueX0[1] + 2.0 * sumEvenIndex[1] + 4.0 * sumOddIndex[1] + valueXn[1];
            curI = (realSum * realSum + imagSum * imagSum) * del * del;

            if (prevI == 0.0)
                curDifference = std::abs((prevI - curI) / 1e-5);
            else
                curDifference = std::abs((prevI - curI) / curI);

            if (curDifference <= tolerance) k++;
            else k = 0;

            prevI = curI;
        }

        return curI;
    }