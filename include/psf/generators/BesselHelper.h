#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

class BesselHelper {
public:
    inline static BesselHelper& instance() {
        static BesselHelper helper;  // default constructor
        return helper;
    }

    void init(int minVal, int maxVal, double dxVal) {
        min = minVal;
        max = maxVal;
        dx = dxVal;
        besselValues.resize(static_cast<int>((max - min) / dx) + 1);
        for (int i = 0; i < besselValues.size(); i++) {
            double x = min + i * dx;
            besselValues[i] = std::cyl_bessel_j(0, x);
        }
    }

    double get(double x) const {
        int index = static_cast<int>((x - min) / dx);
        index = std::clamp(index, 0, static_cast<int>(besselValues.size() - 1));
        return besselValues[index];
    }

    inline double calcBessel(const double& x) const{
        
        double BesselValue = std::cyl_bessel_j(0, x);
        return BesselValue;
    }
private:
    BesselHelper() = default;

    std::vector<double> besselValues;
    int max;
    int min;
    double dx;

};