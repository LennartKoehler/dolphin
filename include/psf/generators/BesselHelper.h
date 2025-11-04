#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// precomputed the bessel function to be reused later -> speedup, but more memory usage (~1MB)
class BesselHelper {
public:
    inline static BesselHelper& instance() {
        static BesselHelper helper;  // default constructor
        return helper;
    }

    void init(double minVal, double maxVal, double dxVal) {
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
        assert(besselValues.size() != 0 && "BesselHelper not initliazed");
        int index = static_cast<int>((x - min) / dx);
        assert(index < besselValues.size() -1 && index >= 0 && "Index to large or small");
        return besselValues[index];
    }

    inline double calcBessel(const double& x) const{
        
        double BesselValue = std::cyl_bessel_j(0, x);
        return BesselValue;
    }
private:
    BesselHelper() = default;

    std::vector<double> besselValues;
    double max;
    double min;
    double dx;

};