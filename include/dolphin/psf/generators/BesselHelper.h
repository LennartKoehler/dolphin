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
#include <vector>
#include <cmath>

#if defined(_MSC_VER)
inline double dolphin_cyl_bessel_j0(double x) { return _j0(x); }
inline double dolphin_cyl_bessel_j1(double x) { return _j1(x); }
#elif defined(__APPLE__)
#include <math.h>
inline double dolphin_cyl_bessel_j0(double x) { return ::j0(x); }
inline double dolphin_cyl_bessel_j1(double x) { return ::j1(x); }
#else
inline double dolphin_cyl_bessel_j0(double x) { return std::cyl_bessel_j(0, x); }
inline double dolphin_cyl_bessel_j1(double x) { return std::cyl_bessel_j(1, x); }
#endif

// Tabulated J0 with cubic Hermite interpolation. The J1 table provides the
// exact slopes (J0'(x) = -J1(x)), giving O(dx^4) accuracy. One instance per
// PSF generation - not shared between threads.
class BesselHelper {
public:
    BesselHelper() = default;

    void init(double maxArg, double dxVal) {
        dx = dxVal > 0.0 ? dxVal : 1.0;
        invDx = 1.0 / dx;
        n = static_cast<size_t>(maxArg * invDx) + 2;  // last node >= maxArg
        j0Vals.resize(n);
        j1Vals.resize(n);
        for (size_t i = 0; i < n; i++) {
            const double x = static_cast<double>(i) * dx;
            j0Vals[i] = dolphin_cyl_bessel_j0(x);
            j1Vals[i] = dolphin_cyl_bessel_j1(x);
        }
    }

    // Cubic Hermite interpolation of J0(x), error O(dx^4 * |J0''''|)
    double get(double x) const {
        if (n == 0) return dolphin_cyl_bessel_j0(x);
        double t = x * invDx;
        if (t < 0.0) t = 0.0;
        size_t i = static_cast<size_t>(t);
        i = std::min(i, n - 2);  // clamp instead of assert: works in release builds
        const double f = t - static_cast<double>(i);
        if (f == 0.0) return j0Vals[i];

        const double f2 = f * f;
        const double f3 = f2 * f;
        const double h00 = 2.0 * f3 - 3.0 * f2 + 1.0;
        const double h10 = f3 - 2.0 * f2 + f;
        const double h01 = -2.0 * f3 + 3.0 * f2;
        const double h11 = f3 - f2;
        return h00 * j0Vals[i]
             - dx * h10 * j1Vals[i]
             + h01 * j0Vals[i + 1]
             - dx * h11 * j1Vals[i + 1];
    }

    inline double calcBessel(const double& x) const{
        return dolphin_cyl_bessel_j0(x);
    }

    size_t size() const { return n; }

private:
    std::vector<double> j0Vals;
    std::vector<double> j1Vals;
    double dx = 1.0;
    double invDx = 1.0;
    size_t n = 0;
};
