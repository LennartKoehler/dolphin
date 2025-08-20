/**
 * PSFGenerator
 * 
 * Authors: Daniel Sage and Hagai Kirshner
 * Organization: Biomedical Imaging Group (BIG), Ecole Polytechnique Federale de Lausanne
 * Address: EPFL-STI-IMT-LIB, 1015 Lausanne, Switzerland
 * Information: http://bigwww.epfl.ch/algorithms/psfgenerator/
 *
 * References:
 * [1] H. Kirshner, F. Aguet, D. Sage, M. Unser
 * 3-D PSF Fitting for Fluorescence Microscopy: Implementation and Localization Application 
 * Journal of Microscopy, vol. 249, no. 1, pp. 13-25, January 2013.
 * Available at: http://bigwww.epfl.ch/publications/kirshner1301.html
 * 
 * [2] A. Griffa, N. Garin, D. Sage
 * Comparison of Deconvolution Software in 3D Microscopy: A User Point of View
 * G.I.T. Imaging & Microscopy, vol. 12, no. 1, pp. 43-45, March 2010.
 * Available at: http://bigwww.epfl.ch/publications/griffa1001.html
 *
 * Conditions of use:
 * Conditions of use: You are free to use this software for research or
 * educational purposes. In addition, we expect you to include adequate
 * citations and acknowledgments whenever you present or publish results that
 * are based on it.
 */

/**
 * Copyright 2010-2017 Biomedical Imaging Group at the EPFL.
 * 
 * This file is part of PSFGenerator.
 * 
 * PSFGenerator is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * PSFGenerator is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * PSFGenerator. If not, see <http://www.gnu.org/licenses/>.
 */

#include "psf/KirchoffDiffractionSimpson.h"


KirchhoffDiffractionSimpson::KirchhoffDiffractionSimpson(const GibsonLanniPSFConfig& params, int accuracy, double NA_, double lambda_)
    : p(params), NA(NA_), lambda(lambda_)
{
    if (accuracy == 0) K = 4;
    else if (accuracy == 1) K = 5;
    else if (accuracy == 2) K = 6;
    else K = 3;
}

// see equations (1) - (4) of 3-D PSF fitting for fluorescence microscopy: implementation and localization application
std::array<double, 2> KirchhoffDiffractionSimpson::integrand(double rho, double r) {
    std::array<double, 2> I = {0.0, 0.0};
    double k0 = 2.0 * M_PI / lambda;
    double BesselValue = std::cyl_bessel_j(0, k0 * NA * r * rho);

    if ((NA * rho / p.ns) > 1.0)
        std::cout << "Warning: NA*rho/ns > 1, (ns,NA,rho)=(" 
                  << p.ns << ", " << NA << ", " << rho << ")\n";

    double OPD1 = p.ns * p.particleAxialPosition * std::sqrt(1 - std::pow(NA * rho / p.ns, 2));
    double OPD3 = p.ni * (p.ti - p.ti0) * std::sqrt(1 - std::pow(NA * rho / p.ni, 2));
    double OPD = OPD1 + OPD3;

    double W = k0 * OPD;

    I[0] = BesselValue * std::cos(W) * rho;
    I[1] = BesselValue * std::sin(W) * rho;

    return I;
}

double KirchhoffDiffractionSimpson::calculate(double r) {
    double a = 0.0;
    double b = std::min(1.0, p.ns / NA);
    int N = 2;
    int k = 0;
    int iteration = 1;
    double del = (b - a) / 2.0;
    double curDifference = TOL;

    std::array<double, 2> sumOddIndex = {0.0, 0.0};
    std::array<double, 2> sumEvenIndex = {0.0, 0.0};
    std::array<double, 2> valueX0 = {0.0, 0.0};
    std::array<double, 2> valueXn = {0.0, 0.0};
    std::array<double, 2> value = {0.0, 0.0};

    double rho = (b - a) / 2.0;
    sumOddIndex = integrand(rho, r);
    valueX0 = integrand(a, r);
    valueXn = integrand(b, r);

    double realSum = valueX0[0] + 2.0 * sumEvenIndex[0] + 4.0 * sumOddIndex[0] + valueXn[0];
    double imagSum = valueX0[1] + 2.0 * sumEvenIndex[1] + 4.0 * sumOddIndex[1] + valueXn[1];

    double curI = (realSum * realSum + imagSum * imagSum) * del * del;
    double prevI = curI;

    while (k < K && iteration < 10000) {
        iteration++;
        N *= 2;
        del /= 2;
        sumEvenIndex[0] += sumOddIndex[0];
        sumEvenIndex[1] += sumOddIndex[1];
        sumOddIndex[0] = sumOddIndex[1] = 0.0;

        for (int n = 1; n < N; n += 2) {
            rho = n * del;
            value = integrand(rho, r);
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

        if (curDifference <= TOL) k++;
        else k = 0;

        prevI = curI;
    }

    return curI;
}
