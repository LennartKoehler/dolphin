#include "psf/KirchoffDiffractionSimpson.h"



KirchhoffDiffractionSimpson::KirchhoffDiffractionSimpson(const GibsonLanniPSFConfig& params, int accuracy, double NA_, double lambda_)
    : p(params), NA(NA_), lambda(lambda_)
{
    if (accuracy == 0) K = 4;
    else if (accuracy == 1) K = 5;
    else if (accuracy == 2) K = 6;
    else K = 3;
}

std::vector<double> KirchhoffDiffractionSimpson::integrand(double rho, double r) {
    std::vector<double> I(2, 0.0);
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

    std::vector<double> sumOddIndex(2, 0.0);
    std::vector<double> sumEvenIndex(2, 0.0);
    std::vector<double> valueX0(2, 0.0);
    std::vector<double> valueXn(2, 0.0);
    std::vector<double> value(2, 0.0);

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
