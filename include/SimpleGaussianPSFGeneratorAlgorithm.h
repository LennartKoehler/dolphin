#pragma once


#include "PSF.h"
#include "BasePSFGeneratorAlgorithm.h"

// Konkrete Klasse SimpleGaussianPSFGeneratorAlgorithm
class SimpleGaussianPSFGeneratorAlgorithm : public BasePSFGeneratorAlgorithm {
public:
    SimpleGaussianPSFGeneratorAlgorithm(double sigmaX, double sigmaY, double sigmaZ, int sizeX, int sizeY, int sizeZ)
            : sigmaX(sigmaX), sigmaY(sigmaY), sigmaZ(sigmaZ), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {}

    PSF generatePSF() const override;

private:
    double sigmaX;
    double sigmaY;
    double sigmaZ;
    int sizeX;
    int sizeY;
    int sizeZ;
};


