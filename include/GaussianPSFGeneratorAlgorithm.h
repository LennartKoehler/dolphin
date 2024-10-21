#pragma once


#include "PSF.h"
#include "BasePSFGeneratorAlgorithm.h"

// Konkrete Klasse SimpleGaussianPSFGeneratorAlgorithm
class GaussianPSFGeneratorAlgorithm : public BasePSFGeneratorAlgorithm {
public:
    GaussianPSFGeneratorAlgorithm(double sigmaX, double sigmaY, double sigmaZ, int sizeX, int sizeY, int sizeZ)
            : sigmaX(sigmaX), sigmaY(sigmaY), sigmaZ(sigmaZ), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {}

    PSF generatePSF() const override;

    // Setter-Funktion für die PSF-Parameter
    void setParameters(double newSigmaX, double newSigmaY, double newSigmaZ, int newSizeX, int newSizeY, int newSizeZ) override{
        this->sigmaX = newSigmaX;
        this->sigmaY = newSigmaY;
        this->sigmaZ = newSigmaZ;
        this->sizeX = newSizeX;
        this->sizeY = newSizeY;
        this->sizeZ = newSizeZ;
    }

private:
    double sigmaX;
    double sigmaY;
    double sigmaZ;
    int sizeX;
    int sizeY;
    int sizeZ;
};
