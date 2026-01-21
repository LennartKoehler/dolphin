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


#include "psf/PSF.h"
#include "BasePSFGenerator.h"


// Konkrete Klasse SimpleGaussianPSFGeneratorAlgorithm
class BornWolfModel : public BasePSFGenerator {
public:
    BornWolfModel(double sigmaX, double sigmaY, double sigmaZ, int sizeX, int sizeY, int sizeZ)
            : sigmaX(sigmaX), sigmaY(sigmaY), sigmaZ(sigmaZ), sizeX(sizeX), sizeY(sizeY), sizeZ(sizeZ) {}

    PSF generatePSF() const override;

    // Setter-Funktion für die PSF-Parameter
    // void setParameters(double newSigmaX, double newSigmaY, double newSigmaZ, int newSizeX, int newSizeY, int newSizeZ) override{
    //     this->sigmaX = newSigmaX;
    //     this->sigmaY = newSigmaY;
    //     this->sigmaZ = newSigmaZ;
    //     this->sizeX = newSizeX;
    //     this->sizeY = newSizeY;
    //     this->sizeZ = newSizeZ;
    // }

private:
    double sigmaX;
    double sigmaY;
    double sigmaZ;
    int sizeX;
    int sizeY;
    int sizeZ;
};
