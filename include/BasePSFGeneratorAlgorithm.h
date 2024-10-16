#pragma once

#include "PSF.h"

// Abstrakte Basisklasse für verschiedene Algorithmus-Typen
class BasePSFGeneratorAlgorithm {
public:
    virtual ~BasePSFGeneratorAlgorithm() = default;
    virtual PSF generatePSF() const = 0;
    virtual void setParameters(double d, double d1, double d2, int i, int i1, int i2) = 0;
};


