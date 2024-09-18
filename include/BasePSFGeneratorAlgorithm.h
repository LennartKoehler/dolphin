#pragma once

#include "PSF.h"

// Abstrakte Basisklasse für verschiedene Algorithmus-Typen
class BasePSFGeneratorAlgorithm {
public:
    virtual ~BasePSFGeneratorAlgorithm() = default;
    virtual PSF generatePSF() const = 0;
};


