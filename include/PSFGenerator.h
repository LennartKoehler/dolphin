#pragma once


#include <iostream>
#include <memory>
#include "BasePSFGeneratorAlgorithm.h"

// Template-Klasse PSFGenerator, die den Algorithmus akzeptiert
template<typename Algorithm, typename... Args>
class PSFGenerator {
public:
    PSFGenerator(Args&&... args)
            : algo(std::make_unique<Algorithm>(std::forward<Args>(args)...)) {}

    PSF generate() const {
        return algo->generatePSF();
    }

    void setParameters(double d, double d1, double d2, int i, int i1, int i2) const {
        algo->setParameters(d, d1, d2, i, i1, i2);
    }

private:
    std::unique_ptr<BasePSFGeneratorAlgorithm> algo;
};

