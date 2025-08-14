#pragma once


#include <iostream>
#include <memory>
#include "BasePSFGenerator.h"

// Template-Klasse PSFGenerator, die den Algorithmus akzeptiert
// LK TODO make this into simple factory pattern, its basically the same as deconvolution factory, maybe make both into one -> create algorithm, give config
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
    std::unique_ptr<BasePSFGenerator> algo;
};

