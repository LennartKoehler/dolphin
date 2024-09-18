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

private:
    std::unique_ptr<BasePSFGeneratorAlgorithm> algo;
};

