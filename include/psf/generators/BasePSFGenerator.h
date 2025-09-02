#pragma once

#include "psf/PSF.h"

class PSFConfig;

// BasePSFGenerators are classes which use their respective config to set up and then implement a specific algorithm in generatePSF
class BasePSFGenerator {
public:
    BasePSFGenerator() = default;
    virtual ~BasePSFGenerator() = default;
    virtual PSF generatePSF() const = 0;
    virtual void setConfig(const std::shared_ptr<const PSFConfig> config) = 0; // LK i tihnk this can maybe be defined in the base class
	virtual bool hasConfig() = 0; // TODO this could probably implemented in the base class when i also set a abstract shared_ptr<PSFConfig> in the base class (here);

};


