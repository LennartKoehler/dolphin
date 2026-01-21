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

class PSFConfig;
class ThreadPool;
// BasePSFGenerators are classes which use their respective config to set up and then implement a specific algorithm in generatePSF
class BasePSFGenerator {
public:
    BasePSFGenerator() = default;
    virtual ~BasePSFGenerator() = default;
    virtual PSF generatePSF() const = 0;
    virtual void setConfig(const std::shared_ptr<const PSFConfig> config) = 0; // LK i tihnk this can maybe be defined in the base class
	virtual bool hasConfig() = 0; // TODO this could probably implemented in the base class when i also set a abstract shared_ptr<PSFConfig> in the base class (here);
    inline void setThreadPool(ThreadPool* threadPool){ this->threadPool = threadPool;};
protected:
    ThreadPool* threadPool;
};


