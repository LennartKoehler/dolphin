#pragma once

#include "PSF.h"

class PSFConfig;

// BasePSFGenerators are classes which use their respective config to set up and then implement a specific algorithm in generatePSF
class BasePSFGenerator {
public:
    BasePSFGenerator() = default;
    virtual ~BasePSFGenerator() = default;
    virtual PSF generatePSF() const = 0;
    virtual void setConfig(std::unique_ptr<PSFConfig> config) = 0; // LK i tihnk this can maybe be defined in the base class
	virtual bool hasConfig() = 0; // TODO this could probably implemented in the base class when i also set a abstract unique_ptr<PSFConfig> in the base class (here);

    // virtual void setParameters(double d, double d1, double d2, int i, int i1, int i2) = 0; // LKlegaccy

// LK is this the same for all psfconfigs? got it from BIG and how their abstract psfconfig looks
//     void setParameters(int sizeX, int sizeY, int sizeZ, double resLat, double resAx, 
//                   double numericalAperture, double wavelength, 
//                   const std::string& fullName = "Untitled", 
//                   const std::string& shortName = "...", 
//                   int psfType = 0, int psfScale = 1);

// protected:

// 	std::string fullname = "Untitled";
// 	std::string shortname = "...";
// 	int sizeXPixels;
// 	int sizeYPixels;
// 	int sizeZPixels;
// 	int type;
// 	int scale;

// 	double resLateral; 	// in nm
// 	double resAxial; 	// in nm
// 	double NA;
// 	double lambda; 		// in nm
};


