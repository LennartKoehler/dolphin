#pragma once


#include "Image3D.h"
typedef std::string PSFID;
class PSF {
public:
    Image3D image;
    std::string ID;

    bool readFromTifFile(const char *filename);
    bool readFromTifDir(const std::string& directoryPath);

    bool saveAsTifFile(const std::string &directoryPath);
    bool saveAsTifDir(const std::string& directoryPath);

    PSF flip3DPSF();
    void scalePSF(int new_size_x, int new_size_y, int new_size_z);

};


