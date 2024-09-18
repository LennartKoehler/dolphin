#pragma once


#include "Image3D.h"

class PSF {
public:
    Image3D image;

    bool readFromTifFile(const char *filename);
    bool readFromTifDir(const std::string& directoryPath);

    bool saveAsTifFile(const std::string &directoryPath);
    bool saveAsTifDir(const std::string& directoryPath);

    PSF flip3DPSF();
    void scalePSF(int new_size_x, int new_size_y, int new_size_z);

};


