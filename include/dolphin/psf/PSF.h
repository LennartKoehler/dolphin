#pragma once


#include "Image3D.h"
typedef std::string PSFID;
class PSF {
public:
    Image3D image;
    std::string ID;

    void readFromTiffFile(const std::string& path);
    void writeToTiffFile(const std::string& path);
};


