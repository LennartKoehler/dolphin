#pragma once

#include <vector>
#include "Image3D.h"
#include "ImageMetaData.h"

class Channel {
public:
    int id{};
    Image3D image;
};