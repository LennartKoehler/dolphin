#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

class Image3D {
public:

    std::vector<cv::Mat> slices;
    float getPixel(int x, int y, int z);
    bool showSlice(int z);
    bool show();
};

