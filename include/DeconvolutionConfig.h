#pragma once

#include <string>
#include <opencv2/core/base.hpp>

class DeconvolutionConfig {
public:
    int iterations = 1;
    double epsilon = 1e-8;
    bool grid = true;
    double lambda = 0.001;
    int borderType = cv::BORDER_REFLECT;
    int psfSafetyBorder = 20;
    int cubeSize = 50;
    std::vector<int> secondpsflayers = {};

    void loadFromJSON(const std::string &directoryPath);
};



