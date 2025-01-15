#pragma once

#include <string>
#include <opencv2/core/base.hpp>
#include <vector>

class DeconvolutionConfig {
public:
    int iterations = 10;
    double epsilon = 1e-6;
    bool grid = true;
    double lambda = 0.001;
    int borderType = cv::BORDER_REFLECT;
    int psfSafetyBorder = 10;
    int cubeSize = 0;
    std::vector<int> secondpsflayers = {};
    std::vector<int> secondpsfcubes = {};
    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;

    std::string gpu = "";

    void loadFromJSON(const std::string &directoryPath);
};



