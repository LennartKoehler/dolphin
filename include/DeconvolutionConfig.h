#pragma once

#include <string>
#include <opencv2/core/base.hpp>
#include <vector>

class DeconvolutionConfig {
public:
    int iterations = 1;
    double epsilon = 1e-8;
    bool grid = true;
    double lambda = 0.001;
    int borderType = cv::BORDER_REFLECT;
    int psfSafetyBorder = 20;
    int cubeSize = 0;
    std::vector<int> secondpsflayers = {};
    std::vector<int> secondpsfcubes = {};
    std::vector<std::vector<int>> psfCubeVec, psfLayerVec;
    bool secondPSF = false;

    std::string gpu = "";

    void loadFromJSON(const std::string &directoryPath);
};



