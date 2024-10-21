#pragma once

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>

namespace UtlGrid {

    // Function to split image in grid with separate depth division
    std::vector<std::vector<cv::Mat>>
    split3DImageIntoCubes(const std::vector<cv::Mat> &volume, int gridDivision, int depthDivision);

    // Function to mergeCubes grid back to image with separate depth division
    std::vector<cv::Mat>
    mergeCubesInto3DImage(const std::vector<std::vector<cv::Mat>> &subVolumes, int gridDivision, int depthDivision,
                          int originalDepth, int originalHeight, int originalWidth);

    void
    getMinXYZ(const std::vector<std::vector<cv::Mat>> &split_vec, int &new_size_x, int &new_size_y, int &new_size_z);

    //INFO borderType: cv::BORDER_REFLECT/cv::BORDER_REPLICATE/cv::BORDER_CONSTANT
    void extendImage(std::vector<cv::Mat> &image3D, int &padding, int borderType);

    std::vector<std::vector<cv::Mat>> splitWithoutCubePadding(std::vector<cv::Mat> &image3D, int cubeSize, int padding);

    std::vector<std::vector<cv::Mat>> splitWithCubePadding(std::vector<cv::Mat> &image3D, int cubeSize, int padding, int cubePadding);

    void cropCubePadding(std::vector<std::vector<cv::Mat>>& split, int cubePadding);

    //INFO cubes have to be cropped from cubePadding before merge
    std::vector<cv::Mat> mergeCubes(const std::vector<std::vector<cv::Mat>>& cubes, int imageWidth, int imageHeight, int imageDepth, int cubeSize);

    void adjustCubeOverlap(std::vector<std::vector<cv::Mat>>& cubes, int cubePadding);



    }