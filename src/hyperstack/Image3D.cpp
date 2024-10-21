//
// Created by christoph on 08.05.24.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Image3D.h"

float Image3D::getPixel(int x, int y, int z) {
    return this->slices[z].at<float>(x,y);
}

bool Image3D::showSlice(int z) {
    if (!this->slices.empty()) {
        if(z < 0 || z > size(this->slices)){
            std::cerr << "Slice " << std::to_string(z) << " out of Range" << std::endl;
            return false;
        }else {
            cv::Mat& slice = this->slices[z];

            if (slice.type() != CV_32F) {
                std::cerr << "Expected CV_32F data type." << std::endl;
                return false;
            }
            if (slice.empty()) {
                std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
                return false;
            }

            // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
            cv::Mat img8bit;
            double minVal, maxVal;
            cv::minMaxLoc(slice, &minVal, &maxVal);

            // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
            // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
            slice.convertTo(img8bit, CV_8U, 255.0 / maxVal);
            cv::imshow("Slice " + std::to_string(z), img8bit);
            cv::waitKey();

            std::cout << "Slice " + std::to_string(z) + " shown" << std::endl;
            return true;
        }
    }else{
        std::cerr<< "Layer " <<  std::to_string(z) << " cannot shown" << std::endl;
        return false;
    }
}

bool Image3D::show() {
    for (const auto &mat : this->slices) {
        if (mat.type() != CV_32F) {
            std::cerr << "Expected CV_32F data type." << std::endl;
            continue;
        }
        if (mat.empty()) {
            std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
            continue;            }

        // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
        cv::Mat img8bit;
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);

        // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
        // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
        mat.convertTo(img8bit, CV_8U, 255.0 / maxVal);
        cv::imshow("Image3D Slices", img8bit);
        cv::waitKey();
    }

    return true;
}
