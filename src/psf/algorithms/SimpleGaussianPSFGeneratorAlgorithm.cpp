//
// Created by christoph on 12.05.24.
//

#include <opencv2/core.hpp>
#include "SimpleGaussianPSFGeneratorAlgorithm.h"


//LK is this supposed to be the same as the gaussian psf generator?
PSF SimpleGaussianPSFGeneratorAlgorithm::generatePSF() const {
    int width = sizeX, height = sizeY;  // Größe des Bildes
    double meanX = width / 2, meanY = height / 2;  // Mittelpunkte
    int layers = sizeZ;  // Anzahl der Schichten
    std::vector<cv::Mat> sphereLayers;  // Vektor für die Gaußschen Schichten

    double sigmaXPercent = sigmaX, sigmaYPercent = sigmaY, sigmaZPercent = sigmaZ; // Prozentangaben für Sigma
    double sigmaXBase;
    double sigmaYBase;
    if(sizeX > sizeY){
        sigmaXBase = height * (sigmaXPercent / 100.0);
        sigmaYBase = height * (sigmaYPercent / 100.0);
    }else if(sizeY > sizeX){
        sigmaXBase = width * (sigmaXPercent / 100.0);
        sigmaYBase = width * (sigmaYPercent / 100.0);
    }else{
        sigmaXBase = width * (sigmaXPercent / 100.0);
        sigmaYBase = height * (sigmaYPercent / 100.0);
    }

    //double sigmaZBase = layers * (sigmaZPercent / 100.0);  // Basiswert für sigmaZ

    double sigmaXFactor = 2.0 * sigmaXBase / layers;
    double sigmaYFactor = 2.0 * sigmaYBase / layers;
    //double sigmaZFactor = 2.0 * sigmaZBase / layers;

    for (int i = 0; i < layers; i++) {
        double factor = std::abs(i - layers / 2);
        double sigmaX = std::max(sigmaXBase - factor * sigmaXFactor, 1.0);
        double sigmaY = std::max(sigmaYBase - factor * sigmaYFactor, 1.0);
        //double intensityScale = 1.0 - std::abs(i - layers / 2) * (1.0 / (layers / 2));  // Intensitätsskalierung
        //TODO type an bild anpassen
        // siehe getDatatype()
        cv::Mat gauss(height, width, CV_32F);
        for (int y = 0; y < gauss.rows; y++) {
            for (int x = 0; x < gauss.cols; x++) {
                float value = exp(-0.5 * (pow((x - meanX) / sigmaX, 2.0) + pow((y - meanY) / sigmaY, 2.0))); //* intensityScale;
                gauss.at<float>(y, x) = value;
            }
        }
        //TODO debug
        //cv::normalize(gauss, gauss, 0, 255, cv::NORM_MINMAX);
        //std::cout << "DATENTYPE: " << image.layers[0].data.type() << std::endl;
        //gauss.convertTo(gauss, image.layers[0].data.type());

        // Hinzufügen zur Liste der Schichten

        //double min, max;
        //cv::minMaxLoc(gauss, &min, &max);
        //gauss.convertTo(gauss, CV_32F, 0.000001/max);// / (max - min), -min * 255.0 / (max - min));

        sphereLayers.push_back(gauss);
    }

    //TODO debug option
    /*for (const auto& layer : sphereLayers) {
        cv::imshow("Layer", layer);
        cv::waitKey(50);
    }

    cv::destroyAllWindows();*/
    Image3D psfImage;
    psfImage.slices = sphereLayers;
    PSF gaussianPsf;
    gaussianPsf.image = psfImage;

    return gaussianPsf;

}

