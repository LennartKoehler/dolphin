#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "GaussianPSFGeneratorAlgorithm.h"

PSF GaussianPSFGeneratorAlgorithm::generatePSF() const {
    int width = sizeX, height = sizeY, layers = sizeZ;  // Größe des Bildes
    double centerX = (width - 1) / 2.0;
    double centerY = (height - 1) / 2.0;
    double centerZ = (layers - 1) / 2.0;

    std::vector<cv::Mat> sphereLayers;  // Vektor für die Gaußschen Schichten

    double sigmaXBase = width * (sigmaX / 100.0);
    double sigmaYBase = height * (sigmaY / 100.0);
    double sigmaZBase = layers * (sigmaZ / 100.0);

    for (int z = 0; z < layers; ++z) {
        double dz = (z - centerZ) / sigmaZBase;  // Abstandsmaß in z-Richtung
        cv::Mat gauss(height, width, CV_32F);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double dx = (x - centerX) / sigmaXBase;
                double dy = (y - centerY) / sigmaYBase;

                // Berechne den 3D-Gauss-Wert
                float value = static_cast<float>(exp(-0.5 * (dx * dx + dy * dy + dz * dz)));
                gauss.at<float>(y, x) = value;
            }
        }

        sphereLayers.push_back(gauss);
    }

    // Normiere die PSF so, dass die Summe aller Werte 1 ergibt
    double sum = 0.0;
    for (const auto& layer : sphereLayers) {
        sum += cv::sum(layer)[0];
    }

    for (auto& layer : sphereLayers) {
        layer /= sum;
    }

    // Erstelle das PSF-Objekt und fülle es mit den erzeugten Schichten
    Image3D psfImage;
    psfImage.slices = sphereLayers;
    PSF gaussianPsf;
    gaussianPsf.image = psfImage;

    return gaussianPsf;
}
