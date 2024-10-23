#include <opencv2/core.hpp>
#include "BornWolfModel.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <complex>

// Beispiel-Implementierung der PSF nach dem Born-Wolf-Modell
PSF BornWolfModel::generatePSF() const {
    // Berechne die Größe und sorge dafür, dass Breite, Höhe und Tiefe ungerade sind
    int width = (sizeX % 2 == 0) ? sizeX + 1 : sizeX;
    int height = (sizeY % 2 == 0) ? sizeY + 1 : sizeY;
    int layers = (sizeZ % 2 == 0) ? sizeZ + 1 : sizeZ;

    // Mittelpunkte der PSF (sicherstellen, dass die PSF zentriert ist)
    double meanX = width / 2.0;
    double meanY = height / 2.0;
    double meanZ = layers / 2.0;

    std::vector<cv::Mat> psfLayers;  // Vektor für die PSF-Schichten

    // Physikalische Parameter
    double wavelength = 488.0;  // Wellenlänge in Nanometern (nm)
    double refractiveIndex = 1.515;  // Brechungsindex des Immersionsöls
    double NA = 1.4;  // Numerische Apertur des Objektivs
    double voxelSizeZ = 1.0;// 500;
    double voxelSizeXY =1.0;// 113.5;

    // Erforderliche Berechnungen für das Born-Wolf-Modell
    double k = 2 * M_PI * refractiveIndex / wavelength;  // Wellenzahl
    double maxTheta = asin(NA / refractiveIndex);  // Maximaler Winkel (Halböffnung)

    // Erzeuge die PSF basierend auf dem Born-Wolf-Modell
    for (int z = 0; z < layers; ++z) {
        double dz = (z - meanZ) * voxelSizeZ;  // Abstandsmaß in Z-Richtung (z in physikalischen Einheiten)
        cv::Mat psfSlice(height, width, CV_32F);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                double dx = (x - meanX) * voxelSizeXY;  // Abstand in X-Richtung (in physikalischen Einheiten)
                double dy = (y - meanY) * voxelSizeXY;  // Abstand in Y-Richtung (in physikalischen Einheiten)

                // Radialer Abstand von der Mitte
                double r = sqrt(dx * dx + dy * dy);

                // Berechnung der Bessel-Funktion und der Phasenverschiebung
                // Der Faktor 2 * J1(k * r * sin(maxTheta)) / (k * r * sin(maxTheta)) ergibt den lateralen Beitrag
                double besselFactor = 2.0 * j1(k * r * sin(maxTheta)) / (k * r * sin(maxTheta));

                // Berechnung des axiale Intensitätsbeitrags (Longitudinale Komponente)
                double axialFactor = cos(k * dz * cos(maxTheta));

                // Berechnung des PSF-Wertes an dieser Stelle
                float value = static_cast<float>(besselFactor * axialFactor);
                psfSlice.at<float>(y, x) = value;
            }
        }

        // Füge die Schicht zur Liste hinzu
        psfLayers.push_back(psfSlice);
    }

    // Normiere die PSF so, dass die Summe aller Werte 1 ergibt
    double sum = 0.0;
    for (const auto& layer : psfLayers) {
        sum += cv::sum(layer)[0];
    }

    for (auto& layer : psfLayers) {
        layer /= sum;
    }

    // Erstelle das PSF-Objekt und fülle es mit den erzeugten Schichten
    Image3D psfImage;
    psfImage.slices = psfLayers;
    PSF bornWolfPsf;
    bornWolfPsf.image = psfImage;

    return bornWolfPsf;
}