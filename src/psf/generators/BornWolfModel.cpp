/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "dolphin/psf/generators/BornWolfModel.h"
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


    // Physikalische Parameter
    double wavelength = 488.0;  // Wellenlänge in Nanometern (nm)
    double refractiveIndex = 1.515;  // Brechungsindex des Immersionsöls
    double NA = 1.4;  // Numerische Apertur des Objektivs
    double voxelSizeZ = 1.0;// 500;
    double voxelSizeXY =1.0;// 113.5;

    // Erforderliche Berechnungen für das Born-Wolf-Modell
    double k = 2 * M_PI * refractiveIndex / wavelength;  // Wellenzahl
    double maxTheta = asin(NA / refractiveIndex);  // Maximaler Winkel (Halböffnung)

    ImageType::Pointer itkImage = ImageType::New();
    ImageType::SizeType size;
    size[0] = width;
    size[1] = height;
    size[2] = layers;

    ImageType::IndexType start;
    start.Fill(0);

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    itkImage->SetRegions(region);
    itkImage->Allocate();

    itk::ImageRegionIterator<ImageType> it(itkImage, region);
    
    
    double sum = 0.0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it){
        ImageType::IndexType index = it.GetIndex();
        int x = index[0];
        int y = index[1];
        int z = index[2];

        double dx = (x - meanX) * voxelSizeXY;  // Abstand in X-Richtung (in physikalischen Einheiten)
        double dy = (y - meanY) * voxelSizeXY;  // Abstand in Y-Richtung (in physikalischen Einheiten)
        double dz = (z - meanZ) * voxelSizeZ;  // Abstandsmaß in Z-Richtung (z in physikalischen Einheiten)
        // Radialer Abstand von der Mitte
        double r = sqrt(dx * dx + dy * dy);

        // Berechnung der Bessel-Funktion und der Phasenverschiebung
        // Der Faktor 2 * J1(k * r * sin(maxTheta)) / (k * r * sin(maxTheta)) ergibt den lateralen Beitrag
        double besselFactor = 2.0 * j1(k * r * sin(maxTheta)) / (k * r * sin(maxTheta));

        // Berechnung des axiale Intensitätsbeitrags (Longitudinale Komponente)
        double axialFactor = cos(k * dz * cos(maxTheta));

        // Berechnung des PSF-Wertes an dieser Stelle
        double value = besselFactor * axialFactor;
        it.Set(value);
        sum += value;
    }


    // Normiere die PSF so, dass die Summe aller Werte 1 ergibt
    
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() / sum);
    }



    // Erstelle das PSF-Objekt und fülle es mit den erzeugten Schichten
    Image3D psfImage(std::move(itkImage));
    PSF bornWolfPsf;
    bornWolfPsf.image = psfImage;

    return bornWolfPsf;
}