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

#include <cmath>
#include <itkImage.h>
#include <itkImageRegionIterator.h>
#include "dolphin/psf/generators/GaussianPSFGenerator.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"




void GaussianPSFGenerator::setConfig(const std::shared_ptr<const PSFConfig> config){
    auto* ucfg = dynamic_cast<const GaussianPSFConfig*>(config.get());
    if (!ucfg) throw std::runtime_error("Wrong config type");
    this->config = std::make_unique<GaussianPSFConfig>(*ucfg);

    // this->config.reset(static_cast<GaussianPSFConfig*>(config.release()));
}

bool GaussianPSFGenerator::hasConfig(){
    return config != nullptr;
}


PSF GaussianPSFGenerator::generatePSF() const {
    int width = config->sizeX, height = config->sizeY, layers = config->sizeZ;
    double centerX = (width - 1) / 2.0;
    double centerY = (height - 1) / 2.0;
    double centerZ = (layers - 1) / 2.0;

    // Create ITK image
    ImageType::Pointer itkImage = ImageType::New();

    // Set the image dimensions
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

    // Create iterator for the entire image
    itk::ImageRegionIterator<ImageType> it(itkImage, region);

    // Generate Gaussian PSF values
    double sum = 0.0;

    int max = width * height * layers;
    progressTracker.setMax(max);
    int counter = 0;

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        counter ++;

        ImageType::IndexType index = it.GetIndex();
        int x = index[0];
        int y = index[1];
        int z = index[2];

        double dx = (x - centerX) / config->sigmaX;
        double dy = (y - centerY) / config->sigmaY;
        double dz = (z - centerZ) / config->sigmaZ;

        // Calculate 3D Gaussian value
        float value = static_cast<float>(exp(-0.5 * (dx * dx + dy * dy + dz * dz)));
        it.Set(value);
        sum += value;


        if (counter % (max / 20) == 0){
            progressTracker.add(static_cast<float>(max)/20);
        }
    }

    // Normalize the PSF so that the sum of all values is 1
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() / sum);
    }

    return PSF(std::move(itkImage));
}
