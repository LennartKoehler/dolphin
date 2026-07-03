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

#include <algorithm>
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

CuboidShape GaussianPSFGenerator::getPadding(PaddingStrategyType paddingType) const {
    switch(paddingType){
        case(NONE):
            return CuboidShape{0,0,0};
        case(PARENT):
            return CuboidShape{static_cast<size_t>(config->sigmaX * 4), static_cast<size_t>(config->sigmaY * 4), static_cast<size_t>(config->sigmaZ * 4)};
        case(FULL_PSF):
            return CuboidShape{config->sizeX, config->sizeY, config->sizeZ};
        case(MANUAL):
            return CuboidShape{SIZE_MAX, SIZE_MAX, SIZE_MAX};
        default:
            return CuboidShape{SIZE_MAX, SIZE_MAX, SIZE_MAX};
    }
}

ImageType::RegionType GaussianPSFGenerator::getNonNegligibleRegion(size_t width, size_t height, size_t layers,
                                                      double centerX, double centerY, double centerZ) const {
    constexpr double cutoffFactor = 5.0;

    size_t xMin = static_cast<size_t>(std::max(0L, static_cast<long>(std::floor(centerX - cutoffFactor * config->sigmaX))));
    size_t xMax = std::min(width - 1, static_cast<size_t>(std::max(0L, static_cast<long>(std::ceil(centerX + cutoffFactor * config->sigmaX)))));
    size_t yMin = static_cast<size_t>(std::max(0L, static_cast<long>(std::floor(centerY - cutoffFactor * config->sigmaY))));
    size_t yMax = std::min(height - 1, static_cast<size_t>(std::max(0L, static_cast<long>(std::ceil(centerY + cutoffFactor * config->sigmaY)))));
    size_t zMin = static_cast<size_t>(std::max(0L, static_cast<long>(std::floor(centerZ - cutoffFactor * config->sigmaZ))));
    size_t zMax = std::min(layers - 1, static_cast<size_t>(std::max(0L, static_cast<long>(std::ceil(centerZ + cutoffFactor * config->sigmaZ)))));

    ImageType::IndexType start;
    start[0] = static_cast<itk::IndexValueType>(xMin);
    start[1] = static_cast<itk::IndexValueType>(yMin);
    start[2] = static_cast<itk::IndexValueType>(zMin);

    ImageType::SizeType size;
    size[0] = xMax - xMin + 1;
    size[1] = yMax - yMin + 1;
    size[2] = zMax - zMin + 1;

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);

    return region;
}

PSF GaussianPSFGenerator::generatePSF() const {
    size_t width = config->sizeX, height = config->sizeY, layers = config->sizeZ;
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
    itkImage->FillBuffer(0.0f);

    // Compute the effective region where the Gaussian is non-negligible
    ImageType::RegionType clippedRegion = getNonNegligibleRegion(width, height, layers, centerX, centerY, centerZ);

    // Create iterator over only the clipped sub-region
    itk::ImageRegionIterator<ImageType> it(itkImage, clippedRegion);

    // Generate Gaussian PSF values
    double sum = 0.0;

    int64_t max = clippedRegion.GetNumberOfPixels();
    progressTracker.setMax(max);
    size_t counter = 0;
    int64_t step = std::max<int64_t>(1, max / 20);

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        counter ++;

        ImageType::IndexType index = it.GetIndex();
        size_t x = static_cast<size_t>(index[0]);
        size_t y = static_cast<size_t>(index[1]);
        size_t z = static_cast<size_t>(index[2]);

        double dx = (static_cast<double>(x) - centerX) / config->sigmaX;
        double dy = (static_cast<double>(y) - centerY) / config->sigmaY;
        double dz = (static_cast<double>(z) - centerZ) / config->sigmaZ;

        // Calculate 3D Gaussian value
        float value = static_cast<float>(exp(-0.5 * (dx * dx + dy * dy + dz * dz)));
        it.Set(value);
        sum += value;


        if (counter % step == 0){
            progressTracker.add(static_cast<float>(max)/20);
        }
    }

    // Normalize the PSF so that the sum of all values is 1
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set(it.Get() / sum);
    }

    return PSF(std::move(itkImage), config->ID);
}
