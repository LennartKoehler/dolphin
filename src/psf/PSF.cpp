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

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <vector>
#include "dolphin/psf/PSF.h"
#include "dolphin_image/IO/TiffReader.h"
#include "dolphin_image/IO/TiffWriter.h"

namespace fs = std::filesystem;
std::string getFilenameFromPath(const std::string& path) {
    fs::path filePath(path);
    return filePath.stem().string();
}

void PSF::readFromTiffFile(const std::string& path){
    std::optional<Image3D> image_o = TiffReader::readTiffFile(path, 0);
    if (image_o.has_value()){
         this->image = image_o.value().getItkImage();
    }
    else throw std::runtime_error("Unable to read psf");
    ID = getFilenameFromPath(path);
}



void PSF::writeToTiffFile(const std::string& path){
    TiffWriter::writeToFile(path , image);
}

size_t computeEnergyHalfExtent(const std::vector<float>& profile, size_t peakIndex, double fraction) {
    double total = 0.0;
    for (float v : profile) {
        total += v;
    }
    if (total <= 0.0) return 0;

    double cumulative = profile[peakIndex] / total;
    for (size_t d = 1; d < profile.size() && cumulative < fraction; d++) {
        double sum = 0.0;
        if (peakIndex >= d) sum += profile[peakIndex - d];
        if (peakIndex + d < profile.size()) sum += profile[peakIndex + d];
        cumulative += sum / total;
        if (cumulative >= fraction) return d;
    }
    return 0;
}

PSFExtent PSF::computeEnergyExtent(double lateralFraction, double axialFraction) const {
    CuboidShape shape = getShape();
    size_t cx = (shape.width - 1) / 2;
    size_t cy = (shape.height - 1) / 2;

    std::vector<float> axialProfile(shape.depth);
    for (size_t z = 0; z < shape.depth; z++) {
        axialProfile[z] = getPixel(cx, cy, z);
    }

    auto maxIt = std::max_element(axialProfile.begin(), axialProfile.end());
    size_t zPeak = static_cast<size_t>(std::distance(axialProfile.begin(), maxIt));

    size_t zHalfExtent = computeEnergyHalfExtent(axialProfile, zPeak, axialFraction);

    std::vector<float> lateralProfile(shape.width);
    for (size_t x = 0; x < shape.width; x++) {
        lateralProfile[x] = getPixel(x, cy, zPeak);
    }

    size_t lateralExtent = computeEnergyHalfExtent(lateralProfile, cx, lateralFraction);

    return PSFExtent{zHalfExtent, lateralExtent};
}
