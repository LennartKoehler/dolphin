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
#include "dolphin/psf/PSF.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"

namespace fs = std::filesystem;
std::string getFilenameFromPath(const std::string& path) {
    fs::path filePath(path);
    return filePath.stem().string();
}

void PSF::readFromTiffFile(const std::string& path){
    std::optional<Image3D> image_o = TiffReader::readTiffFile(path, 0);
    if (image_o.has_value()){
        image = image_o.value();
    }
    else throw std::runtime_error("Unable to read psf");
    ID = getFilenameFromPath(path);
}


void PSF::writeToTiffFile(const std::string& path){
    std::string filename = path + "/" + ID + ".tiff";
    TiffWriter::writeToFile(filename , image);
}

