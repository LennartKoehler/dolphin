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
#include "psf/PSF.h"
#include "IO/TiffReader.h"
#include "IO/TiffWriter.h"

namespace fs = std::filesystem;
std::string getFilenameFromPath(const std::string& path) {
    fs::path filePath(path);
    return filePath.stem().string();
}

void PSF::readFromTiffFile(const std::string& path){
    image = TiffReader::readTiffFile(path, 0);
    ID = getFilenameFromPath(path);
}


void PSF::writeToTiffFile(const std::string& path){
    TiffWriter::writeToFile(path, image);
}

