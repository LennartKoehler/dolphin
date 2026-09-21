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

#pragma once
#include <stdexcept>
#include <string>

class TiffException : public std::runtime_error {
public:
    explicit TiffException(const std::string& message) : std::runtime_error(message) {}
};

class TiffFileOpenException : public TiffException {
public:
    explicit TiffFileOpenException(const std::string& filename) 
        : TiffException("Cannot open TIFF file: " + filename) {}
};

class TiffReadException : public TiffException {
public:
    explicit TiffReadException(const std::string& message) 
        : TiffException("TIFF read error: " + message) {}
};

class TiffWriteException : public TiffException{
public:
    explicit TiffWriteException(const std::string& message)
        :TiffException("Tiff write error: " + message) {}
};

class TiffMetadataException : public TiffException {
public:
    explicit TiffMetadataException(const std::string& message) 
        : TiffException("TIFF metadata error: " + message) {}
};

class TiffMemoryException : public TiffException {
public:
    explicit TiffMemoryException(const std::string& message) 
        : TiffException("TIFF memory error: " + message) {}
};
