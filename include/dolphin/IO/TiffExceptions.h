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