/*
Test for padding types (linear, quadratic, sinusoid, gaussian).

Usage: linear_padding_test <input.tif> <output.tif> [padding_type] [pad_width] [pad_height] [pad_depth] [shape_scale]

Reads a TIFF image, applies the specified padding, and writes the padded image.

padding_type: linear | quadratic | sinusoid | gaussian  (default: linear)
shape_scale: float, controls shape for quadratic and gaussian (default: 1.0)
*/

#include "dolphin/Image3D.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include <iostream>
#include <string>
#include <map>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.tif> <output.tif> [padding_type] [pad_width] [pad_height] [pad_depth] [shape_scale]" << std::endl;
        std::cerr << "  padding_type: linear | quadratic | sinusoid | gaussian (default: linear)" << std::endl;
        std::cerr << "  Default padding: 20 on each side per dimension" << std::endl;
        std::cerr << "  Default shape_scale: 1.0" << std::endl;
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    std::string paddingTypeName = "linear";
    if (argc >= 4) paddingTypeName = argv[3];

    int padWidth = 20;
    int padHeight = 20;
    int padDepth = 20;

    if (argc >= 5) padWidth = std::stoi(argv[4]);
    if (argc >= 6) padHeight = std::stoi(argv[5]);
    if (argc >= 7) padDepth = std::stoi(argv[6]);

    float shapeScale = 1.0f;
    if (argc >= 8) shapeScale = std::stof(argv[7]);

    std::map<std::string, PaddingFillType> typeMap = {
        {"linear", PaddingFillType::LINEAR},
        {"quadratic", PaddingFillType::QUADRATIC},
        {"sinusoid", PaddingFillType::SINUSOID},
        {"gaussian", PaddingFillType::GAUSSIAN},
    };

    auto it = typeMap.find(paddingTypeName);
    if (it == typeMap.end()) {
        std::cerr << "Unknown padding type: " << paddingTypeName << std::endl;
        std::cerr << "Available types: linear, quadratic, sinusoid, gaussian" << std::endl;
        return 1;
    }
    PaddingFillType paddingType = it->second;

    try {
        // Read input image
        int channel = 0;
        auto maybeImage = TiffReader::readTiffFile(inputPath, channel);
        if (!maybeImage.has_value()) {
            std::cerr << "Failed to read input image: " << inputPath << std::endl;
            return 2;
        }

        Image3D image = std::move(maybeImage.value());
        CuboidShape shape = image.getShape();
        std::cout << "Input image size: " << shape.width << " x " << shape.height << " x " << shape.depth << std::endl;

        // Set up symmetric padding
        Padding padding{
            CuboidShape{padWidth, padHeight, padDepth},   // before (left, top, front)
            CuboidShape{padWidth, padHeight, padDepth}    // after  (right, bottom, back)
        };

        // Apply padding
        Preprocessor::padImage(image, padding, paddingType, shapeScale);

        CuboidShape paddedShape = image.getShape();
        std::cout << "Padded image size: " << paddedShape.width << " x " << paddedShape.height << " x " << paddedShape.depth << std::endl;

        // Write output
        bool ok = TiffWriter::writeToFile(outputPath, image);
        if (!ok) {
            std::cerr << "Failed to write output image: " << outputPath << std::endl;
            return 3;
        }

        std::cout << "Successfully wrote padded image to: " << outputPath << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 4;
    }

    return 0;
}
