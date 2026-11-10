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

#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include <cmath>
#include <cstring>
#include <itkConstantPadImageFilter.h>
#include <itkMirrorPadImageFilter.h>
#include <itkRegionOfInterestImageFilter.h>
#include <itkImageDuplicator.h>


ComplexData Preprocessor::convertImageToComplexData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    ComplexData result = BackendFactory::getInstance().getHostBackendMemoryManager().allocateMemoryOnDeviceComplexFull(shape);

    size_t index = 0;

    for (const auto& it : input) {
        result[index][0] = static_cast<real_t>(it);
        result[index][1] = 0.0;
        index ++;
    }

    return result;
}

void Preprocessor::convertImageToRealData(
    const Image3D& input,
    RealData& allocatedBuffer) {

    size_t index = 0;

    assert (allocatedBuffer.isValid() && allocatedBuffer.getPaddedSize() >= input.getShape());

    for (const auto& it : input) {
        allocatedBuffer[index] = static_cast<real_t>(it);
        index ++;
    }
}

RealData Preprocessor::convertImageToRealData(
    const Image3D& input) {

    CuboidShape shape = input.getShape();
    RealData result = BackendFactory::getInstance().getHostBackendMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(shape);
    // since the memory layout for inplace fft is noncontiguous the image pointer cant be reinterpreted as real_t*
    // thats also why this has to be done on the cpu first, and then another copy operation of the memory
    // with correct layout to gpu

    size_t index = 0;

    for (const auto& it : input) {
        result[index] = static_cast<real_t>(it);
        index ++;
    }

    return result;
}

Image3D Preprocessor::convertComplexDataToImage(
        const ComplexData& input){

    Image3D output(input.getSize(), 0.0f);

    size_t index = 0;
    for (auto& it : output) {
        real_t real = input[index][0];
        real_t imag = input[index][1];
        it = static_cast<float>(std::sqrt(real * real + imag * imag));
        index ++;
    }

    return output;
}

// TODO make this conversion part of the backend? because if data is on cuda device then this wont work
Image3D Preprocessor::convertRealDataToImage(
        const RealData& input){

    Image3D output(input.getSize(), 0.0f);
    // since the memory layout for inplace fft is noncontiguous the image pointer cant be reinterpreted as real_t*
    // thats also why this has to be done on the cpu first, and then another copy operation of the memory
    // with correct layout to gpu

    size_t index = 0;
    for (auto& it : output) {
        real_t real = input[index];
        it = static_cast<float>(real);
        index ++;
    }

    return output;
}

void Preprocessor::convertImageToRealDataPadded(
        Image3D&& interior, const Padding& padding, PaddingFillType fill, RealData& allocatedOutBuffer){

    const CuboidShape interiorShape = interior.getShape();
    const CuboidShape workShape = interiorShape + padding.before + padding.after;

    assert (allocatedOutBuffer.isValid() && allocatedOutBuffer.getSize() == workShape);

    if (fill != PaddingFillType::ZERO) {
        // Non-zero fills need the ITK pad filters: pad the interior image and
        // convert the padded result. The padded image is freed on return.
        Image3D padded = std::move(interior);
        ImagePadding::padImage(padded, padding, fill);
        convertImageToRealData(padded, allocatedOutBuffer);
        return;
    }

    // Zero everything (row padding included), then copy the interior once.
    std::memset(allocatedOutBuffer.getData(), 0, allocatedOutBuffer.getDataBytes());

    const size_t workWidth = workShape.width;
    const size_t workHeight = workShape.height;
    const size_t beforeWidth = padding.before.width;
    const size_t beforeHeight = padding.before.height;
    const size_t beforeDepth = padding.before.depth;

    for (size_t z = 0; z < interiorShape.depth; ++z) {
        for (size_t y = 0; y < interiorShape.height; ++y) {
            const size_t srcBase = (z * interiorShape.height + y) * interiorShape.width;
            const size_t dstBase = ((z + beforeDepth) * workHeight + (y + beforeHeight)) * workWidth + beforeWidth;
            for (size_t x = 0; x < interiorShape.width; ++x) {
                allocatedOutBuffer[dstBase + x] = interior[srcBase + x];
            }
        }
    }
}


// Padding function implementations have been moved to src/image/ImagePadding.cpp
// The Preprocessor namespace now delegates to ImagePadding namespace via inline functions in the header.

