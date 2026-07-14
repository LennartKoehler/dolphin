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


// Padding function implementations have been moved to src/image/ImagePadding.cpp
// The Preprocessor namespace now delegates to ImagePadding namespace via inline functions in the header.

