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

/**
 * @file fft_real_vs_complex_test.cpp
 * @brief Test comparing Real FFT vs Complex FFT results
 *
 * This test compares two approaches:
 * 1. Real-to-Complex FFT -> Complex-to-Real FFT (standard approach)
 * 2. Complex-to-Complex FFT -> Complex-to-Complex FFT (all in complex space)
 *
 * For real-valued input data, both approaches should produce identical results.
 */

#include <iostream>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <limits>

#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/Image3D.h"
#include "dolphin/Logging.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"

int main(int argc, char** argv) {
    std::cout << "=== FFT Real vs Complex Comparison Test ===" << std::endl;

    // Parse command line arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input_image.tif> <output_real_fft.tif> <output_complex_fft.tif> [backend=cpu]" << std::endl;
        std::cout << "  backend: cpu (default) or cuda (if compiled with CUDA support)" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputRealFFT = argv[2];
    std::string outputComplexFFT = argv[3];
    std::string backendName = (argc > 4) ? argv[4] : "cpu";

    std::cout << "Input file: " << inputFile << std::endl;
    std::cout << "Backend: " << backendName << std::endl;

    try {
        // Step 1: Read the input image
        std::cout << "\n[Step 1] Reading input image..." << std::endl;
        int channel = 0;
        auto optImage = TiffReader::readTiffFile(inputFile, channel);

        if (!optImage) {
            std::cerr << "Error: Failed to read image from " << inputFile << std::endl;
            return 1;
        }

        Image3D inputImage = std::move(*optImage);
        CuboidShape imageShape = inputImage.getShape();
        std::cout << "Image shape: " << imageShape.width << " x " << imageShape.height
                  << " x " << imageShape.depth << std::endl;

        // Step 2: Get the backend
        std::cout << "\n[Step 2] Getting backend..." << std::endl;
        IBackendManager& manager = BackendFactory::getInstance().getBackendManager(backendName);
        BackendConfig config;
        config.nThreads = 1;
        IBackend& backend = manager.getBackend(config);
        IDeconvolutionBackend& deconvBackend = backend.mutableDeconvManager();
        IBackendMemoryManager& memoryManager = backend.mutableMemoryManager();
        std::cout << "Backend device: " << backend.getDeviceString() << std::endl;

        Image3D outputImage2;
        Image3D outputImage1;
        {
            //======================================================================
            // Approach 1: Real-to-Complex FFT -> Complex-to-Real FFT (Standard)
            //======================================================================
            // Step 3: Convert image to RealData
            std::cout << "\n[Step 3] Converting image to RealData..." << std::endl;
            RealData inputReal = Preprocessor::convertImageToRealData(inputImage);

            // Copy to device
            RealData inputOnDevice = memoryManager.copyDataToDevice(inputReal);
            std::cout << "Data copied to device" << std::endl;
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "APPROACH 1: Real-to-Complex -> Complex-to-Real" << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            std::cout << "\n[Approach 1 - Step 1] Forward FFT (Real -> Complex)..." << std::endl;
            ComplexData complexFromReal = memoryManager.allocateMemoryOnDeviceComplex(imageShape);
            deconvBackend.forwardFFT(inputOnDevice, complexFromReal);
            backend.sync();
            std::cout << "Forward FFT completed" << std::endl;

            std::cout << "\n[Approach 1 - Step 2] Backward FFT (Complex -> Real)..." << std::endl;
            RealData resultRealFromComplex = memoryManager.allocateMemoryOnDeviceRealFFTInPlace(imageShape);
            deconvBackend.backwardFFT(complexFromReal, resultRealFromComplex);
            backend.sync();
            std::cout << "Backward FFT completed" << std::endl;

            // Copy back and convert to image
            std::cout << "\n[Approach 1 - Step 3] Converting to Image3D..." << std::endl;
            RealData resultRealHost1 = memoryManager.moveDataFromDevice(
                resultRealFromComplex,
                BackendFactory::getInstance().getDefaultBackendMemoryManager()
            );
            outputImage1 = Preprocessor::convertRealDataToImage(resultRealHost1);
        }
        {

            //======================================================================
            // Approach 1: Real-to-Complex FFT -> Complex-to-Real FFT (Standard)
            //======================================================================
            // Step 3: Convert image to RealData
            std::cout << "\n[Step 3] Converting image to RealData..." << std::endl;
            RealData inputReal = Preprocessor::convertImageToRealData(inputImage);

            // Copy to device
            RealData inputOnDevice = memoryManager.copyDataToDevice(inputReal);

            ComplexData resultOutOfPlace = memoryManager.allocateMemoryOnDeviceComplex(imageShape);

            std::cout << "Data copied to device" << std::endl;
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "APPROACH 2: Real-to-ComplexInterpreted -> Complex-to-RealInterpreted" << std::endl;
            std::cout << std::string(60, '=') << std::endl;


            std::cout << "\n[Approach 2 - Step 2] Forward FFT (Real -> Complex)..." << std::endl;
            ComplexView complexResult= memoryManager.reinterpret(inputOnDevice);
            //inplace
            deconvBackend.forwardFFT(inputOnDevice, complexResult);
            backend.sync();
            std::cout << "Forward FFT completed" << std::endl;

            std::cout << "\n[Approach 2 - Step 3] Backward FFT (Complex -> Real)..." << std::endl;
            RealView resultReal = memoryManager.reinterpret(complexResult);
            deconvBackend.backwardFFT(complexResult, resultReal);
            backend.sync();
            std::cout << "Backward FFT completed" << std::endl;


            // Copy back and convert to image
            std::cout << "\n[Approach 2 - Step 5] Converting to Image3D..." << std::endl;
            RealData resultRealHost2 = memoryManager.moveDataFromDevice(
                resultReal,
                BackendFactory::getInstance().getDefaultBackendMemoryManager()
            );
            outputImage2 = Preprocessor::convertRealDataToImage(resultRealHost2);
        }

        TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/test1.tif", outputImage1);
        TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/test2.tif", outputImage2);

        if (outputImage1 == outputImage2  && outputImage2 == inputImage){
            std::cout << "\n=== FFT Backend Test PASSED ===" << std::endl;
            return 0;
        }
        std::cout << "\n=== FFT Backend Test FAILED ===" << std::endl;
        return 1;



    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
