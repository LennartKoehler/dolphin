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
 * @file fft_backend_test.cpp
 * @brief Test file to verify forward and backward FFT using the backend
 *
 * This test:
 * 1. Reads an input image using TiffReader
 * 2. Gets a backend from BackendFactory
 * 3. Performs forward FFT
 * 4. Performs backward FFT
 * 5. Writes the result to an output file
 */

#include <iostream>
#include <memory>
#include <string>
#include <cmath>

#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/Image3D.h"
#include "dolphin/Logging.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendManager.h"
#include "dolphinbackend/ComplexData.h"

int main(int argc, char** argv) {
    std::cout << "=== FFT Backend Test ===" << std::endl;

    Logging::init();
    // Parse command line arguments
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_image.tif> <output_image.tif> [backend=cpu]" << std::endl;
        std::cout << "  backend: cpu (default) or cuda (if compiled with CUDA support)" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    std::string backendName = (argc > 3) ? argv[3] : "cpu";

    std::cout << "Input file: " << inputFile << std::endl;
    std::cout << "Output file: " << outputFile << std::endl;
    std::cout << "Backend: " << backendName << std::endl;

    try {
        // Step 1: Read the input image
        std::cout << "\n[Step 1] Reading input image..." << std::endl;
        int channel = 0;  // Read first channel
        auto optImage = TiffReader::readTiffFile(inputFile, channel);

        if (!optImage) {
            std::cerr << "Error: Failed to read image from " << inputFile << std::endl;
            return 1;
        }

        Image3D image = std::move(*optImage);
        CuboidShape imageShape = image.getShape();
        std::cout << "Image shape: " << imageShape.width << " x " << imageShape.height
                  << " x " << imageShape.depth << std::endl;

        // Step 2: Get the backend
        std::cout << "\n[Step 2] Getting backend manager..." << std::endl;
        IBackendManager& manager = BackendFactory::getInstance().getBackendManager(backendName);

        // Create backend configuration
        BackendConfig config;
        config.nThreads = 1;

        // Get the backend
        std::cout << "\n[Step 3] Creating backend..." << std::endl;
        IBackend& backend = manager.getBackend(config);
        std::cout << "Backend device: " << backend.getDeviceString() << std::endl;

        // Get the deconvolution backend (for FFT operations)
        IDeconvolutionBackend& deconvBackend = backend.mutableDeconvManager();

        // Get the memory manager
        IBackendMemoryManager& memoryManager = backend.mutableMemoryManager();

        // Step 3: Convert image to RealData
        std::cout << "\n[Step 4] Converting image to RealData..." << std::endl;
        RealData inputReal = Preprocessor::convertImageToRealData(image);
        std::cout << "RealData created with shape: " << inputReal.getSize().width << " x "
                  << inputReal.getSize().height << " x " << inputReal.getSize().depth << std::endl;

        // Step 4: Copy data to device
        std::cout << "\n[Step 5] Copying data to device..." << std::endl;
        RealData inputOnDevice = memoryManager.copyDataToDevice(inputReal);
        std::cout << "Data copied to device" << std::endl;

        // Step 5: Perform forward FFT (Real -> Complex)
        std::cout << "\n[Step 6] Performing forward FFT..." << std::endl;
        CuboidShape shape = inputReal.getSize();

        // Allocate complex output on device
        ComplexData complexOnDevice = memoryManager.allocateMemoryOnDeviceComplex(shape);

        // Perform the forward FFT
        deconvBackend.forwardFFT(inputOnDevice, complexOnDevice);
        backend.sync();
        std::cout << "Forward FFT completed" << std::endl;

        // Step 6: Perform backward FFT (Complex -> Real)
        std::cout << "\n[Step 7] Performing backward FFT..." << std::endl;

        // Allocate real output on device
        RealData outputOnDevice = memoryManager.allocateMemoryOnDeviceReal(shape);

        // Perform the backward FFT
        deconvBackend.backwardFFT(complexOnDevice, outputOnDevice);
        backend.sync();
        std::cout << "Backward FFT completed" << std::endl;

        // Step 7: Copy result back to host
        std::cout << "\n[Step 8] Copying result back to host..." << std::endl;
        RealData outputReal = memoryManager.moveDataFromDevice(
            outputOnDevice,
            BackendFactory::getInstance().getDefaultBackendMemoryManager()
        );
        std::cout << "Data copied back to host" << std::endl;

        // Step 8: Convert RealData back to Image3D
        std::cout << "\n[Step 9] Converting RealData back to Image3D..." << std::endl;
        Image3D outputImage = Preprocessor::convertRealDataToImage(outputReal);
        std::cout << "Output image created with shape: " << outputImage.getShape().width << " x "
                  << outputImage.getShape().height << " x " << outputImage.getShape().depth << std::endl;

        if (outputImage == optImage){
            std::cout << "\n=== FFT Backend Test PASSED ===" << std::endl;
            return 0;
        }
        std::cout << "\n=== FFT Backend Test FAILED ===" << std::endl;
        return 1;


        //
        // // Step 9: Write the output image
        // std::cout << "\n[Step 10] Writing output image..." << std::endl;
        // if (!TiffWriter::writeToFile(outputFile, outputImage)) {
        //     std::cerr << "Error: Failed to write output image to " << outputFile << std::endl;
        //     return 1;
        // }
        // std::cout << "Output image written successfully to " << outputFile << std::endl;
        //
        // // Compute some statistics for verification
        // std::cout << "\n[Verification] Computing statistics..." << std::endl;
        // float minVal = std::numeric_limits<float>::max();
        // float maxVal = std::numeric_limits<float>::lowest();
        // float sum = 0.0f;
        // size_t count = 0;
        //
        // CuboidShape outShape = outputImage.getShape();
        // for (int z = 0; z < outShape.depth; ++z) {
        //     for (int y = 0; y < outShape.height; ++y) {
        //         for (int x = 0; x < outShape.width; ++x) {
        //             float val = outputImage.getPixel(x, y, z);
        //             minVal = std::min(minVal, val);
        //             maxVal = std::max(maxVal, val);
        //             sum += val;
        //             ++count;
        //         }
        //     }
        // }
        //
        // float mean = sum / count;
        // std::cout << "Output image statistics:" << std::endl;
        // std::cout << "  Min: " << minVal << std::endl;
        // std::cout << "  Max: " << maxVal << std::endl;
        // std::cout << "  Mean: " << mean << std::endl;
        //
        // std::cout << "\n=== FFT Backend Test PASSED ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
