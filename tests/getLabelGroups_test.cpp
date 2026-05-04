/*
Test for LabeledDeconvolutionExecutor::getLabelGroups and makeMasksWeighted

Reads a label image from a path given as the first command-line argument,
creates a dummy PSF vector and a psfLabelMap, calls getLabelGroups,
applies weighted feathering via makeMasksWeighted, and saves each resulting
label mask image (both binary and weighted) to a TIFF file.

The feathering kernel can be either generated as a Gaussian or loaded from
a TIFF image file.

Usage:
    getLabelGroups_test <label_image_path> [output_dir] [feathering_kernel_image]

If feathering_kernel_image is provided, it is used as the feathering kernel
instead of generating a default Gaussian kernel.
*/

#include "dolphin/deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/Image3D.h"
#include "dolphin/HelperClasses.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphin/Logging.h"
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Subclass to expose protected methods for testing
class TestableLabeledDeconvolutionExecutor : public LabeledDeconvolutionExecutor {
public:
    using LabeledDeconvolutionExecutor::getLabelGroups;
    using LabeledDeconvolutionExecutor::makeMasksWeighted;
    using LabeledDeconvolutionExecutor::createGaussianKernel;
};

int main(int argc, char* argv[]){

    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <label_image_path> [output_dir] [feathering_kernel_image]" << std::endl;
        return 1;
    }

    Logging::init();

    std::string labelImagePath = argv[1];
    std::string outputDir = "./label_test_output";
    std::string featheringKernelPath;
    if (argc >= 3){
        outputDir = argv[2];
    }
    if (argc >= 4){
        featheringKernelPath = argv[3];
    }

    // Create output directory if it doesn't exist
    fs::create_directories(outputDir);

    // Read label image
    int channel = 0;
    auto maybeLabelImage = TiffReader::readTiffFile(labelImagePath, channel);
    if (!maybeLabelImage.has_value()){
        std::cerr << "Failed to read label image: " << labelImagePath << std::endl;
        return 1;
    }
    Image3D labelImage = std::move(*maybeLabelImage);

    CuboidShape shape = labelImage.getShape();
    std::cout << "Label image shape: " << shape.print() << std::endl;

    // Create dummy PSFs with IDs that match the label map
    // We create a small 3x3x3 dummy PSF for each ID
    auto createDummyPSF = [](const std::string& id) -> std::shared_ptr<PSF> {
        Image3D psfImage(CuboidShape{3, 3, 3}, 1.0f);
        return std::make_shared<PSF>(std::move(psfImage), id);
    };

    std::vector<std::shared_ptr<PSF>> psfs;
    psfs.push_back(createDummyPSF("psf_a"));
    psfs.push_back(createDummyPSF("psf_b"));
    psfs.push_back(createDummyPSF("psf_c"));

    // Create psfLabelMap: label value -> PSF ID(s)
    // label 1 maps to psf_a, label 2 maps to psf_b, label 3 maps to psf_c
    RangeMap<std::string> psfLabelMap;
    psfLabelMap.addRange(0, 1, "psf_a");
    psfLabelMap.addRange(1, 200, "psf_b");
    psfLabelMap.addRange(240, 257, "psf_c");
    // Labels 0 and others without mapping will not produce masks

    // Call getLabelGroups via the testable subclass
    TestableLabeledDeconvolutionExecutor executor;
    std::vector<Label<Image3D>> labelGroups = executor.getLabelGroups(psfs, labelImage, psfLabelMap);

    std::cout << "Number of label groups: " << labelGroups.size() << std::endl;

    // --- Save binary (unweighted) label masks ---
    for (size_t i = 0; i < labelGroups.size(); i++){
        const Image3D* mask = labelGroups[i].getMask();
        if (!mask){
            std::cerr << "Label group " << i << " has no mask!" << std::endl;
            continue;
        }

        std::vector<std::shared_ptr<PSF>> groupPsfs = labelGroups[i].getPSFs();
        std::string psfIds;
        for (size_t j = 0; j < groupPsfs.size(); j++){
            if (j > 0) psfIds += "_";
            psfIds += groupPsfs[j]->ID;
        }

        std::string outputPath = outputDir + "/label_mask_binary_" + std::to_string(i) + "_" + psfIds + ".tif";
        bool success = TiffWriter::writeToFile(outputPath, *mask);
        if (success){
            std::cout << "Saved binary label mask " << i << " (PSFs: " << psfIds << ") to " << outputPath << std::endl;
        } else {
            std::cerr << "Failed to save binary label mask " << i << " to " << outputPath << std::endl;
        }
    }

    // --- Create or load feathering kernel, then make weighted masks ---
    int featheringRadius = 10;
    std::shared_ptr<PSF> gaussianKernel;

    if (!featheringKernelPath.empty()){
        // Load feathering kernel from image file
        auto maybeKernelImage = TiffReader::readTiffFile(featheringKernelPath, channel);
        if (!maybeKernelImage.has_value()){
            std::cerr << "Failed to read feathering kernel image: " << featheringKernelPath << std::endl;
            return 1;
        }
        Image3D kernelImage = std::move(*maybeKernelImage);
        std::cout << "Loaded feathering kernel from image, shape: " << kernelImage.getShape().print() << std::endl;
        gaussianKernel = std::make_shared<PSF>(std::move(kernelImage), "feathering_kernel");
    } else {
        // Generate a default Gaussian kernel
        gaussianKernel = executor.createGaussianKernel(featheringRadius);
        std::cout << "Generated Gaussian feathering kernel, shape: " << gaussianKernel->getShape().print() << std::endl;
    }

    // Set up CPU backend
    IBackendManager& manager = BackendFactory::getInstance().getBackendManager("cpu");
    BackendConfig config;
    config.nThreads = 1;
    IBackend& backend = manager.getBackend(config);
    std::cout << "Backend device: " << backend.getDeviceString() << std::endl;

    // Create PSF preprocessor with the standard preprocessing function
    PSFPreprocessor psfPreprocessor;
    psfPreprocessor.setPreprocessingFunction(
        [](const CuboidShape targetShape, std::shared_ptr<PSF> inputPSF, IBackend& psfBackend) -> std::unique_ptr<ComplexData> {
            Preprocessor::padToShape(*inputPSF, targetShape, PaddingFillType::ZERO);
            RealData h = Preprocessor::convertImageToRealData(*inputPSF);
            RealData h_device = psfBackend.getMemoryManager().copyDataToDevice(h);
            std::unique_ptr<ComplexView> h_result_device = std::make_unique<ComplexView>(std::move(psfBackend.getMemoryManager().reinterpret(h_device)));
            psfBackend.getDeconvManager().octantFourierShift(h_device);
            psfBackend.getDeconvManager().forwardFFT(h_device, *h_result_device);
            h_result_device->setBackend(h_device.getBackend());
            h_device.setBackend(nullptr);
            psfBackend.sync();
            return std::move(h_result_device);
        }
    );

    // Preprocess the feathering kernel (FFT in frequency domain)
    const ComplexData* preprocessedGaussianKernel = psfPreprocessor.getPreprocessedPSF(shape, gaussianKernel, backend);
    if (!preprocessedGaussianKernel){
        std::cerr << "Failed to preprocess Gaussian kernel" << std::endl;
        return 1;
    }
    std::cout << "Feathering kernel preprocessed successfully" << std::endl;

    // Create weighted masks
    std::vector<Label<RealData>> weightedLabels = executor.makeMasksWeighted(labelGroups, labelImage, *preprocessedGaussianKernel, backend);
    std::cout << "Number of weighted label groups: " << weightedLabels.size() << std::endl;

    // --- Save weighted label masks ---
    for (size_t i = 0; i < weightedLabels.size(); i++){
        const RealData* mask = weightedLabels[i].getMask();
        if (!mask){
            std::cerr << "Weighted label group " << i << " has no mask!" << std::endl;
            continue;
        }

        std::vector<std::shared_ptr<PSF>> groupPsfs = weightedLabels[i].getPSFs();
        std::string psfIds;
        for (size_t j = 0; j < groupPsfs.size(); j++){
            if (j > 0) psfIds += "_";
            psfIds += groupPsfs[j]->ID;
        }

        // Move weighted mask from device to host, then convert to Image3D for saving
        RealData maskHost = backend.getMemoryManager().moveDataFromDevice(
            *mask, BackendFactory::getInstance().getDefaultBackendMemoryManager());
        Image3D maskImage = Preprocessor::convertRealDataToImage(maskHost);

        std::string outputPath = outputDir + "/label_mask_weighted_" + std::to_string(i) + "_" + psfIds + ".tif";
        bool success = TiffWriter::writeToFile(outputPath, maskImage);
        if (success){
            std::cout << "Saved weighted label mask " << i << " (PSFs: " << psfIds << ") to " << outputPath << std::endl;
        } else {
            std::cerr << "Failed to save weighted label mask " << i << " to " << outputPath << std::endl;
        }
    }

    std::cout << "Test completed successfully." << std::endl;
    return 0;
}

