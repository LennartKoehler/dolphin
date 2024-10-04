#include "InverseFilterDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include <cstring>
#include "UtlGrid.h"
#include "UtlFFT.h"
#include "UtlImage.h"

// Perform deconvolution on the given data using the PSF
Hyperstack InverseFilterDeconvolutionAlgorithm::deconvolve(Hyperstack& data, PSF& psf) {
    std::cout << "Starting deconvolution..." << std::endl;

    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};

    // Deconvolve every channel
    int channel_z = 0;
    for (auto& channel : data.channels) {
        std::cout << "Processing channel " << channel_z + 1 << "..." << std::endl;
        // Find and display global min and max of the data
        double globalMin, globalMax;
        UtlImage::findGlobalMinMax(data.channels[0].image.slices, globalMin, globalMax);
        std::cout << "Image: Min/Max: " << globalMin << "/" << globalMax << std::endl;

        // Adjust PSF pixel value range for inverse filter
        std::cout << "Adjusting PSF pixel value range..." << std::endl;
        for(auto& layer : psf.image.slices) {
            layer.convertTo(layer, CV_32F, 0.000001 / globalMax);
        }
        UtlImage::findGlobalMinMax(psf.image.slices, globalMin, globalMax);
        std::cout << "PSF: Min/Max: " << globalMin << "/" << globalMax << std::endl;

        std::vector<std::vector<cv::Mat>> split;
        int originImageWidth = data.metaData.imageWidth;
        int originImageHeight = data.metaData.imageLength;
        int originImageDepth = data.metaData.slices;
        int originImageVolume = originImageWidth * originImageHeight * originImageDepth;
        int originPsfWidth = psf.image.slices[0].cols;
        int originPsfHeight = psf.image.slices[0].rows;
        int originPsfDepth = psf.image.slices.size();
        int originPsfVolume = originPsfWidth * originPsfHeight * originPsfDepth;

        int safetyBorderPsfWidth = psf.image.slices[0].cols+(2*this->psfSafetyBorder);
        int safetyBorderPsfHeight = psf.image.slices[0].rows+(2*this->psfSafetyBorder);
        int safetyBorderPsfDepth = psf.image.slices.size()+(2*this->psfSafetyBorder);
        int safetyBorderPsfVolume = safetyBorderPsfWidth * safetyBorderPsfHeight * safetyBorderPsfDepth;
        int imagePadding = originImageWidth / 2;
        int cubePadding = this->psfSafetyBorder;
        if(safetyBorderPsfWidth < this->cubeSize){
            cubePadding = 10;
        }
        if(this->cubeSize+2*cubePadding < safetyBorderPsfWidth){
            cubePadding = (safetyBorderPsfWidth-this->cubeSize)/2;
        }

        int cubeWidth;
        int cubeHeigth;
        int cubeDepth;
        int cubeVolume;

        if(!this->grid){
            split.push_back(channel.image.slices);
            cubeWidth = data.metaData.imageWidth;
            cubeHeigth = data.metaData.imageLength;
            cubeDepth = data.metaData.slices;
            cubeVolume = cubeWidth * cubeHeigth * cubeDepth;

            safetyBorderPsfWidth = cubeWidth;
            safetyBorderPsfHeight = cubeHeigth;
            safetyBorderPsfDepth = cubeDepth;
            safetyBorderPsfVolume = cubeVolume;

        }else {
            if(this->cubeSize < 1 || this->psfSafetyBorder < 0){
                std::cerr << "CubeSize should be greater than 1 and PsfSafetyBorder should be greater than 0" << std::endl;
                return deconvHyperstack;
            }
            UtlGrid::extendImage(channel.image.slices, imagePadding, this->borderType);
            split = UtlGrid::splitWithCubePadding(channel.image.slices, this->cubeSize, imagePadding, cubePadding);
            std::cout << "GridImageProps.: [Depth: " << split[0].size() << " Width:" << split[0][0].cols << " Height:" << split[0][0].rows << " Subimages: " << split.size() << "]" << std::endl;

            if((this->cubeSize + 2*cubePadding) != split[0][0].cols){
                std::cerr << "CubeSize doesnt match with actual CubeSize: " << split[0][0].cols << " (should be: " << (this->cubeSize + 2*cubePadding) << ")" << std::endl;
            }

            cubeWidth = (this->cubeSize + 2*cubePadding);
            cubeHeigth = (this->cubeSize + 2*cubePadding);
            cubeDepth = (this->cubeSize + 2*cubePadding);
            cubeVolume = cubeWidth * cubeHeigth * cubeDepth;
            if(cubeWidth != this->psfSafetyBorder){
                if(cubeWidth > this->psfSafetyBorder){
                    safetyBorderPsfWidth = cubeWidth;
                    safetyBorderPsfHeight = cubeHeigth;
                    safetyBorderPsfDepth = cubeDepth;
                    safetyBorderPsfVolume = cubeVolume;
                }
            }
        }
        if(this->cubeSize < originPsfWidth){
            std::cout << "[WARNING] PSF is larger than image/cube" << std::endl;
        }

        // Fourier Transformation of PSF
        std::cout << "Performing Fourier Transform on PSF..." << std::endl;
        fftw_complex *psfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
        UtlFFT::convertCVMatVectorToFFTWComplex(psf.image.slices, psfFFT, originPsfWidth, originPsfHeight, originPsfDepth);
        UtlFFT::forwardFFT(psfFFT, psfFFT, originPsfDepth, originPsfHeight, originPsfWidth);


        std::cout << "Padding PSF..." << std::endl;
        // Pad the PSF to the size of the image
        fftw_complex *padded_psfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
        UtlFFT::padPSF(psfFFT, originPsfWidth, originPsfHeight, originPsfDepth, padded_psfFFT, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
         // Free FFTW resources for PSF
        fftw_free(psfFFT);

        // Allocate memory for intermediate FFTW arrays
        fftw_complex *image = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);

        std::cout << "Running Deconvolution..." << std::endl;
        int gridNum = 0;
        for(auto& gridImage : split){
            std::flush(std::cout);

            std::cout << "\rChannel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                      << gridNum + 1 << "/" << split.size() << " ";
            //Convert image to fftcomplex
            UtlFFT::convertCVMatVectorToFFTWComplex(gridImage, image, cubeWidth, cubeHeigth, cubeDepth);

            // Forward FFT on image
            //std::cout << "Performing forward FFT on image..." << std::endl;
            UtlFFT::forwardFFT(image, image, cubeDepth, cubeHeigth, cubeWidth);

            //std::cout << "Performing complex division in frequency domain..." << std::endl;
            UtlFFT::complexDivision(image, padded_psfFFT, image, cubeVolume, this->epsilon);

            // Inverse FFT
            //std::cout << "Performing inverse FFT..." << std::endl;
            UtlFFT::backwardFFT(image, image, cubeDepth, cubeHeigth, cubeWidth);
            UtlFFT::octantFourierShift(image, cubeWidth, cubeHeigth, cubeDepth);
            //UtlFFT::reorderLayers(image, cubeWidth, cubeHeigth, cubeDepth);


            // Convert the result FFTW complex array back to OpenCV Mat vector
            UtlFFT::convertFFTWComplexToCVMatVector(image, gridImage, cubeWidth, cubeHeigth, cubeDepth);


            gridNum++;
            std::flush(std::cout);

        }

        std::vector<cv::Mat> mergedVolume;
        if(this->grid){
            UtlGrid::cropCubePadding(split, cubePadding);
            cubeWidth = this->cubeSize;
            cubeHeigth = this->cubeSize;
            cubeDepth = this->cubeSize;
            cubeVolume = cubeWidth * cubeHeigth * cubeDepth;

            std::cout << "Merging Grid back to Image..." << std::endl;
            mergedVolume = UtlGrid::mergeCubes(split, originImageWidth, originImageHeight, originImageDepth, this->cubeSize);
            std::cout << mergedVolume.size() << " " << mergedVolume[0].cols << " " << mergedVolume[0].rows << std::endl;
        }else{
            mergedVolume = split[0];
        }

        // Global normalization of the merged volume
        double global_max_val= 0.0;
        double global_min_val = MAXFLOAT;
        for (const auto& slice : mergedVolume) {
            double min_val, max_val;
            cv::minMaxLoc(slice, &min_val, &max_val);
            global_max_val = std::max(global_max_val, max_val);
            global_min_val = std::min(global_min_val, min_val);
        }

        for (auto& slice : mergedVolume) {
            slice.convertTo(slice, CV_32F, 1.0 / (global_max_val-global_min_val), -global_min_val*(1/(global_max_val-global_min_val)));  // Add epsilon to avoid division by zero
        }


        // Save the result
        Image3D deconvolutedImage;
        deconvolutedImage.slices = mergedVolume;
        deconvHyperstack.channels[channel.id].image = deconvolutedImage;
        channel_z++;

        // Free FFTW resources for the current channel
        fftw_free(padded_psfFFT);
        fftw_free(image);
    }

    std::cout << "Deconvolution complete." << std::endl;
    return deconvHyperstack;
}

// Configure the deconvolution parameters
void InverseFilterDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    this->epsilon = config.epsilon;
    this->grid = config.grid;
    std::cout << "Configured InverseFilter algorithm with epsilon: " << epsilon << std::endl;
    std::cout << "Configured InverseFilter algorithm with grid: " << this->grid << std::endl;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    if(this->grid){
        std::cout << "Configured InverseFilter algorithm with borderType: " << this->borderType << std::endl;
        std::cout << "Configured InverseFilter algorithm with psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "Configured InverseFilter algorithm with cubeSize: " << this->cubeSize << std::endl;
    }
}
