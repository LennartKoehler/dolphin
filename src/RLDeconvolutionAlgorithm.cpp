#include "RLDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>
/*
   1. Vorwärtsprojektion:
        U = FFT(u)
        U' = U * H
        u' = IFFT(U')

     2. Berechnung des Korrekturfaktors:
        C = g / u'
        C = max(C, ε)

     3. Rückwärtsprojektion:
        C' = FFT(C)
        C'' = C' * conj(H)
        c = IFFT(C'')

     4. Update der Schätzung:
        u = u * c
   */



Hyperstack RLDeconvolutionAlgorithm::deconvolve(Hyperstack& data, PSF& psf) {

    std::cout << "Starting deconvolution..." << std::endl;

    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};
   /* if (init_fftw_threads() != 0) {
        return deconvHyperstack;
    }*/


        // Deconvolve every channel
        int channel_z = 0;
        for (auto& channel : data.channels) {
            std::cout << "Processing channel " << channel_z + 1 << "..." << std::endl;
            // Find and display global min and max of the data
            double globalMin, globalMax;
            UtlImage::findGlobalMinMax(data.channels[0].image.slices, globalMin, globalMax);
            std::cout << "Image: Min/Max: " << globalMin << "/" << globalMax << std::endl;
            double globalMinPsf, globalMaxPsf;
            UtlImage::findGlobalMinMax(psf.image.slices, globalMinPsf, globalMaxPsf);
            std::cout << "PSF: Min/Max: " << globalMinPsf << "/" << globalMaxPsf << std::endl;

            std::vector<std::vector<cv::Mat>> split;
            int originImageWidth = data.metaData.imageWidth;
            int originImageHeight = data.metaData.imageLength;
            int originImageDepth = data.metaData.slices;
            int originImageVolume = originImageWidth * originImageHeight * originImageDepth;
            int originPsfWidth = psf.image.slices[0].cols;
            int originPsfHeight = psf.image.slices[0].rows;
            int originPsfDepth = psf.image.slices.size();
            int originPsfVolume = originPsfWidth * originPsfHeight * originPsfDepth;

            //int psfSafetyBorder = 20;//originPsfWidth/2;
            int safetyBorderPsfWidth = psf.image.slices[0].cols+(2*this->psfSafetyBorder);
            int safetyBorderPsfHeight = psf.image.slices[0].rows+(2*this->psfSafetyBorder);
            int safetyBorderPsfDepth = psf.image.slices.size()+(2*this->psfSafetyBorder);
            int safetyBorderPsfVolume = safetyBorderPsfWidth * safetyBorderPsfHeight * safetyBorderPsfDepth;
            int imagePadding = originImageWidth / 2;
            //int cubeSize = 20;
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

            // Flip the PSF
            PSF flippedPSF = psf.flip3DPSF();
            // Fourier Transformation of flipped PSF
            std::cout << "Flipping PSF..." << std::endl;
            fftw_complex *flippedPsfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
            UtlFFT::convertCVMatVectorToFFTWComplex(flippedPSF.image.slices, flippedPsfFFT, originPsfWidth, originPsfHeight,originPsfDepth);
            std::cout << "Performing Fourier Transform on flipped PSF..." << std::endl;
            UtlFFT::forwardFFT(flippedPsfFFT, flippedPsfFFT, originPsfDepth, originPsfHeight, originPsfWidth);

            std::cout << "Padding PSF and FlippedPSF..." << std::endl;
            // Pad the PSF to the size of the image
            fftw_complex *padded_psfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
            UtlFFT::padPSF(psfFFT, originPsfWidth, originPsfHeight, originPsfDepth, padded_psfFFT, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
            //UtlFFT::visualizeFFT(padded_psfFFT, cubeWidth,cubeHeigth, cubeDepth);

            // Debug
            //std::cout << originPsfWidth << " " << originPsfHeight << " " << originPsfDepth << std::endl;
            //std::cout << safetyBorderPsfWidth << " " << safetyBorderPsfHeight << " " << safetyBorderPsfDepth << std::endl;
            //std::cout << cubeWidth << " " << cubeHeigth << " " << cubeDepth << std::endl;

            // Pad the flipped PSF to the size of the image
            fftw_complex *padded_flippedPsfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
            UtlFFT::padPSF(flippedPsfFFT, originPsfWidth, originPsfHeight, originPsfDepth, padded_flippedPsfFFT, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
            // Free FFTW resources for PSF
            fftw_free(psfFFT);
            fftw_free(flippedPsfFFT);

            // Allocate memory for intermediate FFTW arrays
            fftw_complex *originImage = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
            fftw_complex *resultImage = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
            fftw_complex *interimImage = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
            fftw_complex *convolutedRatio = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);

            std::cout << "Running Deconvolution..." << std::endl;
            int gridNum = 0;
            //fftw_make_planner_thread_safe();
            //#pragma omp parallel for
            for(auto& gridImage : split){

                //Convert image to fftcomplex
                UtlFFT::convertCVMatVectorToFFTWComplex(gridImage, originImage, cubeWidth, cubeHeigth, cubeDepth);
                std::memcpy(resultImage, originImage, sizeof(fftw_complex) * cubeVolume);

                for (int i = 0; i < this->iterations; i++) {
                    std::flush(std::cout);

                        std::cout << "\rChannel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                                  << gridNum + 1 << "/" << split.size() << " Iteration: " << i + 1 << "/"
                                  << this->iterations << " ";

                    // Perform forward FFT
                    UtlFFT::forwardFFT(resultImage, resultImage, cubeDepth, cubeHeigth, cubeWidth);

                    // Convolve the result image with the PSF in frequency domain
                    UtlFFT::complexMultiplication(resultImage, padded_psfFFT, convolutedRatio, cubeVolume);

                    UtlFFT::backwardFFT(convolutedRatio, convolutedRatio, cubeDepth, cubeHeigth, cubeWidth);
                    UtlFFT::quadrantShift(convolutedRatio, cubeWidth, cubeHeigth, cubeDepth);
                    // Compute the ratio of the input image to the convoluted image
                    UtlFFT::complexDivision(originImage, convolutedRatio, convolutedRatio, cubeVolume, this->epsilon);

                    // Perform forward FFT on the ratio
                    UtlFFT::forwardFFT(convolutedRatio, convolutedRatio, cubeDepth, cubeHeigth, cubeWidth);

                    //Correlate the ratio with the flipped PSF in frequency domain
                    UtlFFT::complexMultiplication(convolutedRatio, padded_flippedPsfFFT, interimImage, cubeVolume);

                    // Perform inverse FFT on the interim image and result image
                    UtlFFT::backwardFFT(resultImage, resultImage, cubeDepth, cubeHeigth, cubeWidth);
                    UtlFFT::backwardFFT(interimImage, interimImage, cubeDepth, cubeHeigth, cubeWidth);
                    UtlFFT::quadrantShift(interimImage, cubeWidth, cubeHeigth, cubeDepth);

                    // Multiple temp result image with interim image to get result
                    UtlFFT::complexMultiplication(resultImage, interimImage, resultImage, cubeVolume);

                    // Normalize to avoid very small values
                    //UtlFFT::normalizeImage(resultImage, size, this->epsilon);

                    // Save intermediate results for debugging
                    //UtlFFT::saveInterimImages(resultImage, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);

                    std::flush(std::cout);
                }
                // Convert the result FFTW complex array back to OpenCV Mat vector
                UtlFFT::convertFFTWComplexToCVMatVector(resultImage, gridImage, cubeWidth, cubeHeigth, cubeDepth);


                gridNum++;
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
            fftw_free(padded_flippedPsfFFT);
            fftw_free(interimImage);
            fftw_free(convolutedRatio);
            fftw_free(resultImage);

        }

    std::cout << "Deconvolution complete." << std::endl;
    return deconvHyperstack;
}

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    this->iterations = config.iterations;
    this->epsilon = config.epsilon;
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    std::cout << "Configured RL algorithm with iterations: " << this->iterations << std::endl;
    std::cout << "Configured RL algorithm with epsilon: " << this->epsilon << std::endl;
    std::cout << "Configured RL algorithm with grid: " << std::to_string(this->grid) << std::endl;
    if(this->grid){
        std::cout << "Configured RL algorithm with borderType: " << this->borderType << std::endl;
        std::cout << "Configured RL algorithm with psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "Configured RL algorithm with cubeSize: " << this->cubeSize << std::endl;
    }
}