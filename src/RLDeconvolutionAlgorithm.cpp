#include "RLDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>


bool isValidForFloat(fftw_complex* fftwData, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // Überprüfen der Real- und Imaginärteile
        if (fftwData[i][0] < std::numeric_limits<float>::lowest() ||
            fftwData[i][0] > std::numeric_limits<float>::max() ||
            fftwData[i][1] < std::numeric_limits<float>::lowest() ||
            fftwData[i][1] > std::numeric_limits<float>::max()) {
            return false; // Ein Wert ist außerhalb des gültigen Bereichs
        }
    }
    return true; // Alle Werte sind gültig
}

void normalizePSF(std::vector<cv::Mat>& psf) {
    // Berechne die Summe aller Werte in der PSF
    double totalSum = 0.0;
    for (const auto& slice : psf) {
        totalSum += cv::sum(slice)[0]; // Summe über alle Pixel in jedem Slice
    }

    // Normiere jeden Slice, indem du ihn durch die Gesamtsumme teilst
    for (auto& slice : psf) {
        slice /= totalSum;
    }
}


Hyperstack RLDeconvolutionAlgorithm::deconvolve(Hyperstack& data, PSF& psf) {

    std::cout << "Starting deconvolution..." << std::endl;
    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};
    if(fftw_init_threads() > 0){
        std::cout << "FFTW init threads" << std::endl;
        fftw_plan_with_nthreads(omp_get_max_threads());
        std::cout << "Available threads: " << omp_get_max_threads() << std::endl;
        //fftw_make_planner_thread_safe();
    }

        // Deconvolve every channel
        int channel_z = 0;
        for (auto& channel : data.channels) {
            std::cout << "Processing channel " << channel_z + 1 << "..." << std::endl;
            // Find and display global min and max of the data
            double globalMin, globalMax;
            UtlImage::findGlobalMinMax(data.channels[0].image.slices, globalMin, globalMax);
            std::cout << "Image: Min/Max: " << globalMin << "/" << globalMax << std::endl;
            double globalMinPsf, globalMaxPsf;
            //TODO
            normalizePSF(psf.image.slices);
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
            int cubePadding = this->psfSafetyBorder;
            if(this->cubeSize < 1){
                // Auto function for cubeSize, sets cubeSize to fit PSF
                std::cout << "[INFO] CubeSize fitted to PSF size" << std::endl;
                this->cubeSize = std::min({originPsfWidth, originPsfHeight, originPsfDepth});
            }
            if(safetyBorderPsfWidth < this->cubeSize){
                cubePadding = 10;
                std::cout << "[INFO] PSF with safety border smaller than cubeSize" << std::endl;
            }
            if(this->cubeSize+2*cubePadding < safetyBorderPsfWidth){
                cubePadding = (safetyBorderPsfWidth-this->cubeSize)/2;
                //std::cout <<  "[INFO] cubeSize smaller than PSF with safety border" << std::endl;
            }

            int cubeWidth;
            int cubeHeight;
            int cubeDepth;
            int cubeVolume;

            if(!this->grid){
                split.push_back(channel.image.slices);
                cubeWidth = data.metaData.imageWidth;
                cubeHeight = data.metaData.imageLength;
                cubeDepth = data.metaData.slices;
                cubeVolume = cubeWidth * cubeHeight * cubeDepth;

                safetyBorderPsfWidth = cubeWidth;
                safetyBorderPsfHeight = cubeHeight;
                safetyBorderPsfDepth = cubeDepth;
                safetyBorderPsfVolume = cubeVolume;

            }else {

                if(this->psfSafetyBorder < 1){
                    std::cout << "CubeSize should be greater than 1 and PsfSafetyBorder should be greater than 0" << std::endl;
                    return deconvHyperstack;
                }

                UtlGrid::extendImage(channel.image.slices, imagePadding, this->borderType);

                split = UtlGrid::splitWithCubePadding(channel.image.slices, this->cubeSize, imagePadding, cubePadding);
                std::cout << "GridImageProps.: [Depth: " << split[0].size() << " Width:" << split[0][0].cols << " Height:" << split[0][0].rows << " Subimages: " << split.size() << "]" << std::endl;

                if((this->cubeSize + 2*cubePadding) != split[0][0].cols){
                    std::cerr << "CubeSize doesnt match with actual CubeSize: " << split[0][0].cols << " (should be: " << (this->cubeSize + 2*cubePadding) << ")" << std::endl;
                }

                cubeWidth = (this->cubeSize + 2*cubePadding);
                cubeHeight = (this->cubeSize + 2 * cubePadding);
                cubeDepth = (this->cubeSize + 2*cubePadding);
                cubeVolume = cubeWidth * cubeHeight * cubeDepth;

                if(cubeWidth != this->psfSafetyBorder){
                    if(cubeWidth > this->psfSafetyBorder){
                        safetyBorderPsfWidth = cubeWidth;
                        safetyBorderPsfHeight = cubeHeight;
                        safetyBorderPsfDepth = cubeDepth;
                        safetyBorderPsfVolume = cubeVolume;
                    }
                }

            }
            if(this->cubeSize < originPsfWidth){
                std::cout << "[WARNING] PSF is larger than image/cube" << std::endl;
            }

            // In-line fftplan for fast ft calculation and inverse
            fftw_complex *fftwPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
            fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
            fftw_plan forwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_FORWARD, FFTW_MEASURE);
            fftw_plan forwardPSFPlan = fftw_plan_dft_3d(originPsfDepth, originPsfHeight, originPsfWidth, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);
            fftw_plan backwardPlan = fftw_plan_dft_3d(cubeDepth, cubeHeight, cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_BACKWARD, FFTW_MEASURE);

            // Fourier Transformation of PSF
            std::cout << "Performing Fourier Transform on PSF..." << std::endl;
            fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
            UtlFFT::convertCVMatVectorToFFTWComplex(psf.image.slices, h, originPsfWidth, originPsfHeight, originPsfDepth);
            fftw_execute_dft(forwardPSFPlan, h, h);
            UtlFFT::octantFourierShift(h, originPsfWidth, originPsfHeight, originPsfDepth);

            std::cout << "Padding PSF..." << std::endl;
            // Pad the PSF to the size of the image
            fftw_complex *paddedH = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
            UtlFFT::padPSF(h, originPsfWidth, originPsfHeight, originPsfDepth, paddedH, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
            // Free FFTW resources for PSF
            fftw_free(h);
            fftw_free(fftwPSFPlanMem);
            fftw_destroy_plan(forwardPSFPlan);

            // Debug
            //std::cout << originPsfWidth << " " << originPsfHeight << " " << originPsfDepth << std::endl;
            //std::cout << safetyBorderPsfWidth << " " << safetyBorderPsfHeight << " " << safetyBorderPsfDepth << std::endl;
            //std::cout << cubeWidth << " " << cubeHeight << " " << cubeDepth << std::endl;

            std::cout << "Running Deconvolution..." << std::endl;
            int gridNum = 0;

            // Parallelization of grid for
            // Using static scheduling because the execution time for each iteration is similar, which reduces overhead costs by minimizing task assignment.
            #pragma omp parallel for schedule(static)
            for(auto& gridImage : split){
                // Allocate memory for intermediate FFTW arrays
                // INFO
                // if allocations takes to much memory put outside the loop (dont forget the free lines at the end)
                fftw_complex *g = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
                fftw_complex *f = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);
                fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * cubeVolume);


                // Convert image to fftcomplex
                UtlFFT::convertCVMatVectorToFFTWComplex(gridImage, g, cubeWidth, cubeHeight, cubeDepth);
                std::memcpy(f, g, sizeof(fftw_complex) * cubeVolume);

                for (int n = 0; n < this->iterations; ++n) {
                    std::cout << "\rChannel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                              << gridNum + 1 << "/" << split.size() << " Iteration: " << n + 1 << "/"
                              << this->iterations << " ";

                    // a) First transformation:
                    // Fn = FFT(fn)
                    fftw_execute_dft(forwardPlan, f, f);
                    UtlFFT::octantFourierShift(f, cubeWidth, cubeHeight, cubeDepth);


                    // Fn' = Fn * H
                    UtlFFT::complexMultiplication(f, paddedH, c, cubeVolume);

                    // fn' = IFFT(Fn')
                    UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);
                    fftw_execute_dft(backwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

                    // b) Calculation of the Correction Factor:
                    // c = g / fn'
                    // c = max(c, ε)
                    UtlFFT::complexDivision(g, c, c, cubeVolume, this->epsilon);

                    // c) Second transformation:
                    // C = FFT(c)
                    fftw_execute_dft(forwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

                    // C' = C * conj(H)
                    UtlFFT::complexMultiplicationWithConjugate(c, paddedH, c, cubeVolume);

                    // c' = IFFT(C')
                    UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);
                    fftw_execute_dft(backwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, cubeWidth, cubeHeight, cubeDepth);

                    // d) Update the estimated image:
                    // fn = IFFT(Fn)
                    UtlFFT::octantFourierShift(f, cubeWidth, cubeHeight, cubeDepth);
                    fftw_execute_dft(backwardPlan, f, f);

                    // fn+1 = fn * c
                    UtlFFT::complexMultiplication(f, c, f, cubeVolume);

                    // Uncomment the following lines for debugging
                    // UtlFFT::normalizeImage(f, size, this->epsilon);
                    // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
                    // Überprüfung
                    if (!(isValidForFloat(f, cubeVolume))) {
                        std::cout << "Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
                    }
                    std::flush(std::cout);
                }
                // Convert the result FFTW complex array back to OpenCV Mat vector
                UtlFFT::convertFFTWComplexToCVMatVector(f, gridImage, cubeWidth, cubeHeight, cubeDepth);


                gridNum++;
                fftw_free(g);
                fftw_free(c);
                fftw_free(f);
            }

            // Free FFTW resources for the current channel
            fftw_free(paddedH);
            fftw_destroy_plan(forwardPlan);
            fftw_destroy_plan(backwardPlan);
            fftw_free(fftwPlanMem);

            std::vector<cv::Mat> mergedVolume;
            if(this->grid){
                UtlGrid::cropCubePadding(split, cubePadding);
                cubeWidth = this->cubeSize;
                cubeHeight = this->cubeSize;
                cubeDepth = this->cubeSize;
                cubeVolume = cubeWidth * cubeHeight * cubeDepth;

                std::cout << "Merging Grid back to Image..." << std::endl;
                mergedVolume = UtlGrid::mergeCubes(split, originImageWidth, originImageHeight, originImageDepth, this->cubeSize);
                std::cout << mergedVolume.size() << " " << mergedVolume[0].cols << " " << mergedVolume[0].rows << std::endl;
            }else{
                mergedVolume = split[0];
            }

            // Global normalization of the merged volume
            double global_max_val= 0.0;
            double global_min_val = MAXFLOAT;
            int j = 0;
            for (const auto& slice : mergedVolume) {
                double min_val, max_val;
                cv::minMaxLoc(slice, &min_val, &max_val);
                global_max_val = std::max(global_max_val, max_val);
                global_min_val = std::min(global_min_val, min_val);

            }

            for (auto& slice : mergedVolume) {
                slice.convertTo(slice, CV_32F, 1.0 / (global_max_val-global_min_val), -global_min_val*(1/(global_max_val-global_min_val)));  // Add epsilon to avoid division by zero
                //TODO
                //cv::normalize(slice, slice, 0, 1, cv::NORM_MINMAX);
            }


            // Save the result
            Image3D deconvolutedImage;
            deconvolutedImage.slices = mergedVolume;
            deconvHyperstack.channels[channel.id].image = deconvolutedImage;
            channel_z++;
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