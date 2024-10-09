#include "RLDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>






Hyperstack RLDeconvolutionAlgorithm::deconvolve(Hyperstack& data, PSF& psf) {

    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};
    if(fftw_init_threads() > 0){
        std::cout << "[STATUS] FFTW init threads" << std::endl;
        fftw_plan_with_nthreads(omp_get_max_threads());
        std::cout << "[INFO] Available threads: " << omp_get_max_threads() << std::endl;
        //fftw_make_planner_thread_safe();
    }

        // Deconvolve every channel
        int channel_z = 0;
        for (auto& channel : data.channels) {
            if(preprocess(channel, psf)){
                std::cout << "[STATUS] Preprocessing channel " << channel_z + 1 << " finished" << std::endl;
            }else{
                std::cerr << "[ERROR] Preprocessing channel " << channel_z + 1 << " failed" << std::endl;
                return deconvHyperstack;
            }

            // Debug
            //std::cout << originPsfWidth << " " << originPsfHeight << " " << originPsfDepth << std::endl;
            //std::cout << safetyBorderPsfWidth << " " << safetyBorderPsfHeight << " " << safetyBorderPsfDepth << std::endl;
            //std::cout << cubeWidth << " " << cubeHeight << " " << cubeDepth << std::endl;

            std::cout << "[STATUS] Running Deconvolution..." << std::endl;
            int gridNum = 0;

            // Parallelization of grid for
            // Using static scheduling because the execution time for each iteration is similar, which reduces overhead costs by minimizing task assignment.
            #pragma omp parallel for schedule(static)
            for(auto& gridImage : this->gridImages){
                // Allocate memory for intermediate FFTW arrays
                // INFO
                // if allocations takes to much memory put outside the loop (dont forget the free lines at the end)
                fftw_complex *g = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
                fftw_complex *f = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
                fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);


                // Convert image to fftcomplex
                UtlFFT::convertCVMatVectorToFFTWComplex(gridImage, g, this->cubeWidth, this->cubeHeight, this->cubeDepth);
                std::memcpy(f, g, sizeof(fftw_complex) * this->cubeVolume);

                for (int n = 0; n < this->iterations; ++n) {
                    std::cout << "\r[STATUS] Channel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                              << gridNum + 1 << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                              << this->iterations << " ";

                    // a) First transformation:
                    // Fn = FFT(fn)
                    fftw_execute_dft(this->forwardPlan, f, f);
                    UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);

                    // Fn' = Fn * H
                    UtlFFT::complexMultiplication(f, this->paddedH, c, this->cubeVolume);

                    // fn' = IFFT(Fn')
                    UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);
                    fftw_execute_dft(this->backwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

                    // b) Calculation of the Correction Factor:
                    // c = g / fn'
                    // c = max(c, ε)
                    UtlFFT::complexDivision(g, c, c, this->cubeVolume, this->epsilon);

                    // c) Second transformation:
                    // C = FFT(c)
                    fftw_execute_dft(this->forwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

                    // C' = C * conj(H)
                    UtlFFT::complexMultiplicationWithConjugate(c, this->paddedH, c, this->cubeVolume);

                    // c' = IFFT(C')
                    UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);
                    fftw_execute_dft(backwardPlan, c, c);
                    UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

                    // d) Update the estimated image:
                    // fn = IFFT(Fn)
                    UtlFFT::octantFourierShift(f, this->cubeWidth, this->cubeHeight, this->cubeDepth);
                    fftw_execute_dft(this->backwardPlan, f, f);

                    // fn+1 = fn * c
                    UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

                    // Uncomment the following lines for debugging
                    // UtlFFT::normalizeImage(f, size, this->epsilon);
                    // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
                    // Überprüfung
                    if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                        std::cout << "[WARNING] Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
                    }
                    std::flush(std::cout);
                }
                // Convert the result FFTW complex array back to OpenCV Mat vector
                UtlFFT::convertFFTWComplexToCVMatVector(f, gridImage, this->cubeWidth, this->cubeHeight, this->cubeDepth);

                gridNum++;
                fftw_free(g);
                fftw_free(c);
                fftw_free(f);
            }

            if(postprocess(data, this->epsilon)){
                std::cout << "[STATUS] Postprocessing channel " << channel_z + 1 << " finished" << std::endl;
            }else{
                std::cerr << "[ERROR] Postprocessing channel " << channel_z + 1 << " failed" << std::endl;
                return deconvHyperstack;
            }
            // Save the result
            std::cout << "[STATUS] Saving result of channel " << channel_z + 1 << std::endl;
            Image3D deconvolutedImage;
            deconvolutedImage.slices = this->mergedVolume;
            deconvHyperstack.channels[channel.id].image = deconvolutedImage;
            channel_z++;
            this->mergedVolume.clear();
        }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return deconvHyperstack;
}

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    this->iterations = config.iterations;
    this->epsilon = config.epsilon;
    this->grid = config.grid;
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] grid: " << std::to_string(this->grid) << std::endl;
    if(this->grid){
        std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
        std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
        std::cout << "[CONFIGURATION] cubeSize: " << this->cubeSize << std::endl;
    }
}