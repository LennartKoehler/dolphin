#include "HyperstackImage.h"
#include "UtlFFT.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fftw3.h>


Hyperstack::Hyperstack() = default;
Hyperstack::Hyperstack(const Hyperstack &rhs) : channels{rhs.channels}, metaData{rhs.metaData} {}
Hyperstack::~Hyperstack() = default;

bool Hyperstack::showSlice(int channel, int z) {
    if (!this->channels[channel].image.slices.empty()) {
        if(z < 0 || z > this->metaData.totalImages){
            std::cerr << "Slice " << std::to_string(z) << " out of Range in Channel " << std::to_string(channel) << std::endl;
            return false;
        }else {
            cv::Mat& slice = channels[channel].image.slices[z];

            if (slice.type() != CV_32F) {
                std::cerr << "Expected CV_32F data type." << std::endl;
                return false;
            }
            if (slice.empty()) {
                std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
                return false;
            }

            // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
            cv::Mat img8bit;
            double minVal, maxVal;
            cv::minMaxLoc(slice, &minVal, &maxVal);

            // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
            // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
            slice.convertTo(img8bit, CV_8U, 255.0 / maxVal);
            cv::imshow("Slice " + std::to_string(z) + " in Channel " + std::to_string(channel), img8bit);
            cv::waitKey();

            std::cout << "Slice " + std::to_string(z) + " in Channel " + std::to_string(channel) + " shown" << std::endl;
            return true;
        }
    }else{
        std::cerr<< "Layer " <<  std::to_string(z) << " cannot shown" << std::endl;
        return false;
    }
}
bool Hyperstack::showChannel(int channel){
    for (const auto &mat : this->channels[channel].image.slices) {
            if (mat.type() != CV_32F) {
                std::cerr << "Expected CV_32F data type." << std::endl;
                continue;
            }
            if (mat.empty()) {
                std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
                continue;            }

            // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
        cv::Mat img8bit;
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);

        // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
        // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
        mat.convertTo(img8bit, CV_8U, 255.0 / maxVal);
        cv::imshow("Channel " + std::to_string(channel), img8bit);
        cv::waitKey();
    }

    return true;
}

float Hyperstack::getPixel(int channel, int x, int y, int z) {
    return this->channels[channel].image.slices[z].at<float>(x,y);
}

Hyperstack Hyperstack::convolve(PSF psf) {

    Hyperstack convolved;
    convolved.channels = this->channels;
    int psfWidth = psf.image.slices[0].cols;
    int psfHeight = psf.image.slices[0].rows;
    int psfDepth = psf.image.slices.size();
    int psfVolume = psfWidth * psfHeight * psfDepth;
    fftw_complex *psfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * psfVolume);
    UtlFFT::convertCVMatVectorToFFTWComplex(psf.image.slices, psfFFT, psfWidth, psfHeight, psfDepth);
    UtlFFT::forwardFFT(psfFFT, psfFFT, psfDepth, psfHeight, psfWidth);

    for(auto& channel : convolved.channels){
        int depth = channel.image.slices.size();
        int width = channel.image.slices[0].cols;
        int heigth = channel.image.slices[0].rows;
        int volume =  depth * width * heigth;
        fftw_complex *imageFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * volume);
        UtlFFT::convertCVMatVectorToFFTWComplex(channel.image.slices, imageFFT, width, heigth, depth);
        UtlFFT::forwardFFT(imageFFT, imageFFT, depth, heigth, width);

        fftw_complex *padded_psfFFT = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * volume);
        UtlFFT::padPSF(psfFFT, psfWidth, psfHeight, psfDepth, padded_psfFFT, width, heigth, depth);


        // Convolve the result image with the PSF in frequency domain
        UtlFFT::complexMultiplication(imageFFT, padded_psfFFT, imageFFT, volume);

        UtlFFT::backwardFFT(imageFFT, imageFFT, depth, heigth, width);
        UtlFFT::octantFourierShift(imageFFT, width, heigth, depth);

        UtlFFT::convertFFTWComplexToCVMatVector(imageFFT, channel.image.slices, width, heigth, depth);
    }

    return convolved;
}




