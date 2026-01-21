#include "Image3D.h"  // Use explicit path to local header
#include "deconvolution/Postprocessor.h"
#include "IO/TiffReader.h"
#include "IO/TiffWriter.h"

int main(){

    std::string filename1 = "/home/lennart-k-hler/data/labeledImage/simpleMaskSmall.tif";
    int channel = 0;
    Image3D image1 = TiffReader::readTiffFile(filename1, channel);
    std::string filename2 = "/home/lennart-k-hler/data/labeledImage/simpleMaskSmall_darker.tif";
    Image3D image2 = TiffReader::readTiffFile(filename2, channel);
    PaddedImage pImage1{image1};

    PaddedImage pImage2{image2};

    int radius = 5;
    double epsilon = 5.0;

    // Create masks using ITK-based Image3D methods
    Image3D mask1 = image1.getInRange(255.0f, 255.0f);  // Create mask where pixel value equals 255
    Image3D mask2 = image1.getInRange(0.0f, 0.0f);      // Create mask where pixel value equals 0
    
    std::vector<ImageMaskPair> pairs{ImageMaskPair{image1, mask1}, ImageMaskPair{image2, mask2}};

    Image3D out = Postprocessor::addFeathering(
        pairs,
        radius,
        epsilon
    );

    TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/test.tif", out);

}