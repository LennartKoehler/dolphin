#include "deconvolution/Postprocessor.h"
#include "IO/TiffReader.h"
#include "IO/TiffWriter.h"

int main(){

    std::string filename1 = "/home/lennart-k-hler/data/labeledImage/simpleMaskSmall.tif";
    Image3D image1 = TiffReader::readTiffFile(filename1);
    std::string filename2 = "/home/lennart-k-hler/data/labeledImage/simpleMaskSmall_darker.tif";
    Image3D image2 = TiffReader::readTiffFile(filename2);
    PaddedImage pImage1{image1};

    PaddedImage pImage2{image2};


    Image3D mask1;
    Image3D mask2;

    int radius = 5;
    double epsilon = 5.0;

    for (int z = 0; z < image1.slices.size(); z++){
        mask1.slices.push_back(image1.slices[z] == 255);
        mask2.slices.push_back(image1.slices[z] == 0);
    }
    std::vector<ImageMaskPair> pairs{ImageMaskPair{image1, mask1}, ImageMaskPair{image2, mask2}};


    Image3D out = Postprocessor::addFeathering(
        pairs,
        radius,
        epsilon
    );

    TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/test.tif", out, TiffReader::extractMetadata(filename1));


}