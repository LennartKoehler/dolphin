#include "dolphin/Image3D.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include <iostream>
#include <string>

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <input.tif> [output.tif] [channel]" << std::endl;
        return 1;
    }

    std::string input = argv[1];
    std::string output;
    if(argc >= 3){
        output = argv[2];
    } else {
        // default output file name
        output = input + "_copy.tif";
    }

    int channel = 0;
    if(argc >= 4){
        try{
            channel = std::stoi(argv[3]);
        } catch(...){
            std::cerr << "Invalid channel, using 0" << std::endl;
            channel = 0;
        }
    }

    try{
        auto maybeImage = TiffReader::readTiffFile(input, channel);
        if(!maybeImage.has_value()){
            std::cerr << "Failed to read input TIFF file: " << input << std::endl;
            return 4;
        }
        Image3D image = std::move(*maybeImage);

        bool ok = TiffWriter::writeToFile(output, image);
        if(!ok){
            std::cerr << "Failed to write output file: " << output << std::endl;
            return 2;
        }
    } catch(const std::exception& e){
        std::cerr << "Exception: " << e.what() << std::endl;
        return 3;
    }

    std::cout << "Successfully wrote: " << output << std::endl;
    return 0;
}
