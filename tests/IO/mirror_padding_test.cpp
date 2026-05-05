#include "dolphin/Image3D.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include <iostream>
#include "itkTestingComparisonImageFilter.h"
#include <string>




bool isEqual(ImageType::Pointer im1, ImageType::Pointer im2){
    using CompareType =
        itk::Testing::ComparisonImageFilter<ImageType, ImageType>;

    auto compare = CompareType::New();
    compare->SetValidInput(im1);
    compare->SetTestInput(im2);

    // tolerance for floating point differences
    compare->SetDifferenceThreshold(1e-6);
    compare->SetVerifyInputInformation(false);
    compare->Update();
    return compare->GetNumberOfPixelsWithDifferences() == 0;
}


int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <input.tif> [output.tif]" << std::endl;
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
    TiffReader readerPadding = TiffReader(input, channel);

    TiffReader reader = TiffReader(input, channel);
    BoxCoordWithPadding box_withPadding(
        BoxCoord{
            CuboidShape(120, 120, 20),
            CuboidShape(50,50,50),
        },
        Padding{
            CuboidShape(50, 50, 20),
            CuboidShape(50, 50, 20)
        }
    );
    BoxCoordWithPadding box_withoutPadding(
        BoxCoord{
            CuboidShape(70, 70, 0),
            CuboidShape(150,150,90)
        },
        Padding{
            CuboidShape(0, 0, 0),
            CuboidShape(0, 0, 0)
        }
    );
    try{
        auto maybeImage1 = readerPadding.getSubimage(box_withPadding);
        auto maybeImageTest = readerPadding.getSubimage(box_withoutPadding);
        // shouldnt actually read new slice as its the same image data as above and uses same reader which should have it in buffer
        std::cerr << "Should have only read one slice by now" << std::endl;
        auto maybeImage2 = reader.getSubimage(box_withoutPadding);

        Image3D imagep = std::move(maybeImage1.value().image);
        Image3D image = std::move(maybeImage2.value().image);

        bool same = isEqual(imagep.getItkImage(), image.getItkImage());
        std::cerr << "Images are the same: " << same << std::endl;

        // bool ok = TiffWriter::writeToFile(output, image);
        // if(!ok){
        //     std::cerr << "Failed to write output file: " << output << std::endl;
        //     return 2;
        // }
    } catch(const std::exception& e){
        std::cerr << "Exception: " << e.what() << std::endl;
        return 3;
    }

    return 0;
}

