#pragma once
#include <vector>
#include "deconvolution/ImageMap.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "psf/PSF.h"
#include "Image3D.h"

enum class ExecutionStrategy {
    PARALLEL,
    SEQUENTIAL
};

class Label{
public:
    Label() = default;
    Label(int label, Image3D* labelImage) :label(label), labelImage(labelImage){}
    void setLabel(int label) {this->label = label;}
    void setLabelImage(Image3D* labelImage){ this->labelImage = labelImage;}
    void setPSFs(std::vector<std::shared_ptr<PSF>> psfs) {this->assignedPSFs = psfs;}

    cv::Mat getMask(const cv::Rect& roi, int z) const {
        cv::Mat labelSlice = (*labelImage).slices[z](roi);
        
        // Create mask where label equals labelGroup
        cv::Mat mask = (labelSlice == label);
        return mask;
    }
    std::vector<std::shared_ptr<PSF>> getPSFs() const {
        return assignedPSFs;
    }

private:
    int label;
    Image3D* labelImage;
    std::vector<std::shared_ptr<PSF>> assignedPSFs;

};


struct CubeTaskDescriptor {
    int taskId;
    int channelNumber;
    BoxCoordWithPadding paddedBox; 
    size_t estimatedMemoryUsage;
    virtual std::string getType() const = 0;
};

struct StandardCubeTaskDescriptor : public CubeTaskDescriptor{
    std::vector<std::shared_ptr<PSF>> psfs;
    std::string getType() const override {return "standard";}
};

struct LabeledCubeTaskDescriptor : public CubeTaskDescriptor{
   std::string getType() const override {return "labeled";}

};

struct ChannelPlan {
    std::unique_ptr<IBackend> backend;
    std::unique_ptr<DeconvolutionAlgorithm> algorithm;
    ExecutionStrategy executionStrategy;
    Padding imagePadding;
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    size_t totalTasks;
};

class LoadingBar{
public:
    LoadingBar() = default;
    LoadingBar(size_t max) : max(max){}
    void setMax(size_t max) {this->max = max;}
    void reset() {counter.store(0);}
    void update(){
        std::unique_lock<std::mutex> lock(mutex);
        // Calculate progress
        size_t progress = (counter * 100) / max;
        size_t barWidth = 50;
        size_t pos = (counter * barWidth) / max;
        
        // Print progress bar
        std::cerr << "\rDeconvoluting Image [ ";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cerr << "=";
            else if (i == pos) std::cerr << ">";
            else std::cerr << " ";
        }
        std::cerr <<  "] " << std::setw(3) << progress << "% (" 
        
                << counter << "/" << max << ")";
        std::cerr.flush();

    }
    
    void addOne(){
        ++counter;
        update();
    }
private:
    size_t max;
    std::atomic<size_t> counter{0};
    std::mutex mutex;
};





class PaddingStrategy{
public:
    Padding getPadding(const RectangleShape& imageSize, const std::vector<RectangleShape>& psfSizes) const {
        RectangleShape maxPsfShape{0, 0, 0};
        
        // Find the largest PSF dimensions
        for (const auto& psf : psfSizes) {

            
            maxPsfShape.width = std::max(maxPsfShape.width, psf.width);
            maxPsfShape.height = std::max(maxPsfShape.height, psf.height);
            maxPsfShape.depth = std::max(maxPsfShape.depth, psf.depth);
        }
        
        RectangleShape paddingbefore = RectangleShape(
            static_cast<int>(maxPsfShape.width / 2),
            static_cast<int>(maxPsfShape.height / 2),
            static_cast<int>(maxPsfShape.depth / 2)
        );
        paddingbefore = paddingbefore + 1;
        // paddingbefore = RectangleShape{10,10,20}; //TESTVALUE
        return Padding{paddingbefore, paddingbefore};
    }
};



std::vector<BoxCoordWithPadding> splitImageHomogeneous(
    const RectangleShape& subimageShape,
    const Padding& cubePadding,
    const RectangleShape& imageOriginalShape)
{
    std::vector<BoxCoordWithPadding> cubePositions;
    // Calculate number of cubes in each dimension
    int cubesInDepth = std::max(1,(imageOriginalShape.depth + subimageShape.depth - 1) / subimageShape.depth);
    int cubesInWidth = std::max(1,(imageOriginalShape.width + subimageShape.width - 1) / subimageShape.width);
    int cubesInHeight = std::max(1,(imageOriginalShape.height + subimageShape.height - 1) / subimageShape.height);
    
    // Calculate total number of cubes
    int totalCubes = cubesInDepth * cubesInWidth * cubesInHeight;
    cubePositions.reserve(totalCubes);

    for (int d = 0; d < cubesInDepth; ++d) {
        for (int w = 0; w < cubesInWidth; ++w) {
            for (int h = 0; h < cubesInHeight; ++h) {
                
                // Calculate current position in original image coordinates
                RectangleShape currentPos(
                    w * subimageShape.width,
                    h * subimageShape.height,
                    d * subimageShape.depth
                );

                // Calculate remaining size for this cube
                RectangleShape remainingSize(
                    std::min(subimageShape.width, imageOriginalShape.width - w * subimageShape.width),
                    std::min(subimageShape.height, imageOriginalShape.height - h * subimageShape.height),
                    std::min(subimageShape.depth, imageOriginalShape.depth - d * subimageShape.depth)
                );

                // Skip if no remaining size (shouldn't happen with proper calculation)
                if (remainingSize.depth <= 0 || remainingSize.width <= 0 || remainingSize.height <= 0) {
                    continue;
                }

                // Determine actual cube positions - use overlap for boundary cubes
                RectangleShape actualPos = currentPos;
                RectangleShape actualDimensions = subimageShape;
                Padding adjustedPadding = cubePadding;
                
                // Check if padded cube exceeds image size and adjust accordingly
                RectangleShape paddedCubeSize = subimageShape + cubePadding.before + cubePadding.after;
                
                // If padded cube is larger than image in any dimension, adjust to make padding after larger while making dimensions of box smaller
                if (subimageShape.width > imageOriginalShape.width) {
                    actualDimensions.width = imageOriginalShape.width;
                    adjustedPadding.after.width = cubePadding.after.width + subimageShape.width - imageOriginalShape.width;
                }
                if (subimageShape.height > imageOriginalShape.height) {
                    actualDimensions.height = imageOriginalShape.height;
                    adjustedPadding.after.height = cubePadding.after.height + subimageShape.height - imageOriginalShape.height;
                }
                if (subimageShape.depth > imageOriginalShape.depth) {
                    actualDimensions.depth = imageOriginalShape.depth;
                    adjustedPadding.after.depth = cubePadding.after.depth + subimageShape.depth - imageOriginalShape.depth;
                }
                
                // If this would be the last cube and doesn't fit completely, shift it back to create overlap
                if (remainingSize.depth < actualDimensions.depth && remainingSize.depth > 0) {
                    actualPos.depth = currentPos.depth - (actualDimensions.depth - remainingSize.depth);
                }
                if (remainingSize.width < actualDimensions.width && remainingSize.width > 0) {
                    actualPos.width = currentPos.width - (actualDimensions.width - remainingSize.width);
                }
                if (remainingSize.height < actualDimensions.height && remainingSize.height > 0) {
                    actualPos.height = currentPos.height - (actualDimensions.height - remainingSize.height);
                }
                
                BoxCoord cube;
                cube.position = actualPos;
                cube.dimensions = actualDimensions;
                
                BoxCoordWithPadding cubeWithPadding;
                cubeWithPadding.box = cube;
                cubeWithPadding.padding = adjustedPadding;
                
                cubePositions.push_back(std::move(cubeWithPadding));
            }
        }
    }

    return cubePositions;
}

