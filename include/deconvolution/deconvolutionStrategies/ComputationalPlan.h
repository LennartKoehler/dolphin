#pragma once
#include <vector>
#include "deconvolution/DeconvolutionConfig.h"
#include "psf/PSF.h"
#include "Image3D.h"
#include <atomic>
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>

enum class ExecutionStrategy {
    PARALLEL,
    SEQUENTIAL
};

class Label{
public:
    Label() = default;
    // Label(Image3D* labelImage) :labelImage(labelImage){}
    // void setLabelImage(Image3D* labelImage){ this->labelImage = labelImage;}
    void setRange(Range<std::shared_ptr<PSF>> psfs) {this->psfs = psfs;}

    // cv::Mat getMask(const cv::Rect& roi, int z) const {
    //     cv::Mat labelSlice = (*labelImage).slices[z](roi);
        
    //     // Create mask where label is inrange of labelgroup 
    //     cv::Mat mask;
    //     cv::inRange(labelSlice, psfs.start, psfs.end, mask); // -1 because its inclusive, we dont want that
    //     return mask;
    // }
    Image3D getMask(const Image3D& labelImage) const {
        Image3D result;
        for (auto& slice : labelImage.slices){
            cv::Mat mask;
            // cv::Rect cvRoi(roi.position.width, roi.position.height, roi.dimensions.width, roi.dimensions.height);
            // cv::Mat labelSlice = labelImage.slices[z];
            cv::inRange(slice, psfs.start, psfs.end, mask);
            result.slices.push_back(mask);
        }
        return result;
    }

    std::vector<std::shared_ptr<PSF>> getPSFs() const {
        return psfs.values;
    }

private:
    Range<std::shared_ptr<PSF>> psfs; 
    // Image3D* labelImage;
    Image3D mask;


};


struct CubeTaskDescriptor {
    int taskId;
    int channelNumber;
    BoxCoordWithPadding paddedBox;
    std::shared_ptr<IBackend> backend;
    std::shared_ptr<DeconvolutionAlgorithm> algorithm;
    size_t estimatedMemoryUsage;
    std::vector<std::shared_ptr<PSF>> psfs;
};



struct ChannelPlan {
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
    const RectangleShape& imageOriginalShape);