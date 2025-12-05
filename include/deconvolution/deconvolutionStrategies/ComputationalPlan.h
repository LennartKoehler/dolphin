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
    BoxCoord srcBox;
    Padding requiredPadding;
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

struct ComputationalPlan {
    ExecutionStrategy executionStrategy;
    Padding imagePadding;
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    size_t totalTasks;
};

