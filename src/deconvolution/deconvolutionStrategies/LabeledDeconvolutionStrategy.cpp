#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionStrategy.h"
#include "deconvolution/Postprocessor.h"
#include <set>
#include <stdexcept>
#include "UtlImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#include "deconvolution/Preprocessor.h"
#include "backend/BackendFactory.h"
#include "backend/Exceptions.h"
#include "deconvolution/ImageMap.h"
#include "HelperClasses.h"



ChannelPlan LabeledDeconvolutionStrategy::createPlan(
    const ImageMetaData& metadata, 
    const std::vector<PSF>& psfs,
    const DeconvolutionConfig& config

) {


    std::vector<std::shared_ptr<PSF>> psfPointers;
    for (const auto& psf : psfs) {
        psfPointers.push_back(std::make_shared<PSF>(psf));
    }
 
    RectangleShape imageSize = RectangleShape{metadata.imageWidth, metadata.imageLength, metadata.slices};
    std::unique_ptr<DeconvolutionAlgorithm> algorithm = getAlgorithm(config);
    std::unique_ptr<IBackend> backend = getBackend(config);

    size_t t = config.nThreads;
    size_t memoryPerCube = maxMemoryPerCube(t, config.maxMem_GB * 1e9, algorithm.get());
    Padding cubePadding = getCubePadding(imageSize, psfs);
    RectangleShape idealCubeSize = getCubeShape(memoryPerCube, config.nThreads, imageSize, cubePadding);
    Padding imagePadding = getImagePadding(imageSize, idealCubeSize, cubePadding);
    
    std::vector<BoxCoordWithPadding> cubeCoordinatesWithPadding = splitImageHomogeneous(idealCubeSize, cubePadding, imageSize);
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    tasks.reserve(cubeCoordinatesWithPadding.size());
    
    for (size_t i = 0; i < cubeCoordinatesWithPadding.size(); ++i) {
        LabeledCubeTaskDescriptor descriptor;
        descriptor.taskId = static_cast<int>(i);
        descriptor.paddedBox = cubeCoordinatesWithPadding[i];
        descriptor.channelNumber = 0; // Default channel, can be modified as needed
        descriptor.estimatedMemoryUsage = estimateMemoryUsage(idealCubeSize, algorithm.get());

        tasks.push_back(std::make_unique<LabeledCubeTaskDescriptor>(descriptor));
    }

    size_t totalTasks = tasks.size();
    return ChannelPlan{
        std::move(backend),
        std::move(algorithm),
        ExecutionStrategy::PARALLEL,
        std::move(imagePadding),
        std::move(tasks),
        std::move(totalTasks)
    };
}



void LabeledDeconvolutionStrategy::setLabeledImage(std::shared_ptr<Hyperstack> labelImage) {
    this->labelImage = labelImage;
}

void LabeledDeconvolutionStrategy::setLabelPSFMap(const RangeMap<std::string>& labelPSFMap) {
    this->psfLabelMap = labelPSFMap;
}

std::vector<Label> LabeledDeconvolutionStrategy::getLabelGroups(int channelNumber, const BoxCoord& roi, std::vector<std::shared_ptr<PSF>>& psfs) {
    std::vector<Label> labelGroups;

    if (!labelImage) {
        return labelGroups;
    }

    Image3D* labelChannel = &labelImage->channels[channelNumber].image;
    std::set<int> uniqueLabels;

    int endZ = std::min(roi.position.depth + roi.dimensions.depth, static_cast<int>(labelChannel->slices.size()));

    for (int z = roi.position.depth; z < endZ; ++z) {
        if (z >= 0 && z < static_cast<int>(labelChannel->slices.size())) {
            const cv::Mat& slice = labelChannel->slices[z];
            cv::Rect sliceRoi(roi.position.width, roi.position.height, roi.dimensions.width, roi.dimensions.height);
            sliceRoi &= cv::Rect(0, 0, slice.cols, slice.rows);

            if (!sliceRoi.empty()) {
                cv::Mat roiSlice = slice(sliceRoi);

                if (roiSlice.type() == CV_32S) {
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = roiSlice.at<int>(y, x);
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else if (roiSlice.type() == CV_16U) {
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = static_cast<int>(roiSlice.at<uint16_t>(y, x));
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else if (roiSlice.type() == CV_8U) {
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = static_cast<int>(roiSlice.at<uint8_t>(y, x));
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else {
                    cv::Mat convertedRoi;
                    roiSlice.convertTo(convertedRoi, CV_32S);
                    for (int y = 0; y < convertedRoi.rows; ++y) {
                        for (int x = 0; x < convertedRoi.cols; ++x) {
                            int labelValue = convertedRoi.at<int>(y, x);
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
            }
        }
    }

    labelGroups.reserve(uniqueLabels.size());
    for (int label : uniqueLabels) {
        Label labelgroup;
        labelgroup.setLabel(label);
        labelgroup.setLabelImage(labelChannel);
        labelgroup.setPSFs(getPSFForLabel(label, psfs));
        labelGroups.push_back(labelgroup);
    }

    return labelGroups;
}

std::vector<std::shared_ptr<PSF>> LabeledDeconvolutionStrategy::getPSFForLabel(int label, std::vector<std::shared_ptr<PSF>>& psfs) {
    std::vector<std::string> psfids = psfLabelMap.get(label);
    std::vector<std::shared_ptr<PSF>> assignedpsfs;

    for (const auto& psfid : psfids) {
        for (const auto& psf : psfs) {
            if (psf->ID == psfid) {
                assignedpsfs.push_back(psf);
            }
        }
    }
    return assignedpsfs;
}


RectangleShape LabeledDeconvolutionStrategy::getCubeShape(
    size_t memoryPerCube,
    size_t numberThreads,
    const RectangleShape& imageOriginalShape,
    const Padding& cubePadding
){
    size_t width = 128;
    size_t height = 256;
    size_t depth = 128;

    return RectangleShape(width, height, depth) - cubePadding.before - cubePadding.after;
}

Padding LabeledDeconvolutionStrategy::getImagePadding(
    const RectangleShape& imageSize,
    const RectangleShape& cubeSizeUnpadded,
    const Padding& cubePadding
){
    RectangleShape paddingBefore = cubePadding.before;
    RectangleShape paddingAfter;

    paddingAfter.width = std::max(cubePadding.after.width, cubeSizeUnpadded.width - imageSize.width + cubePadding.before.width);
    paddingAfter.height = std::max(cubePadding.after.height, cubeSizeUnpadded.height - imageSize.height + cubePadding.before.height);
    paddingAfter.depth = std::max(cubePadding.after.depth, cubeSizeUnpadded.depth - imageSize.depth + cubePadding.before.depth);
    return Padding{paddingBefore, paddingAfter};
}

Padding LabeledDeconvolutionStrategy::getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs){
    std::vector<RectangleShape> psfSizes;
    for (const auto& psf : psfs){
        psfSizes.push_back(psf.image.getShape());
    }
    
    RectangleShape maxPsfShape{0, 0, 0};
    
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
    return Padding{paddingbefore, paddingbefore};
}