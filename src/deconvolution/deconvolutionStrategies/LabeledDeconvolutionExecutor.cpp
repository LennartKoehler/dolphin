#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
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
#include "HelperClasses.h"
#include "frontend/SetupConfig.h"

LabeledDeconvolutionExecutor::LabeledDeconvolutionExecutor(){
    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
    const RectangleShape shape,
    std::shared_ptr<PSF> inputPSF,
    std::shared_ptr<IBackend> backend
    ) -> ComplexData* {
        Preprocessor::padToShape(inputPSF->image, shape, 0);
        ComplexData h = convertCVMatVectorToFFTWComplex(inputPSF->image, shape);
        ComplexData h_device = backend->getMemoryManager().copyDataToDevice(h);
        backend->getDeconvManager().octantFourierShift(h_device);
        backend->getDeconvManager().forwardFFT(h_device, h_device);
        backend->sync();
        return new ComplexData(std::move(h_device));
    };
    psfPreprocessor.setPreprocessingFunction(psfPreprocessFunction);

}


void LabeledDeconvolutionExecutor::configure(const SetupConfig& setupConfig){
    this->labelReader = std::make_unique<TiffReader>(setupConfig.labeledImage);
    this->featheringRadius = setupConfig.featheringRadius;
    // Load PSF label map if provided
    if (!setupConfig.labelPSFMap.empty()) {
        RangeMap<std::string> labelPSFMap;
        labelPSFMap.loadFromString(setupConfig.labelPSFMap);
        this->psfLabelMap = labelPSFMap;
    }

}


std::function<void()> LabeledDeconvolutionExecutor::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const ImageReader& reader,
    const ImageWriter& writer) {
    
    return [this, task = *taskDesc, &reader, &writer]() {

        RectangleShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;
        PaddedImage cubeImage = reader.getSubimage(task.paddedBox);
        PaddedImage labelImage = labelReader->getSubimage(task.paddedBox);
        
        std::shared_ptr<IBackend> iobackend = task.backend->onNewThread(task.backend);
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);
        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);

        // TODO is this async safe?
        std::vector<Label> tasklabels = getLabelGroups(
            task.channelNumber,
            BoxCoord{RectangleShape(0,0,0), workShape},
            task.psfs,
            labelImage.image,
            psfLabelMap
        );

        std::vector<ImageMaskPair> tempResults;
        tempResults.reserve(tasklabels.size());
        Image3D result = cubeImage.image;

        for (const Label& labelgroup : tasklabels){
            std::vector<std::shared_ptr<PSF>> psfs = labelgroup.getPSFs();
            
            if (psfs.size() != 0){
                ComplexData local_g_device = iobackend->getMemoryManager().copyData(g_device);
                ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};

                ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);

                std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();
                try {
                    std::future<void> resultDone = processor.deconvolveSingleCube(
                        iobackend,
                        std::move(algorithm),
                        workShape,
                        psfs,
                        local_g_device,
                        f_device,
                        psfPreprocessor);

                    resultDone.get(); //wait for result
                    f_host = iobackend->getMemoryManager().moveDataFromDevice(f_device, *cpuMemoryManager);
                }
                catch (...) {
                    throw; // dont overwrite image if exception
                }
                PaddedImage resultCube;
                resultCube.padding = task.paddedBox.padding;
                resultCube.image = convertFFTWComplexToCVMatVector(f_host);

                ImageMaskPair pair{resultCube.image, labelgroup.getMask(labelImage.image)};
                tempResults.push_back(pair);
            }


        }
        float epsilon = 5;
        
        if (tempResults.size() > 1){
            result = Postprocessor::addFeathering(tempResults, featheringRadius, epsilon);
        }
        else if (tempResults.size() == 1){
            result = tempResults[0].image;
        }

        writer.setSubimage(result, task.paddedBox);
        iobackend->releaseBackend();
        loadingBar.addOne();
    };
}




std::vector<Label> LabeledDeconvolutionExecutor::getLabelGroups(
        int channelNumber,
		const BoxCoord& roi,
		const std::vector<std::shared_ptr<PSF>>& psfs,
		const Image3D& image,
		RangeMap<std::string> psfLabelMap) {
    std::vector<Label> labelGroups;


    // Image3D* labelChannel = &labelImage->channels[channelNumber].image;
    

    // Track which label ranges we've already added to avoid duplicates
    std::set<std::pair<int, int>> addedRanges;
    
    std::set<int> uniqueLabels;
    

    int endZ = std::min(roi.position.depth + roi.dimensions.depth, static_cast<int>(image.slices.size()));

    for (int z = roi.position.depth; z < endZ; ++z) {
        if (z >= 0 && z < static_cast<int>(image.slices.size())) {
            const cv::Mat& slice = image.slices[z];
            cv::Rect sliceRoi(roi.position.width, roi.position.height, roi.dimensions.width, roi.dimensions.height);
            sliceRoi &= cv::Rect(0, 0, slice.cols, slice.rows);

            if (!sliceRoi.empty()) {
                cv::Mat roiSlice = slice(sliceRoi);
               
                cv::Mat convertedRoi;
                roiSlice.convertTo(convertedRoi, CV_32S);
                //TODO for different cv mat types
                for (int y = 0; y < convertedRoi.rows; ++y) {
                    for (int x = 0; x < convertedRoi.cols; ++x) {
                        int labelValue = convertedRoi.at<int>(y, x);
                        uniqueLabels.insert(labelValue);
                    }
                }
            }
        }
    }

    for (int label : uniqueLabels) {
        std::vector<Range<std::string>> psfids = psfLabelMap.get(label);
        if(psfids.size() > 1){

            // TODO if overlap, then create new range of that overlap with the combined psfs, should they ever overlap?
            throw std::runtime_error("PSF Maps cant overlap");
        }
        if (psfids.size() != 0){
            // Create a unique key for this label range
            std::pair<int, int> rangeKey = {psfids[0].start, psfids[0].end};
            
            // Only add if we haven't already added this range
            if (addedRanges.find(rangeKey) == addedRanges.end()) {
                Label labelgroup;
                int start = psfids[0].start;
                int end = psfids[0].end;
                end = end ? start == end : end - 1; // because in rangemap the end is exclusive while in range its inclusive;
                labelgroup.setRange(Range<std::shared_ptr<PSF>>(psfids[0].start, psfids[0].end, getPSFForLabel(psfids[0], psfs)));
                labelGroups.push_back(labelgroup);
                addedRanges.insert(rangeKey);
            }
        }
        
    }

    return labelGroups;
}
std::vector<std::shared_ptr<PSF>> LabeledDeconvolutionExecutor::getPSFForLabel(Range<std::string>& psfids, const std::vector<std::shared_ptr<PSF>>& psfs) {
    std::vector<std::shared_ptr<PSF>> assignedpsfs;

    for (const auto& psfid : psfids.get()) {
        for (const auto& psf : psfs) {
            if (psf->ID == psfid) {
                assignedpsfs.push_back(psf);
            }
        }
    }
    if (assignedpsfs.size() == 0){
        std::cout << "Cant find a PSF for the desired Label, please check your input" <<std::endl;
    }
    
    return assignedpsfs;
}