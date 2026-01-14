#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "deconvolution/Postprocessor.h"
#include <set>
#include <stdexcept>
#include <iostream>
#include "deconvolution/Preprocessor.h"
#include "backend/BackendFactory.h"
#include "dolphinbackend/Exceptions.h"
#include "HelperClasses.h"
#include "frontend/SetupConfig.h"
#include "itkImageRegionConstIterator.h"

LabeledDeconvolutionExecutor::LabeledDeconvolutionExecutor(){
    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend>)> psfPreprocessFunction = [&](
    const RectangleShape shape,
    std::shared_ptr<PSF> inputPSF,
    std::shared_ptr<IBackend> backend
    ) -> ComplexData* {
        Preprocessor::padToShape(inputPSF->image, shape, PaddingType::ZERO);
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

    // Track which label ranges we've already added to avoid duplicates
    std::set<std::pair<int, int>> addedRanges;
    
    std::set<int> uniqueLabels;
    
    RectangleShape imageSize = image.getShape();
    int endZ = std::min(roi.position.depth + roi.dimensions.depth, imageSize.depth);

    // Get the ITK image for direct access
    ImageType::Pointer itkImage = image.getItkImage();
    
    // Define the region to iterate over based on the ROI
    ImageType::IndexType roiStart;
    roiStart[0] = roi.position.width;
    roiStart[1] = roi.position.height;
    roiStart[2] = roi.position.depth;
    
    ImageType::SizeType roiSize;
    roiSize[0] = roi.dimensions.width;
    roiSize[1] = roi.dimensions.height;
    roiSize[2] = std::min(roi.dimensions.depth, imageSize.depth - roi.position.depth);
    
    ImageType::RegionType roiRegion;
    roiRegion.SetIndex(roiStart);
    roiRegion.SetSize(roiSize);
    
    // Ensure the region is within image bounds
    ImageType::RegionType imageRegion = itkImage->GetLargestPossibleRegion();
    if (!imageRegion.IsInside(roiRegion)) {
        // Crop the ROI to fit within the image bounds
        roiRegion.Crop(imageRegion);
    }
    
    // Iterate through the ROI using ITK iterator
    itk::ImageRegionConstIterator<ImageType> iterator(itkImage, roiRegion);
    for (iterator.GoToBegin(); !iterator.IsAtEnd(); ++iterator) {
        float pixelValue = iterator.Get();
        int labelValue = static_cast<int>(pixelValue);
        uniqueLabels.insert(labelValue);
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