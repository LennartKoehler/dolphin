/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#include "dolphin/deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "dolphin/Image3D.h"
#include "dolphin/deconvolution/Postprocessor.h"
#include <functional>
#include <itkImageRegionIterator.h>
#include <set>
#include <stdexcept>
#include <iostream>
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/generators/GaussianPSFGenerator.h"
#include "dolphinbackend/Exceptions.h"
#include "dolphin/HelperClasses.h"
#include "dolphin/SetupConfig.h"
#include <itkImageRegionConstIterator.h>
#include <spdlog/spdlog.h>

LabeledDeconvolutionExecutor::LabeledDeconvolutionExecutor(){
}


void LabeledDeconvolutionExecutor::configure(const SetupConfig& setupConfig, const DeconvolutionConfig& deconvConfig, progressCallbackFn fn){
    int channel = 0;
    this->labelReader = std::make_unique<TiffReader>(setupConfig.labeledImage, channel);
    this->loadingBar.setCallback(fn);

    // Load PSF label map if provided
    if (!setupConfig.labelPSFMap.empty()) {
        RangeMap<std::string> labelPSFMap;
        labelPSFMap.loadFromString(setupConfig.labelPSFMap);
        this->psfLabelMap = labelPSFMap;
    }

    this->featheringRadius = deconvConfig.featheringRadius;
}

/*
Deconvolution using a labelimage which allows for different psfs for different parts of the image.
For each unique label within a cube the deconvolution is performed, and at the end the deconvolved images are stitched together
according to the specifications in the labelimage.
TODO still slow
*/
void LabeledDeconvolutionExecutor::runTask(const CubeTaskDescriptor& task){

    std::shared_ptr<TaskContext> context = task.context;
    thread_local IBackend& iobackend = context->manager.getBackend(context->ioconfig);
    thread_local IBackend& workerbackend = context->manager.cloneSharedMemory(iobackend, context->workerconfig); // copied in deconvolutionprocessor



    std::shared_ptr<ImageReader> reader = task.reader;
    std::shared_ptr<ImageWriter> writer = task.writer;
    if (reader->getMetaData().getShape() != labelReader->getMetaData().getShape()){
        throw std::runtime_error("Size of input image is not the same as label image"); // this shouldnt really happend in the runTask function
    }


    CuboidShape workShape = task.paddedBox.box.dimensions + task.paddedBox.padding.before + task.paddedBox.padding.after;

    std::optional<PaddedImage> cubeImage_o = reader->getSubimage(task.paddedBox);
    std::optional<PaddedImage> labelImage_o = labelReader->getSubimage(task.paddedBox);
    if (!cubeImage_o.has_value()){
        throw std::runtime_error("LabeledDeconvolutionExecutor: No input image recieved from reader");
    }
    if (!labelImage_o.has_value()){
        throw std::runtime_error("LabeledDeconvolutionExecutor: No label image recieved from reader");
    }
    PaddedImage& inputCube = *cubeImage_o;
    const PaddedImage& labelImage = *labelImage_o;


    RealData g_host = Preprocessor::convertImageToRealData(inputCube.image);

    RealData g_device = iobackend.getMemoryManager().copyDataToDevice(g_host);

    BackendFactory::getInstance().getDefaultBackendMemoryManager().freeMemoryOnDevice(g_host);

    std::vector<Label<Image3D>> tasklabels = getLabelGroups(
        task.psfs,
        labelImage.image,
        psfLabelMap
    );

    Image3D result(workShape, 0.0f);

    std::shared_ptr<PSF> gaussianKernel = createGaussianKernel(featheringRadius);

    // make the memory and preprocessing of this kernel which is used for the masks be maanged by the psfpreprocessor
    const ComplexData* preprocessedGaussianKernel = context->psfpreprocessor->getPreprocessedPSF(workShape, gaussianKernel, workerbackend);
    std::vector<Label<RealData>> preprocessedTaskLabels = makeMasksWeighted(tasklabels, labelImage.image, *preprocessedGaussianKernel, iobackend);


    for (const Label<RealData>& labelgroup : preprocessedTaskLabels){
        std::vector<std::shared_ptr<PSF>> psfs = labelgroup.getPSFs();

        using progressFunction = std::function<void(int)>;
        progressFunction tracker = [this, numberLabels = tasklabels.size(), numPsfs = psfs.size()](int maxiterations){
            float iteration = 1.0 / (maxiterations * numberLabels * numPsfs);
            this->loadingBar.add(iteration);
        };

        if (psfs.size() == 0){
        }

        else{
            RealData local_g_device = iobackend.getMemoryManager().createCopy(g_device);
            RealData f_device = iobackend.getMemoryManager().allocateMemoryOnDeviceRealFFTInPlace(workShape);

            std::unique_ptr<DeconvolutionAlgorithm> algorithm = task.algorithm->clone();
            algorithm->setProgressTracker(tracker);

            std::future<void> resultDone = context->processor.deconvolveSingleCube(
                workerbackend,
                std::move(algorithm),
                workShape,
                psfs,
                local_g_device,
                f_device,
                *context->psfpreprocessor.get());

            resultDone.get(); //wait for result
            iobackend.sync();

            // TiffWriter::writeToFile("/home/lennart-k-hler/data/dolphin_results/image.tif", Preprocessor::convertComplexDataToImage(f_device));

            iobackend.getDeconvManager().multiplication(f_device, *labelgroup.getMask(), f_device); // multiply with weighted mask to get weighted values
            RealData f_host = iobackend.getMemoryManager().moveDataFromDevice(f_device, BackendFactory::getInstance().getDefaultBackendMemoryManager());

            Postprocessor::addCubeToImage(Preprocessor::convertRealDataToImage(f_host), result);

        }
    }
    writer->setSubimage(result, task.paddedBox);
}

std::shared_ptr<PSF> LabeledDeconvolutionExecutor::createGaussianKernel(int featheringRadius){
    // TODO appropriately use featheringRadius
    int sizeX = 20;
    int sizeY = 20;
    int sizeZ = 20;
    float sigmaX = 5;
    float sigmaY = 5;
    float sigmaZ = 5;

    std::shared_ptr<GaussianPSFConfig> config = std::make_shared<GaussianPSFConfig>();
    config->sizeX = sizeX;
    config->sizeY = sizeY;
    config->sizeZ = sizeZ;
    config->sigmaX = sigmaX;
    config->sigmaY = sigmaY;
    config->sigmaZ = sigmaZ;

    GaussianPSFGenerator generator(config);
    std::shared_ptr<PSF> gaussianKernel = std::make_shared<PSF>(generator.generatePSF());
    return gaussianKernel;
}


std::vector<Label<RealData>> LabeledDeconvolutionExecutor::makeMasksWeighted(
    std::vector<Label<Image3D>>& labels,
    const Image3D& labelImage,
    const ComplexData& frequencyFeatheringKernel,
    IBackend& backend
) const {
    std::vector<Label<RealData>> newlabels;
    std::vector<RealData*> binaryMasks; // for later access
    for (int i = 0; i < labels.size(); i++){
        Image3D& image = *labels[i].getMask();
        RealData mask = Preprocessor::convertImageToRealData(image);
        newlabels.emplace_back(Label<RealData>{std::move(mask), labels[i].getPSFs()});
        binaryMasks.push_back(newlabels[i].getMask());
    }

    Postprocessor::createWeightMasks(
        binaryMasks,
        frequencyFeatheringKernel,
        backend);

    return newlabels;
}


std::vector<std::shared_ptr<PSF>> getPSFForLabel(std::vector<std::string>& psfids, const std::vector<std::shared_ptr<PSF>>& psfs) {
    std::vector<std::shared_ptr<PSF>> assignedpsfs;

    bool found = false;
    for (const auto& psfid : psfids) {
        found = false;
        for (const auto& psf : psfs) {
            if (psf->ID == psfid) {
                assignedpsfs.push_back(psf);
                found = true;
                break;
            }
        }

        if (!found)
            spdlog::get("deconvolution")->warn("Cant find a PSF for the specified PSF ID ({}) please check your input", psfid);
    }
    return assignedpsfs;
}

// create all masks in one iteration of the labelImage
void CreateMasksOperation::operator()(
    size_t pixelIndex, float pixelValue)
{
    int labelValue = static_cast<int>(pixelValue);

    if (labelValue != lastLabel){ // if not same as last mask, which doesnt happen often
        // then update the index to access the correct image
        std::vector<Range<std::string>> ranges = psfLabelMap.get(labelValue);
        if (ranges.empty())return; // no psf for that label

        int l_maskIndex = 0;
        std::vector<std::string> psfIDs = ranges[0].get(); // overlap not allowed
        bool found = false;
        // high price for look up if this happens often
        for (auto& [label, image] : masks){
            if (psfIDs == label){
                maskIndex = l_maskIndex;
                found = true;
                break;
            }
            l_maskIndex ++;
        }
        // if labelValue has no image yet, then create a new mask for that label
        if (!found){
            masks.emplace_back(
                std::pair<std::vector<std::string>, Label<Image3D>>
                (psfIDs, Label<Image3D>{Image3D(maskSize, 0.0), getPSFForLabel(psfIDs, psfs)}));

            maskIndex = masks.size() - 1;
        }
        lastLabel = labelValue;
    }
    Image3D* image = masks[maskIndex].second.getMask();
    ImageType::Pointer imagep = image->getItkImage();
    auto* buffer = imagep->GetBufferPointer();
    buffer[pixelIndex] = 1;
}

// destructive
std::vector<Label<Image3D>> CreateMasksOperation::getLabels(){
    std::vector<Label<Image3D>> result;
    for (auto& [id, label] : masks){
        result.push_back(std::move(label));
    }
    return result;
}



std::vector<Label<Image3D>> LabeledDeconvolutionExecutor::getLabelGroups(
		const std::vector<std::shared_ptr<PSF>>& psfs,
		const Image3D& image,
		RangeMap<std::string> psfLabelMap)
{
    std::vector<Label<Image3D>> labelGroups;

    CreateMasksOperation createMasksOp(image.getShape(), psfs, psfLabelMap);
    std::vector<std::reference_wrapper<IConstImageOperation>> operations{createMasksOp};
    image.executeOperations(operations);
    return createMasksOp.getLabels();

}

