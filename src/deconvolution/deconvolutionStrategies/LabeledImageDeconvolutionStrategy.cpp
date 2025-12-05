#include "deconvolution/deconvolutionStrategies/LabeledImageDeconvolutionStrategy.h"
#include "deconvolution/Postprocessor.h"
#include <set>


void LabeledImageDeconvolutionStrategy::setLabeledImage(std::shared_ptr<Hyperstack> labelImage) {
    this->labelImage = labelImage;
}

void LabeledImageDeconvolutionStrategy::setLabelPSFMap(const RangeMap<std::string>& labelPSFMap) {
    this->psfLabelMap = labelPSFMap;
}

ComputationalPlan LabeledImageDeconvolutionStrategy::createComputationalPlan(
    int channelNumber,
    const Image3D& image,
    const std::vector<PSF>& psfs,
    const DeconvolutionConfig& config,
    const std::shared_ptr<IBackend> backend,
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm
) {

    // validateConfiguration(psfs, imageShape, channelNumber, config, backend, algorithm);
    ComputationalPlan plan;
    plan.imagePadding = Padding{ RectangleShape{30,30,30}, RectangleShape{30,30,30} }; //TESTVALUE
    // plan.imagePadding = getChannelPadding();
    plan.executionStrategy = ExecutionStrategy::PARALLEL;



    std::vector<std::shared_ptr<PSF>> psfPointers;
     for (const auto& psf : psfs) {
        psfPointers.push_back(std::make_shared<PSF>(psf));
    }
    this->psfs = psfPointers;

    // Use the imageShape parameter
    RectangleShape imageSize = RectangleShape{ image.slices[0].cols, image.slices[0].rows, static_cast<int>(image.slices.size()) };

    size_t t = config.nThreads;
    // t = static_cast<size_t>(2);
    size_t memoryPerCube = maxMemoryPerCube(t, config.maxMem_GB * 1e9, algorithm);
    Padding cubePadding = getCubePadding(imageSize, psfs);
    RectangleShape idealCubeSize = getCubeShape(memoryPerCube, config.nThreads, imageSize, cubePadding);

    // idealCubeSize = RectangleShape(393, 313, 46);
    std::vector<BoxCoord> cubeCoordinates = splitImageHomogeneous(idealCubeSize, imageSize);
    // Create task descriptors for each cube

    plan.tasks.reserve(cubeCoordinates.size());
    for (size_t i = 0; i < cubeCoordinates.size(); ++i) {
        LabeledCubeTaskDescriptor descriptor;
        descriptor.taskId = static_cast<int>(i);
        descriptor.srcBox = cubeCoordinates[i];
        descriptor.channelNumber = channelNumber;
        descriptor.estimatedMemoryUsage = estimateMemoryUsage(idealCubeSize, algorithm);
        descriptor.requiredPadding = cubePadding;

        plan.tasks.push_back(std::make_unique<LabeledCubeTaskDescriptor>(descriptor));
    }

    plan.totalTasks = plan.tasks.size();
    return plan;
}

std::vector<Label> LabeledImageDeconvolutionStrategy::getLabelGroups(int channelNumber, const BoxCoord& roi, std::vector<std::shared_ptr<PSF>>& psfs) {
    std::vector<Label> labelGroups;

    Image3D* labelChannel = &labelImage->channels[channelNumber].image;

    // Check if labelChannel is available
    if (!labelImage) {
        // Return empty vector if no label image is available
        return labelGroups;
    }

    // Get unique labels within the ROI
    std::set<int> uniqueLabels;

    // Iterate through the region defined by the BoxCoord
    int endZ = std::min(roi.z + roi.dimensions.depth, static_cast<int>(labelChannel->slices.size()));

    for (int z = roi.z; z < endZ; ++z) {
        // Ensure z is within bounds
        if (z >= 0 && z < static_cast<int>(labelChannel->slices.size())) {
            const cv::Mat& slice = labelChannel->slices[z];

            // Define ROI within the slice
            cv::Rect sliceRoi(roi.x + 500, roi.y + 500, roi.dimensions.width, roi.dimensions.height); //TESTVALUE

            // Ensure ROI is within slice bounds
            sliceRoi &= cv::Rect(0, 0, slice.cols, slice.rows);

            if (!sliceRoi.empty()) {
                cv::Mat roiSlice = slice(sliceRoi);

                // Handle different data types for label images
                if (roiSlice.type() == CV_32S) {
                    // 32-bit signed integers
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = roiSlice.at<int>(y, x);
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else if (roiSlice.type() == CV_16U) {
                    // 16-bit unsigned integers
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = static_cast<int>(roiSlice.at<uint16_t>(y, x));
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else if (roiSlice.type() == CV_8U) {
                    // 8-bit unsigned integers
                    for (int y = 0; y < roiSlice.rows; ++y) {
                        for (int x = 0; x < roiSlice.cols; ++x) {
                            int labelValue = static_cast<int>(roiSlice.at<uint8_t>(y, x));
                            uniqueLabels.insert(labelValue);
                        }
                    }
                }
                else {
                    // Convert to 32-bit integers for any other type
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

    // Create Label objects for each unique label
    labelGroups.reserve(uniqueLabels.size());
    for (int label : uniqueLabels) {
        // Skip background label (typically 0)
        Label labelgroup;
        labelgroup.setLabel(label);
        labelgroup.setLabelImage(labelChannel);
        labelgroup.setPSFs(getPSFForLabel(label, psfs));
        labelGroups.push_back(labelgroup);
 
    }

    return labelGroups;
}

std::vector<std::shared_ptr<PSF>> LabeledImageDeconvolutionStrategy::getPSFForLabel(int label, std::vector<std::shared_ptr<PSF>>& psfs) {
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

std::function<void()> LabeledImageDeconvolutionStrategy::createTask(
    const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
    const PaddedImage& inputImagePadded,
    std::vector<cv::Mat>& outputImage,
    std::mutex& writerMutex
) {
    LabeledCubeTaskDescriptor* standardTask = dynamic_cast<LabeledCubeTaskDescriptor*>(taskDesc.get());
    if (!standardTask) {
        throw std::runtime_error("Expected LabeledCubeTaskDescriptor but got different type");
    }

    std::function<void()> taskfunc = [this, task = *standardTask, &inputImagePadded, &writerMutex, &outputImage]() {
        
        RectangleShape workShape = task.srcBox.dimensions + task.requiredPadding.before + task.requiredPadding.after;
        // PaddedImage cubeImage = getCubeImage(inputImagePadded, taskDesc.srcBox, taskDesc.requiredPadding);

        PaddedImage cubeImage = getCubeImage(inputImagePadded, task.srcBox, task.requiredPadding);
        std::vector<Label> labelgroups = getLabelGroups(task.channelNumber, task.srcBox, psfs);
        std::shared_ptr<IBackend> iobackend = backend_->onNewThread();
        ComplexData g_host = convertCVMatVectorToFFTWComplex(cubeImage.image, workShape);
        ComplexData g_device = iobackend->getMemoryManager().copyDataToDevice(g_host);
        cpuMemoryManager->freeMemoryOnDevice(g_host);

        for (const Label& labelgroup : labelgroups){

            ComplexData local_g_device = iobackend->getMemoryManager().copyData(g_device);
            ComplexData f_host{cpuMemoryManager.get(), nullptr, RectangleShape()};

            ComplexData f_device = iobackend->getMemoryManager().allocateMemoryOnDevice(workShape);
            std::vector<std::shared_ptr<PSF>> psfs = labelgroup.getPSFs();
            if (psfs.size() != 0){
                try {
                    std::future<void> resultDone = processor.deconvolveSingleCube(
                        iobackend,
                        algorithm_,
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
                resultCube.padding = cubeImage.padding;
                resultCube.image = convertFFTWComplexToCVMatVector(f_host);
                {
                    std::unique_lock<std::mutex> lock(writerMutex);
                    // deconovlutionStrategy.insertImage(cubeImage, outputImage);
                    Postprocessor::insertLabeledCubeInImage(resultCube, outputImage, task.srcBox, labelgroup);
                    // Postprocessor::insertCubeInImage(resultCube, outputImage, task.srcBox); //TESTVALUE
                }
            }
            break;
        }

        iobackend->releaseBackend();

        loadingBar.addOne();
    };

    return taskfunc;
}




