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

#pragma once
#include <vector>
#include "deconvolution/DeconvolutionConfig.h"
#include "psf/PSF.h"
#include "Image3D.h"
#include <atomic>
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "deconvolution/DeconvolutionProcessor.h"
#include "deconvolution/Preprocessor.h"
#include "IO/TiffWriter.h"
#include "IO/TiffReader.h"


/*
The context for a specific CubeTaskDescription. These can be shared between tasks
Currently a TaskContext is created for each Device. So each device has its own workers, psfpreprocessor and ioThreads
*/
struct TaskContext{
    TaskContext(
        std::shared_ptr<IBackend> backend,
        int nWorkerThreads,
        int nIOThreads
    ) : prototypebackend(std::move(backend)),
          processor(),
          ioPool(nIOThreads)
    {
        processor.init(nWorkerThreads);
    }
    ~TaskContext(){
        psfpreprocessor->cleanup();
    }
    void setPreprocessor(std::unique_ptr<PSFPreprocessor> preprocessor){
        psfpreprocessor = std::move(preprocessor);
    }
    std::shared_ptr<IBackend> prototypebackend;
    DeconvolutionProcessor processor;
    ThreadPool ioPool;
    std::unique_ptr<PSFPreprocessor> psfpreprocessor;
};



/*
This Description is the blueprint of how the specified part of the image is to be deconvolved.
*/
struct CubeTaskDescriptor {
        CubeTaskDescriptor(int taskId,
                        const BoxCoordWithPadding& paddedBox,
                        const std::shared_ptr<DeconvolutionAlgorithm>& algorithm,
                        size_t estimatedMemoryUsage,
                        const std::vector<std::shared_ptr<PSF>>& psfs,
                        const std::shared_ptr<ImageReader> reader,
                        const std::shared_ptr<ImageWriter> writer,
                        std::shared_ptr<TaskContext> context)
        : taskId(taskId),
          paddedBox(paddedBox),
          algorithm(algorithm),
          estimatedMemoryUsage(estimatedMemoryUsage),
          psfs(psfs),
          reader(reader),
          writer(writer),
          context(context)
    {}

    const int taskId;
    const BoxCoordWithPadding paddedBox;
    const std::shared_ptr<DeconvolutionAlgorithm> algorithm;
    const size_t estimatedMemoryUsage;
    const std::vector<std::shared_ptr<PSF>> psfs;
    const std::shared_ptr<ImageReader> reader;
    const std::shared_ptr<ImageWriter> writer;
    std::shared_ptr<TaskContext> context;
};




struct DeconvolutionPlan {
    Padding imagePadding;
    std::vector<std::unique_ptr<CubeTaskDescriptor>> tasks;
    size_t totalTasks;
};
class Label{
public:
    Label() = default;

    void setRange(Range<std::shared_ptr<PSF>> psfs) {this->psfs = psfs;}


    Image3D getMask(const Image3D& labelImage) const {
        return labelImage.getInRange(psfs.start, psfs.end); 
    }

    std::vector<std::shared_ptr<PSF>> getPSFs() const {
        return psfs.values;
    }

private:
    Range<std::shared_ptr<PSF>> psfs; 
    // Image3D* labelImage;
    Image3D mask;


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