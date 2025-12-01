#pragma once
#include "deconvolution/ImageMap.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "ComputationalPlan.h"
#include <atomic>

class DeconvolutionAlgorithm;
class IBackend;

class DeconvolutionStrategy {
public:
    DeconvolutionStrategy() = default;
    virtual ~DeconvolutionStrategy() = default;
 
    virtual Hyperstack run(const Hyperstack& image, const std::vector<PSF>& psfs) = 0; 
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) = 0;

};




std::vector<BoxCoord> splitImageHomogeneous(
    const RectangleShape& subimageShape,
    const RectangleShape& imageOriginalShape);

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
        return Padding{psfSizes[0]/2, psfSizes[0]/2}; //TESTVALUE
    }
};

