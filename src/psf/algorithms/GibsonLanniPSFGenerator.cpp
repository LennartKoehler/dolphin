#include "psf/GibsonLanniPSFGenerator.h"
#include "psf/KirchoffDiffractionSimpson.h"

bool GibsonLanniPSFGenerator::hasConfig(){
    return config != nullptr;
}

void GibsonLanniPSFGenerator::setConfig(std::unique_ptr<PSFConfig> config){
    auto* ucfg = dynamic_cast<GibsonLanniPSFConfig*>(config.get());
    if (!ucfg) throw std::runtime_error("Wrong config type");
    this->config.reset(static_cast<GibsonLanniPSFConfig*>(config.release()));
}

PSF GibsonLanniPSFGenerator::generatePSF() const {
    std::vector<cv::Mat> sphereLayers;
    sphereLayers.reserve(config->sizeZ);

    for (int z = 0; z < config->sizeZ; z++){
        GibsonLanniPSFConfig config = *(this->config); // copy to enale multiprocessing
        config.ti = config.ti0 + config.resAxial_nm * 1E-9 * (z - (config.sizeZ - 1.0) / 2.0);
        config.particleAxialPosition *= 1e-9;
        sphereLayers.push_back(GibsonLanniEquation(config)); // unnecessary copy of cv::Mat
    }
    Image3D psfImage;
    psfImage.slices = sphereLayers;
    PSF psf;
    psf.image = psfImage;
    return psf;
}


cv::Mat GibsonLanniPSFGenerator::GibsonLanniEquation(const GibsonLanniPSFConfig& config) const {    
    int nx = config.sizeX;
    int ny = config.sizeY;
    int OVER_SAMPLING = config.OVER_SAMPLING;
    double NA = config.NA;
    double lambda = config.lambda;
    double resLateral = config.resLateral_nm;
    double resAxial = config.resAxial_nm;
    int accuracy = accuracy;

    
    // The center of the image in units of [pixels]
    double x0 = (nx - 1) / 2.0;
    double y0 = (ny - 1) / 2.0;
    
    // Lateral particle position in units of [pixels]
    double xp = x0; // 0.0/pixelSize;
    double yp = y0; // 0.0/pixelSize;
    
    // Calculate maximum radius
    int maxRadius = static_cast<int>(std::round(std::sqrt((nx - x0) * (nx - x0) + (ny - y0) * (ny - y0)))) + 1;
    
    std::vector<double> r(maxRadius * OVER_SAMPLING);
    std::vector<double> h(r.size());
    
    KirchhoffDiffractionSimpson I(config, accuracy, NA, lambda);
    
    for (size_t n = 0; n < r.size(); n++) {
        r[n] = static_cast<double>(n) / static_cast<double>(OVER_SAMPLING);
        h[n] = I.calculate(r[n] * resLateral * 1E-9);
    }
    
    // Linear interpolation of the pixel values
    cv::Mat slice(nx, ny, CV_32F);
    double rPixel, value;
    int index;
    
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            rPixel = std::sqrt((x - xp) * (x - xp) + (y - yp) * (y - yp));
            index = static_cast<int>(std::floor(rPixel * OVER_SAMPLING));
            
            if (index + 1 < static_cast<int>(h.size())) {
                value = h[index] + (h[index + 1] - h[index]) * (rPixel - r[index]) * OVER_SAMPLING;
            } else {
                value = h[index];
            }
            slice.at<float>(x,y) = value;
        }
    }
    
    return slice;
}
