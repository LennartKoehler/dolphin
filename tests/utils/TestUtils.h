#pragma once

#include <gtest/gtest.h>
#include <string>
#include <filesystem>
#include <random>
#include <cmath>

#include "dolphin_image/Image3D.h"
#include "dolphinbackend/CuboidShape.h"
#include "dolphin_image/Types/BoxCoord.h"

namespace TestUtils {


inline std::string outputPath() {
    auto dir = "/tmp/dolphin";
    std::filesystem::create_directories(dir);
    return dir;
}

inline bool cleanupDirectory()
{
    std::error_code ec;
    const std::uintmax_t removed =
        std::filesystem::remove_all(outputPath(), ec);

    if (ec) {
        return false;
    }

    GTEST_LOG_(INFO) << "Successfully deleted test directory";
    return true;
}

inline Image3D createConstantImage(size_t w, size_t h, size_t d, float value) {
    return Image3D(CuboidShape(w, h, d), value);
}

inline Image3D createRandomImage(size_t w, size_t h, size_t d, unsigned seed = 42) {
    Image3D img(CuboidShape(w, h, d), 0.0f);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto it = img.begin(); it != img.end(); ++it) {
        *it = dist(gen);
    }
    return img;
}


inline Image3D createImpulseImage(size_t w, size_t h, size_t d) {
    Image3D img(CuboidShape(w, h, d), 0.0f);
    img.setPixel(w / 2, h / 2, d / 2, 1.0f);
    return img;
}

inline Image3D createGradientImage(size_t w, size_t h, size_t d) {
    Image3D img(CuboidShape(w, h, d), 0.0f);
    for (size_t z = 0; z < d; z++) {
        for (size_t y = 0; y < h; y++) {
            for (size_t x = 0; x < w; x++) {
                img.setPixel(x, y, z, static_cast<float>(x + y + z));
            }
        }
    }
    return img;
}

inline bool imagesAlmostEqual(const Image3D& a, const Image3D& b, float tolerance) {
    return a.isEqual(b, tolerance);
}

inline bool hasNaN(const Image3D& img) {
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        if (std::isnan(*it)) return true;
    }
    return false;
}

inline bool hasInf(const Image3D& img) {
    for (auto it = img.cbegin(); it != img.cend(); ++it) {
        if (std::isinf(*it)) return true;
    }
    return false;
}

inline std::string gaussianPSFConfigJSON() {
    return R"({
        "model_name": "Gaussian",
        "id": "test_gaussian",
        "res_lateral_nm": 5000,
        "res_axial_nm": 5000,
        "size_x": 32,
        "size_y": 32,
        "size_z": 16,
        "sigma_x": 5,
        "sigma_y": 5,
        "sigma_z": 5
    })";
}

inline std::string gibsonLanniPSFConfigJSON() {
    return R"({
        "model_name": "GibsonLanni",
        "id": "test_gl",
        "res_lateral_nm": 2500,
        "res_axial_nm": 2500,
        "size_x": 64,
        "size_y": 64,
        "size_z": 32,
        "NA": 1.4,
        "lambda_nm": 450.0,
        "accuracy": 32,
        "working_distance_design_nm": 150000.0,
        "working_distance_experimental_nm": 150000.0,
        "immersion_ri_design": 1.515,
        "immersion_ri_experimental": 1.515,
        "coverslip_thickness_design_nm": 170.0,
        "coverslip_thickness_experimental_nm": 170.0,
        "coverslip_ri_design": 1.5,
        "coverslip_ri_experimental": 1.5,
        "sample_ri": 1.33,
        "particle_axial_position_nm": 0.0,
        "pixel_size_axial_nm": 100.0,
        "pixel_size_lateral_nm": 100.0
    })";
}

inline std::string defaultDeconvConfigJSON() {
    return R"({
        "algorithm_name": "RichardsonLucy",
        "iterations": 10,
        "epsilon": 1e-6,
        "lambda": 0.001,
        "padding_fill": "mirror",
        "padding_strategy": "parent"
    })";
}

inline std::string defaultSetupConfigJSON() {
    return R"({
        "deconvolution_config": {
            "algorithm_name": "RichardsonLucy",
            "iterations": 5,
            "epsilon": 1e-6,
            "lambda": 0.001,
            "padding_fill": "mirror"
        },
        "multiple_psf_config_paths": [],
        "psf_file_paths": [],
        "save_psf": false,
        "output": "test_output.tif",
        "backend": "cpu",
        "n_io_threads": 1,
        "n_worker_threads": 1,
        "n_devices": 1,
        "max_mem_gb": 1,
        "image_path": "test_input.tif"
    })";
}

}
