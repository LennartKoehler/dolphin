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

#include <array>
#include <optional>
#include <string>
#include "dolphin_image/HelperClasses.h"
#include "dolphin/Config.h"
#include "dolphin/deconvolution/AlgorithmNames.h"
#include "dolphin_image/Types/PaddingFillType.h"

// PaddingFillType is now defined in dolphin_image/Types/PaddingFillType.h
// It is still available here for backward compatibility.

// Preset PSF padding thresholds, expressed as a fraction of the PSF max value.
// Setting cube_padding (all values >= 0) or padding_relative_max (> 0)
// overrides the preset; see PSFHandler::getPadding for resolution order.
enum PaddingStrategyType {
    DEFAULT, // to track if it was set
    NONE,
    LOW,
    MEDIUM,
    MAX
};

// How the flat background level is obtained. The background is a property of
// the whole image; estimating it per subimage makes neighbouring cubes disagree
// and leaves visible tile seams.
enum BackgroundStrategyType {
    // One background for the whole run, measured once on ONE representative
    // cube (an interior cube, so no image-border padding skews it) and passed
    // to every cube (default). Cost: one extra cube read at plan time.
    BACKGROUND_GLOBAL = 0,
    // Streams the complete image through the reader before deconvolving.
    // Most robust (no single-tile bias) but reads the image twice.
    BACKGROUND_WHOLE_IMAGE = 1,
};

/**
 * @brief Parameters shared by every deconvolution algorithm.
 *
 * This is a base class: each algorithm owns a config class derived from it
 * that registers its own parameters on top of these (see
 * deconvolution/configs/*). The base deliberately holds only what more than one
 * algorithm needs or what the surrounding pipeline (PSF generation, padding,
 * subimage splitting) consumes.
 *
 * Copy semantics: `Config::parameters` stores raw `void*` into this object's
 * members, so copies must re-register their parameters. Derived classes copy the
 * base fields themselves and then call registerAllParameters() (see the pattern
 * in the derived configs).
 */
class DeconvolutionConfig : public Config{
public:
    DeconvolutionConfig(const DeconvolutionConfig& other);
    DeconvolutionConfig& operator=(const DeconvolutionConfig& other);
    DeconvolutionConfig(DeconvolutionConfig&& other) noexcept;
    DeconvolutionConfig& operator=(DeconvolutionConfig&& other) noexcept;

    std::string getName() const override { return std::string("DeconvolutionConfig") + std::string(" ") + algorithmName; };

    // Load a config from JSON: the algorithm is selected by the "algorithm_name"
    // field (default: RichardsonLucy) and the matching config class is created.
    // Throws if the algorithm is unknown or its parameters are invalid.
    static std::shared_ptr<DeconvolutionConfig> createFromJSONFile(const std::string& filePath);
    static std::shared_ptr<DeconvolutionConfig> createFromJSON(const json& jsonData);

    // Parameters shared by all algorithms
    std::string algorithmName = DeconvolutionAlgorithms::kRichardsonLucy;
    int iterations = 10;
    float minDenominator = 1e-6f;
    float signalToNoise = 10.0f;
    PaddingFillType paddingFillType = PaddingFillType::ZERO;
    PaddingStrategyType paddingStrategyType = PaddingStrategyType::DEFAULT;
    float paddingRelativeMax = 0.0f; // > 0: fraction of PSF max to pad to; 0 = use preset
    int featheringRadius = 0;
    std::array<size_t, 3> cubeSize{0, 0, 0}; // currently unused
    std::optional<std::array<size_t, 3>> cubePadding = std::nullopt;
    bool sliceAllDimensions = false;
    // Where the flat background comes from. GLOBAL measures it once on one
    // representative (interior) cube and passes it to every cube, so all cubes
    // agree (no seams) without reading the whole image; TILE keeps the legacy
    // per-subimage estimate.
    BackgroundStrategyType backgroundStrategyType = BACKGROUND_GLOBAL;

protected:
    // Not constructible directly: use the factory / createFromJSON.
    DeconvolutionConfig();
    ~DeconvolutionConfig() = default;

    // Copies the shared fields from `other`. The caller re-registers afterwards:
    // the copied `parameters` cannot be reused because their void* would dangle.
    void assignBase(const DeconvolutionConfig& other);

    // Registers the shared parameters. Starts from an empty list, so a derived
    // override calls this first and then appends its own parameters; the factory
    // view therefore always shows exactly the selected algorithm's parameters.
    virtual void registerAllParameters();

private:
    void copyBaseFields(const DeconvolutionConfig& other);
    void moveBaseFields(DeconvolutionConfig& other);
};
