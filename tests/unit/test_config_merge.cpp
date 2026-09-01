#include <gtest/gtest.h>
#include "dolphin/Config.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/ServiceAbstractions.h"
#include "dolphin/deconvolution/deconvolutionStrategies/PSFHandler.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>

using json = nlohmann::json;

class ConfigMergeTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }

    std::string writeTempJSON(const std::string& content, const std::string& filename) {
        auto path = TestUtils::outputPath() + "/" + filename;
        std::ofstream file(path);
        file << content;
        file.close();
        return path;
    }
};

TEST_F(ConfigMergeTest, SetupConfigFromSubObject) {
    auto path = writeTempJSON(TestUtils::subObjectSetupConfigJSON(), "setup_sub.json");
    auto config = SetupConfig::createFromJSONFile(path);
    EXPECT_EQ(config.backend, "cuda");
    EXPECT_EQ(config.nIOThreads, 3);
    EXPECT_EQ(config.nWorkerThreads, 7);
    EXPECT_EQ(config.nDevices, 2);
    EXPECT_EQ(config.imagePath, "sub_input.tif");
    EXPECT_EQ(config.outputPath, "sub_output.tif");
}

TEST_F(ConfigMergeTest, SetupConfigRootLevelFallback) {
    auto jsonStr = R"({
        "image_path": "root_input.tif",
        "backend": "cpu",
        "output": "root_output.tif",
        "n_io_threads": 5
    })";
    auto path = writeTempJSON(jsonStr, "setup_root.json");
    auto config = SetupConfig::createFromJSONFile(path);
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.nIOThreads, 5);
    EXPECT_EQ(config.imagePath, "root_input.tif");
    EXPECT_EQ(config.outputPath, "root_output.tif");
}

TEST_F(ConfigMergeTest, SetupConfigIgnoresPsfConfigsKey) {
    auto jsonStr = R"({
        "setup_config": {
            "image_path": "test.tif",
            "backend": "cpu"
        },
        "psf_configs": [
            {"model_name": "Gaussian", "size_x": 10, "size_y": 10, "size_z": 10}
        ]
    })";
    auto path = writeTempJSON(jsonStr, "setup_psf_erase.json");
    auto config = SetupConfig::createFromJSONFile(path);
    EXPECT_EQ(config.imagePath, "test.tif");
    EXPECT_EQ(config.backend, "cpu");
}

TEST_F(ConfigMergeTest, SetupConfigIgnoresDeconvConfigKey) {
    auto path = writeTempJSON(TestUtils::combinedSubObjectJSON(), "setup_deconv_erase.json");
    auto config = SetupConfig::createFromJSONFile(path);
    EXPECT_EQ(config.imagePath, "combined_input.tif");
    EXPECT_EQ(config.backend, "cpu");
}

TEST_F(ConfigMergeTest, DeconvConfigFromSubObject) {
    auto path = writeTempJSON(TestUtils::combinedSubObjectJSON(), "deconv_sub.json");
    auto config = DeconvolutionConfig::createFromJSONFile(path);
    EXPECT_EQ(config.algorithmName, "RichardsonLucy");
    EXPECT_EQ(config.iterations, 20);
    EXPECT_FLOAT_EQ(config.epsilon, 1e-5f);
    EXPECT_FLOAT_EQ(config.lambda, 0.02f);
}

TEST_F(ConfigMergeTest, DeconvConfigFromRootLevel) {
    auto jsonStr = R"({
        "algorithm_name": "Convolution",
        "iterations": 3,
        "epsilon": 1e-3,
        "lambda": 0.1
    })";
    auto path = writeTempJSON(jsonStr, "deconv_root.json");
    auto config = DeconvolutionConfig::createFromJSONFile(path);
    EXPECT_EQ(config.algorithmName, "Convolution");
    EXPECT_EQ(config.iterations, 3);
}

TEST_F(ConfigMergeTest, CombinedJSONBothConfigs) {
    auto path = writeTempJSON(TestUtils::combinedSubObjectJSON(), "combined.json");
    auto setupConfig = SetupConfig::createFromJSONFile(path);
    auto deconvConfig = DeconvolutionConfig::createFromJSONFile(path);

    EXPECT_EQ(setupConfig.imagePath, "combined_input.tif");
    EXPECT_EQ(setupConfig.nWorkerThreads, 4);
    EXPECT_EQ(deconvConfig.algorithmName, "RichardsonLucy");
    EXPECT_EQ(deconvConfig.iterations, 20);
}

TEST_F(ConfigMergeTest, JSONOnlyDeconvConfig) {
    auto jsonStr = R"({
        "deconvolution_config": {
            "algorithm_name": "InverseFilter",
            "iterations": 15
        }
    })";
    auto path = writeTempJSON(jsonStr, "only_deconv.json");

    auto deconvConfig = DeconvolutionConfig::createFromJSONFile(path);
    EXPECT_EQ(deconvConfig.algorithmName, "InverseFilter");
    EXPECT_EQ(deconvConfig.iterations, 15);
}

TEST_F(ConfigMergeTest, JSONOnlySetupConfig) {
    auto path = writeTempJSON(TestUtils::subObjectSetupConfigJSON(), "only_setup.json");
    auto setupConfig = SetupConfig::createFromJSONFile(path);
    EXPECT_EQ(setupConfig.backend, "cuda");
    EXPECT_EQ(setupConfig.imagePath, "sub_input.tif");
}

TEST_F(ConfigMergeTest, LoadJSONFilePublic) {
    auto path = writeTempJSON(R"({"key": "value"})", "public_load.json");
    json data = Config::loadJSONFile(path);
    EXPECT_EQ(data["key"], "value");
}

TEST_F(ConfigMergeTest, CombinedJSONWithInlinePSFSections) {
    auto path = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "combined_psf.json");

    auto setupConfig = SetupConfig::createFromJSONFile(path);
    auto deconvConfig = DeconvolutionConfig::createFromJSONFile(path);

    EXPECT_EQ(setupConfig.imagePath, "inline_input.tif");
    EXPECT_EQ(deconvConfig.iterations, 10);

    json jsonData = Config::loadJSONFile(path);
    ASSERT_TRUE(jsonData.contains("psf_configs"));
    ASSERT_EQ(jsonData["psf_configs"].size(), 1);
}

TEST_F(ConfigMergeTest, InlinePSFGaussian) {
    json jsonData = json::parse(TestUtils::combinedWithInlinePSFJSON());
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();

    auto config = factory.createConfig(jsonData["psf_configs"][0]);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "Gaussian");
    EXPECT_EQ(config->sizeX, 32);
    EXPECT_EQ(config->sizeY, 32);
    EXPECT_EQ(config->sizeZ, 16);

    auto* gaussConfig = dynamic_cast<GaussianPSFConfig*>(config.get());
    ASSERT_NE(gaussConfig, nullptr);
    EXPECT_FLOAT_EQ(gaussConfig->sigmaX, 5);
    EXPECT_EQ(config->ID, "inline_gauss");
}

TEST_F(ConfigMergeTest, InlinePSFGibsonLanni) {
    json jsonData = json::parse(TestUtils::gibsonLanniPSFConfigJSON());
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();

    auto config = factory.createConfig(jsonData);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->getModelName(), "GibsonLanni");
    EXPECT_EQ(config->sizeX, 64);
    EXPECT_FLOAT_EQ(config->NA, 1.4f);
}

TEST_F(ConfigMergeTest, InlinePSFMultiple) {
    json jsonData = json::parse(TestUtils::multiInlinePSFJSON());
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();

    std::vector<std::shared_ptr<PSFConfig>> configs;
    for (const auto& psfJson : jsonData["psf_configs"]) {
        configs.push_back(factory.createConfig(psfJson));
    }

    ASSERT_EQ(configs.size(), 2);
    EXPECT_EQ(configs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(configs[0]->ID, "psf1");
    EXPECT_EQ(configs[1]->getModelName(), "GibsonLanni");
    EXPECT_EQ(configs[1]->ID, "psf2");
}

TEST_F(ConfigMergeTest, DeconvolutionRequestInlinePSFEmpty) {
    auto setupConfig = std::make_shared<SetupConfig>();
    auto deconvConfig = std::make_shared<DeconvolutionConfig>();
    DeconvolutionRequest request(setupConfig, deconvConfig);

    EXPECT_FALSE(request.hasInlinePSFConfigs());
    EXPECT_TRUE(request.getInlinePSFConfigs().empty());
}

TEST_F(ConfigMergeTest, DeconvolutionRequestInlinePSFSetGet) {
    auto setupConfig = std::make_shared<SetupConfig>();
    auto deconvConfig = std::make_shared<DeconvolutionConfig>();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    json gaussJson = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto psfConfig = factory.createConfig(gaussJson);

    std::vector<std::shared_ptr<PSFConfig>> configs = {psfConfig};

    DeconvolutionRequest request(setupConfig, deconvConfig);
    request.setInlinePSFConfigs(configs);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 1);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "test_gaussian");
}

TEST_F(ConfigMergeTest, PSFHandlerInlineConfigsPreferredOverFilePaths) {
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    json gaussJson = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto psfConfig = factory.createConfig(gaussJson);

    auto threadPool = std::make_shared<ThreadPool>(1);
    PSFHandler psfHandler(threadPool, [](std::atomic<float>&, float){});

    psfHandler.setInlinePSFConfigs({psfConfig});
    EXPECT_TRUE(psfHandler.hasInlineConfigs());

    SetupConfig setupConfig;
    setupConfig.psfFilePaths = {};

    DeconvolutionConfig deconvConfig;
    deconvConfig.paddingStrategyType = PaddingStrategyType::NONE;

    auto paddingResult = psfHandler.getPadding(setupConfig, deconvConfig, CuboidShape{64, 64, 32});
    ASSERT_TRUE(paddingResult.success);
}

TEST_F(ConfigMergeTest, PSFHandlerDoubleLoadFix) {
    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    json gaussJson = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto psfConfig = factory.createConfig(gaussJson);

    auto threadPool = std::make_shared<ThreadPool>(1);
    PSFHandler psfHandler(threadPool, [](std::atomic<float>&, float){});

    psfHandler.setInlinePSFConfigs({psfConfig});

    SetupConfig setupConfig;
    setupConfig.psfFilePaths = {};

    DeconvolutionConfig deconvConfig;
    deconvConfig.paddingStrategyType = PaddingStrategyType::PARENT;

    auto paddingResult = psfHandler.getPadding(setupConfig, deconvConfig, CuboidShape{32, 32, 16});
    ASSERT_TRUE(paddingResult.success);

    auto shapeResult = psfHandler.getMaxShape(setupConfig, deconvConfig);
    ASSERT_TRUE(shapeResult.success);

    EXPECT_EQ(shapeResult.value.width, 32);
    EXPECT_EQ(shapeResult.value.height, 32);
    EXPECT_EQ(shapeResult.value.depth, 16);
}

TEST_F(ConfigMergeTest, PSFHandlerNoConfigsThrows) {
    auto threadPool = std::make_shared<ThreadPool>(1);
    PSFHandler psfHandler(threadPool, [](std::atomic<float>&, float){});

    SetupConfig setupConfig;
    setupConfig.psfFilePaths = {};

    DeconvolutionConfig deconvConfig;
    deconvConfig.paddingStrategyType = PaddingStrategyType::PARENT;

    auto paddingResult = psfHandler.getPadding(setupConfig, deconvConfig, CuboidShape{32, 32, 16});
    ASSERT_TRUE(paddingResult.success);

    EXPECT_THROW(psfHandler.createPSFs(CuboidShape{32, 32, 16}), std::runtime_error);
}


struct ConfigBundle {
    SetupConfig setupConfig;
    DeconvolutionConfig deconvConfig;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;

    bool hasSetup = false;
    bool hasDeconv = false;
    bool hasPSF = false;
};

static void loadJSONBundle(const json& jsonData, ConfigBundle& bundle) {
    if (jsonData.contains("setup_config")) {
        bundle.setupConfig.loadFromJSON(jsonData["setup_config"]);
        bundle.hasSetup = true;
    }

    if (jsonData.contains("deconvolution_config")) {
        bundle.deconvConfig.loadFromJSON(jsonData["deconvolution_config"]);
        bundle.hasDeconv = true;
    }

    if (jsonData.contains("psf_configs")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        for (const auto& psfJson : jsonData["psf_configs"]) {
            bundle.psfConfigs.push_back(factory.createConfig(psfJson));
        }
        bundle.hasPSF = true;
    }
}

static ConfigBundle mergeBundles(const ConfigBundle& jsonBundle, const ConfigBundle& cliBundle) {
    ConfigBundle merged;

    if (jsonBundle.hasSetup) {
        merged.setupConfig = jsonBundle.setupConfig;
    } else {
        merged.setupConfig = cliBundle.setupConfig;
    }
    merged.hasSetup = true;

    if (jsonBundle.hasDeconv) {
        merged.deconvConfig = jsonBundle.deconvConfig;
    } else {
        merged.deconvConfig = cliBundle.deconvConfig;
    }
    merged.hasDeconv = true;

    if (jsonBundle.hasPSF) {
        merged.psfConfigs = jsonBundle.psfConfigs;
        merged.hasPSF = true;
    }

    return merged;
}

static ConfigBundle makeCLIBundle(const std::string& imagePath, const std::string& backend,
                                    int iterations, const std::string& algorithm) {
    ConfigBundle cli;
    cli.setupConfig.imagePath = imagePath;
    cli.setupConfig.backend = backend;
    cli.deconvConfig.iterations = iterations;
    cli.deconvConfig.algorithmName = algorithm;
    cli.hasSetup = true;
    cli.hasDeconv = true;
    return cli;
}

TEST_F(ConfigMergeTest, CLISim_BothFromJSON_CLIOverwritten) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 99, "InverseFilter");
    ConfigBundle json;
    loadJSONBundle(json::parse(TestUtils::combinedSubObjectJSON()), json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "combined_input.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.deconvConfig.iterations, 20);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "RichardsonLucy");
}

TEST_F(ConfigMergeTest, CLISim_OnlyDeconvInJSON_SetupKeepsCLI) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 99, "InverseFilter");

    ConfigBundle json;
    json.hasSetup = false;
    json.hasDeconv = false;
    json.hasPSF = false;
    loadJSONBundle(R"({
        "deconvolution_config": {
            "algorithm_name": "InverseFilter",
            "iterations": 5
        }
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "cli_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.deconvConfig.iterations, 5);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "InverseFilter");
}

TEST_F(ConfigMergeTest, CLISim_OnlySetupInJSON_DeconvKeepsCLI) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 42, "Convolution");

    ConfigBundle json;
    loadJSONBundle(json::parse(TestUtils::subObjectSetupConfigJSON()), json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "sub_input.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.deconvConfig.iterations, 42);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "Convolution");
}

TEST_F(ConfigMergeTest, CLISim_OnlyPSFInJSON_BothConfigsKept) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 42, "Convolution");

    ConfigBundle json;
    loadJSONBundle(json::parse(TestUtils::multiInlinePSFJSON()), json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "cli_image.tif");
    EXPECT_EQ(merged.deconvConfig.iterations, 42);
    ASSERT_EQ(merged.psfConfigs.size(), 2);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(merged.psfConfigs[1]->getModelName(), "GibsonLanni");
}

TEST_F(ConfigMergeTest, CLISim_SetupAndPSFInJSON_DeconvKeepsCLI) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 77, "Convolution");

    ConfigBundle json;
    loadJSONBundle(R"({
        "setup_config": {
            "psf_file_paths": [],
            "save_psf": false,
            "output": "inline_output.tif",
            "backend": "cpu",
            "n_io_threads": 1,
            "n_worker_threads": 1,
            "n_devices": 1,
            "max_mem_gb": 1,
            "image_path": "inline_input.tif"
        },
        "psf_configs": [
            {
                "model_name": "Gaussian",
                "id": "inline_gauss",
                "res_lateral_nm": 5000,
                "res_axial_nm": 5000,
                "size_x": 32,
                "size_y": 32,
                "size_z": 16,
                "sigma_x": 5,
                "sigma_y": 5,
                "sigma_z": 5
            }
        ]
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "inline_input.tif");
    EXPECT_EQ(merged.deconvConfig.iterations, 77);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "Convolution");
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(ConfigMergeTest, CLISim_RootLevelTreatedAsSetup) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 50, "RichardsonLucy");

    ConfigBundle json;
    loadJSONBundle(R"({
        "setup_config": {
            "image_path": "root_image.tif",
            "backend": "cpu",
            "n_io_threads": 8
        }
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "root_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nIOThreads, 8);
    EXPECT_EQ(merged.deconvConfig.iterations, 50);
}

TEST_F(ConfigMergeTest, CLISim_NoConfigSections_EmptyJSON) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 50, "RichardsonLucy");

    ConfigBundle json;
    loadJSONBundle(json::object(), json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "cli_image.tif");
    EXPECT_EQ(merged.deconvConfig.iterations, 50);
    EXPECT_FALSE(merged.hasPSF);
}

TEST_F(ConfigMergeTest, CLISim_AllThreeSectionsInJSON) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 99, "InverseFilter");

    ConfigBundle json;
    loadJSONBundle(R"({
        "setup_config": {
            "image_path": "json_image.tif",
            "backend": "cpu",
            "n_io_threads": 3,
            "n_worker_threads": 5,
            "n_devices": 1,
            "max_mem_gb": 4,
            "output": "json_output.tif"
        },
        "deconvolution_config": {
            "algorithm_name": "RichardsonLucy",
            "iterations": 15,
            "epsilon": 1e-4,
            "lambda": 0.05
        },
        "psf_configs": [
            {
                "model_name": "Gaussian",
                "id": "inline1",
                "res_lateral_nm": 5000,
                "res_axial_nm": 5000,
                "size_x": 32,
                "size_y": 32,
                "size_z": 16,
                "sigma_x": 5,
                "sigma_y": 5,
                "sigma_z": 5
            }
        ]
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "json_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nIOThreads, 3);
    EXPECT_EQ(merged.deconvConfig.iterations, 15);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "RichardsonLucy");
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline1");
}

TEST_F(ConfigMergeTest, CLISim_OldFormat_RootSetupPlusDeconvSubObject) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 99, "RichardsonLucy");

    ConfigBundle json;
    loadJSONBundle(R"({
        "setup_config": {
            "image_path": "old_image.tif",
            "backend": "cpu",
            "n_io_threads": 6
        },
        "deconvolution_config": {
            "algorithm_name": "RichardsonLucy",
            "iterations": 25
        }
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "old_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nIOThreads, 6);
    EXPECT_EQ(merged.deconvConfig.iterations, 25);
}

TEST_F(ConfigMergeTest, CLISim_OldFormat_RootSetupPlusDeconvPlusPSF) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 99, "RichardsonLucy");

    ConfigBundle json;
    loadJSONBundle(R"({
        "setup_config": {
            "image_path": "old_image.tif",
            "backend": "cpu"
        },
        "deconvolution_config": {
            "algorithm_name": "RichardsonLucy",
            "iterations": 12
        },
        "psf_configs": [
            {"model_name": "Gaussian", "id": "old_psf", "size_x": 16, "size_y": 16, "size_z": 8}
        ]
    })"_json, json);

    ConfigBundle merged = mergeBundles(json, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "old_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.deconvConfig.iterations, 12);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->ID, "old_psf");
}

TEST_F(ConfigMergeTest, CLISim_NoJSON_AllFromCLI) {
    ConfigBundle cli = makeCLIBundle("cli_image.tif", "cuda", 42, "Convolution");

    ConfigBundle merged = mergeBundles(ConfigBundle{}, cli);

    EXPECT_EQ(merged.setupConfig.imagePath, "cli_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.deconvConfig.iterations, 42);
    EXPECT_EQ(merged.deconvConfig.algorithmName, "Convolution");
    EXPECT_FALSE(merged.hasPSF);
}

// --- PSFGenerationRequest inline config tests ---

TEST_F(ConfigMergeTest, PSFGenerationRequestInlinePSFEmpty) {
    auto setupConfig = std::make_shared<SetupConfigPSF>();
    PSFGenerationRequest request(setupConfig);

    EXPECT_FALSE(request.hasInlinePSFConfigs());
    EXPECT_TRUE(request.getInlinePSFConfigs().empty());
}

TEST_F(ConfigMergeTest, PSFGenerationRequestInlinePSFSetGet) {
    auto setupConfig = std::make_shared<SetupConfigPSF>();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    json gaussJson = json::parse(TestUtils::gaussianPSFConfigJSON());
    auto psfConfig = factory.createConfig(gaussJson);

    PSFGenerationRequest request(setupConfig);
    request.setInlinePSFConfigs({psfConfig});

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 1);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "test_gaussian");
}

// --- PSFConfigBundle merge simulation tests ---

struct PSFConfigBundle {
    SetupConfigPSF setupConfig;
    std::vector<std::shared_ptr<PSFConfig>> psfConfigs;

    bool hasSetup = false;
    bool hasPSF = false;
};

static void loadPSFJSONBundle(const json& jsonData, PSFConfigBundle& bundle) {
    if (jsonData.contains("setup_config")) {
        bundle.setupConfig.loadFromJSON(jsonData["setup_config"]);
        bundle.hasSetup = true;
    }

    if (jsonData.contains("psf_configs")) {
        PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
        for (const auto& psfJson : jsonData["psf_configs"]) {
            bundle.psfConfigs.push_back(factory.createConfig(psfJson));
        }
        bundle.hasPSF = true;
    }
}

static PSFConfigBundle mergePSFBundles(const PSFConfigBundle& jsonBundle, const PSFConfigBundle& cliBundle) {
    PSFConfigBundle merged;

    if (jsonBundle.hasSetup) {
        merged.setupConfig = jsonBundle.setupConfig;
    } else {
        merged.setupConfig = cliBundle.setupConfig;
    }
    merged.hasSetup = true;

    if (jsonBundle.hasPSF) {
        merged.psfConfigs = jsonBundle.psfConfigs;
        merged.hasPSF = true;
    }

    return merged;
}

static PSFConfigBundle makePSFCLIBundle(const std::string& outputPath, const std::string& backend,
                                         int nThreads) {
    PSFConfigBundle cli;
    cli.setupConfig.outputPath = outputPath;
    cli.setupConfig.backend = backend;
    cli.setupConfig.nThreads = nThreads;
    cli.hasSetup = true;
    return cli;
}

TEST_F(ConfigMergeTest, PSFSim_BothFromJSON_CLIOverwritten) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle jsonBundle;
    loadPSFJSONBundle(R"({
        "setup_config": {
            "output": "json_output.tif",
            "backend": "cpu",
            "n_threads": 4,
            "n_io_threads": 1,
            "n_worker_threads": 1,
            "n_devices": 1
        }
    })"_json, jsonBundle);

    PSFConfigBundle merged = mergePSFBundles(jsonBundle, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "json_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 4);
    EXPECT_FALSE(merged.hasPSF);
}

TEST_F(ConfigMergeTest, PSFSim_OnlyPSFInJSON_SetupKeepsCLI) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle jsonBundle;
    loadPSFJSONBundle(json::parse(TestUtils::gaussianPSFConfigJSONWrapper()), jsonBundle);

    PSFConfigBundle merged = mergePSFBundles(jsonBundle, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "cli_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.setupConfig.nThreads, 8);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(ConfigMergeTest, PSFSim_SetupAndPSFInJSON_CLIOverwritten) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle jsonBundle;
    loadPSFJSONBundle(R"({
        "setup_config": {
            "output": "json_output.tif",
            "backend": "cpu",
            "n_threads": 2,
            "n_io_threads": 3,
            "n_worker_threads": 5,
            "n_devices": 1
        },
        "psf_configs": [
            {
                "model_name": "Gaussian",
                "id": "inline_gauss",
                "res_lateral_nm": 5000,
                "res_axial_nm": 5000,
                "size_x": 32,
                "size_y": 32,
                "size_z": 16,
                "sigma_x": 5,
                "sigma_y": 5,
                "sigma_z": 5
            }
        ]
    })"_json, jsonBundle);

    PSFConfigBundle merged = mergePSFBundles(jsonBundle, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "json_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 2);
    EXPECT_EQ(merged.setupConfig.nIOThreads, 3);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(ConfigMergeTest, PSFSim_NoJSON_AllFromCLI) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle merged = mergePSFBundles(PSFConfigBundle{}, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "cli_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.setupConfig.nThreads, 8);
    EXPECT_FALSE(merged.hasPSF);
}

TEST_F(ConfigMergeTest, PSFSim_RootLevelTreatedAsSetup) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle jsonBundle;
    loadPSFJSONBundle(R"({
        "setup_config": {
            "output": "root_output.tif",
            "backend": "cpu",
            "n_threads": 6
        }
    })"_json, jsonBundle);

    PSFConfigBundle merged = mergePSFBundles(jsonBundle, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "root_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 6);
    EXPECT_FALSE(merged.hasPSF);
}

TEST_F(ConfigMergeTest, PSFSim_NoJSON_NoPSF) {
    PSFConfigBundle cli;
    cli.hasSetup = true;

    PSFConfigBundle merged = mergePSFBundles(PSFConfigBundle{}, cli);

    EXPECT_FALSE(merged.hasPSF);
}

TEST_F(ConfigMergeTest, PSFSim_RootLevelWithPSFConfigs) {
    PSFConfigBundle cli = makePSFCLIBundle("cli_output.tif", "cuda", 8);

    PSFConfigBundle jsonBundle;
    loadPSFJSONBundle(R"({
        "setup_config": {
            "output": "root_output.tif",
            "backend": "cpu"
        },
        "psf_configs": [
            {"model_name": "Gaussian", "id": "root_psf", "size_x": 16, "size_y": 16, "size_z": 8}
        ]
    })"_json, jsonBundle);

    PSFConfigBundle merged = mergePSFBundles(jsonBundle, cli);

    EXPECT_EQ(merged.setupConfig.outputPath, "root_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->ID, "root_psf");
}
