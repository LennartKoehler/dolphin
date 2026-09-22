#include <gtest/gtest.h>
#include "CLIFrontend.h"
#include "dolphin/Config.h"
#include "dolphin/SetupConfig.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/configs/PSFConfig.h"
#include "dolphin/psf/configs/GaussianPSFConfig.h"
#include "dolphin/psf/configs/GibsonLanniPSFConfig.h"
#include "dolphin/psf/PSFGeneratorFactory.h"
#include "dolphin/ServiceAbstractions.h"
#include "dolphin/Logging.h"
#include "TestUtils.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <filesystem>

using json = nlohmann::json;

class TestableCLIFrontend : public CLIFrontend {
public:
    using CLIFrontend::CLIFrontend;

    using CLIFrontend::mergeBundles;
    using CLIFrontend::mergePSFBundles;
    using CLIFrontend::generateDeconvRequest;
    using CLIFrontend::generatePSFRequest;
    using CLIFrontend::handleDeconvolution;
    using CLIFrontend::handlePSFGeneration;
    using CLIFrontend::loadJSONBundle;
    using CLIFrontend::loadPSFJSONBundle;
    using CLIFrontend::loadSetupConfigFromFile;
    using CLIFrontend::loadDeconvConfigFromFile;
    using CLIFrontend::loadPSFConfigsFromFile;
    using CLIFrontend::loadSetupConfigFromJSON;
    using CLIFrontend::loadDeconvConfigFromJSON;
    using CLIFrontend::loadPSFConfigsFromJSON;

    ConfigBundle& jsonBundleRef() { return jsonBundle; }
    ConfigBundle& cliBundleRef() { return cliBundle; }
    PSFConfigBundle& jsonPsfBundleRef() { return jsonPsfBundle; }
    PSFConfigBundle& cliPsfBundleRef() { return cliPsfBundle; }
};

class CLIFrontendTest : public ::testing::Test {
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

    TestableCLIFrontend makeFrontend() {
        return TestableCLIFrontend(nullptr, 0, nullptr);
    }
};

// =============================================================
// Deconvolution: mergeBundles
// =============================================================

TEST_F(CLIFrontendTest, MergeBundles_JSONSetupWinsOverCLI) {
    auto fe = makeFrontend();

    ConfigBundle jsonBundle;
    jsonBundle.setupConfig.imagePath = "json_image.tif";
    jsonBundle.setupConfig.backend = "cpu";
    jsonBundle.setupConfig.nThreads = 4;
    jsonBundle.hasSetup = true;

    ConfigBundle cliBundle;
    cliBundle.setupConfig.imagePath = "cli_image.tif";
    cliBundle.setupConfig.backend = "cuda";
    cliBundle.setupConfig.nThreads = 8;
    cliBundle.hasSetup = true;

    ConfigBundle merged = fe.mergeBundles(jsonBundle, cliBundle);

    EXPECT_EQ(merged.setupConfig.imagePath, "json_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 4);
}

TEST_F(CLIFrontendTest, MergeBundles_NoJSON_CLIUsed) {
    auto fe = makeFrontend();

    ConfigBundle cliBundle;
    cliBundle.setupConfig.imagePath = "cli_image.tif";
    cliBundle.setupConfig.backend = "cuda";
    cliBundle.setupConfig.nThreads = 8;
    cliBundle.hasSetup = true;

    ConfigBundle merged = fe.mergeBundles(ConfigBundle{}, cliBundle);

    EXPECT_EQ(merged.setupConfig.imagePath, "cli_image.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.setupConfig.nThreads, 8);
}

TEST_F(CLIFrontendTest, MergeBundles_JSONDeconvWinsOverCLI) {
    auto fe = makeFrontend();

    ConfigBundle jsonBundle;
    jsonBundle.deconvConfig.algorithmName = "RichardsonLucy";
    jsonBundle.deconvConfig.iterations = 20;
    jsonBundle.hasDeconv = true;

    ConfigBundle cliBundle;
    cliBundle.deconvConfig.algorithmName = "Convolution";
    cliBundle.deconvConfig.iterations = 5;
    cliBundle.hasDeconv = true;

    ConfigBundle merged = fe.mergeBundles(jsonBundle, cliBundle);

    EXPECT_EQ(merged.deconvConfig.algorithmName, "RichardsonLucy");
    EXPECT_EQ(merged.deconvConfig.iterations, 20);
}

TEST_F(CLIFrontendTest, MergeBundles_PSFOnlyFromJSON) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto psfConfig = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));

    ConfigBundle jsonBundle;
    jsonBundle.psfConfigs = {psfConfig};
    jsonBundle.hasPSF = true;
    jsonBundle.hasSetup = true;

    ConfigBundle cliBundle;
    cliBundle.hasSetup = true;

    ConfigBundle merged = fe.mergeBundles(jsonBundle, cliBundle);

    EXPECT_TRUE(merged.hasPSF);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
}

TEST_F(CLIFrontendTest, MergeBundles_NoPSFInJSON_NoPSFInMerged) {
    auto fe = makeFrontend();

    ConfigBundle jsonBundle;
    jsonBundle.hasSetup = true;

    ConfigBundle cliBundle;
    cliBundle.hasSetup = true;

    ConfigBundle merged = fe.mergeBundles(jsonBundle, cliBundle);

    EXPECT_FALSE(merged.hasPSF);
    EXPECT_TRUE(merged.psfConfigs.empty());
}

// =============================================================
// Deconvolution: generateDeconvRequest
// =============================================================

TEST_F(CLIFrontendTest, GenerateDeconvRequest_WiresAllConfigs) {
    auto fe = makeFrontend();

    ConfigBundle bundle;
    bundle.setupConfig.imagePath = "test_image.tif";
    bundle.setupConfig.backend = "cpu";
    bundle.setupConfig.nThreads = 4;
    bundle.setupConfig.outputPath = "test_output.tif";
    bundle.hasSetup = true;

    bundle.deconvConfig.algorithmName = "RichardsonLucy";
    bundle.deconvConfig.iterations = 10;
    bundle.hasDeconv = true;

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto psfConfig = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));
    bundle.psfConfigs = {psfConfig};
    bundle.hasPSF = true;

    DeconvolutionRequest request = fe.generateDeconvRequest(bundle);

    EXPECT_EQ(request.getConfig()->imagePath, "test_image.tif");
    EXPECT_EQ(request.getConfig()->backend, "cpu");
    EXPECT_EQ(request.getConfig()->nThreads, 4);
    EXPECT_EQ(request.getConfig()->outputPath, "test_output.tif");

    EXPECT_EQ(request.getDeconvolutionConfig()->algorithmName, "RichardsonLucy");
    EXPECT_EQ(request.getDeconvolutionConfig()->iterations, 10);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 1);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "test_gaussian");
}

TEST_F(CLIFrontendTest, GenerateDeconvRequest_NoPSFConfigs) {
    auto fe = makeFrontend();

    ConfigBundle bundle;
    bundle.setupConfig.imagePath = "test_image.tif";
    bundle.hasSetup = true;
    bundle.hasDeconv = true;

    DeconvolutionRequest request = fe.generateDeconvRequest(bundle);

    EXPECT_FALSE(request.hasInlinePSFConfigs());
    EXPECT_TRUE(request.getInlinePSFConfigs().empty());
}

TEST_F(CLIFrontendTest, GenerateDeconvRequest_SetupConfigValues) {
    auto fe = makeFrontend();

    ConfigBundle bundle;
    bundle.setupConfig.nThreads = 12;
    bundle.setupConfig.nIOThreads = 3;
    bundle.setupConfig.nWorkerThreads = 7;
    bundle.setupConfig.backend = "cuda";
    bundle.hasSetup = true;
    bundle.hasDeconv = true;

    DeconvolutionRequest request = fe.generateDeconvRequest(bundle);

    EXPECT_EQ(request.getConfig()->nThreads, 12);
    EXPECT_EQ(request.getConfig()->nIOThreads, 3);
    EXPECT_EQ(request.getConfig()->nWorkerThreads, 7);
    EXPECT_EQ(request.getConfig()->backend, "cuda");
}

TEST_F(CLIFrontendTest, GenerateDeconvRequest_MultipleInlinePSFs) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto gauss = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));
    auto gl = factory.createConfig(json::parse(TestUtils::gibsonLanniPSFConfigJSON()));

    ConfigBundle bundle;
    bundle.psfConfigs = {gauss, gl};
    bundle.hasPSF = true;
    bundle.hasSetup = true;
    bundle.hasDeconv = true;

    DeconvolutionRequest request = fe.generateDeconvRequest(bundle);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 2);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[1]->getModelName(), "GibsonLanni");
}

// =============================================================
// Deconvolution: handleDeconvolution
// =============================================================

TEST_F(CLIFrontendTest, HandleDeconvolution_MissingRequiredFails) {
    auto fe = makeFrontend();

    ConfigBundle bundle;
    bundle.hasSetup = true;
    bundle.hasDeconv = true;

    bool result = fe.handleDeconvolution(bundle);

    EXPECT_FALSE(result);
}

TEST_F(CLIFrontendTest, HandleDeconvolution_WithRequiredPasses) {
    auto fe = makeFrontend();

    ConfigBundle bundle;
    bundle.setupConfig.imagePath = "test_image.tif";
    bundle.setupConfig.outputPath = "test_output.tif";
    bundle.setupConfig.psfFilePaths = {"test_psf.tif"};
    bundle.hasSetup = true;
    bundle.hasDeconv = true;

    bool result = fe.handleDeconvolution(bundle);

    EXPECT_TRUE(result);
}

// =============================================================
// Deconvolution: loadJSONBundle
// =============================================================

TEST_F(CLIFrontendTest, LoadJSONBundle_SubObjectFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "deconv_sub.json");
    fe.loadJSONBundle(path);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_TRUE(jsonBundle.hasSetup);
    EXPECT_TRUE(jsonBundle.hasDeconv);
    EXPECT_TRUE(jsonBundle.hasPSF);

    EXPECT_EQ(jsonBundle.setupConfig.imagePath, "inline_input.tif");
    EXPECT_EQ(jsonBundle.setupConfig.backend, "cpu");
    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "RichardsonLucy");
    EXPECT_EQ(jsonBundle.deconvConfig.iterations, 10);
    ASSERT_EQ(jsonBundle.psfConfigs.size(), 1);
    EXPECT_EQ(jsonBundle.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(jsonBundle.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(CLIFrontendTest, LoadJSONBundle_SetupInDeconvSubObject) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
        "setup_config": {
            "image_path": "root_image.tif",
            "backend": "cpu",
            "n_io_threads": 8,
            "output": "root_output.tif"
        },
        "deconvolution_config": {
            "algorithm_name": "RichardsonLucy",
            "iterations": 25
        }
    })";
    auto path = writeTempJSON(jsonStr, "deconv_root.json");
    fe.loadJSONBundle(path);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_TRUE(jsonBundle.hasSetup);
    EXPECT_TRUE(jsonBundle.hasDeconv);
    EXPECT_FALSE(jsonBundle.hasPSF);

    EXPECT_EQ(jsonBundle.setupConfig.imagePath, "root_image.tif");
    EXPECT_EQ(jsonBundle.setupConfig.backend, "cpu");
    EXPECT_EQ(jsonBundle.setupConfig.nIOThreads, 8);
    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "RichardsonLucy");
    EXPECT_EQ(jsonBundle.deconvConfig.iterations, 25);
}

TEST_F(CLIFrontendTest, LoadJSONBundle_OnlyDeconvConfig) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
        "deconvolution_config": {
            "algorithm_name": "InverseFilter",
            "iterations": 5
        }
    })";
    auto path = writeTempJSON(jsonStr, "only_deconv.json");
    fe.loadJSONBundle(path);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_FALSE(jsonBundle.hasSetup);
    EXPECT_TRUE(jsonBundle.hasDeconv);
    EXPECT_FALSE(jsonBundle.hasPSF);

    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "InverseFilter");
    EXPECT_EQ(jsonBundle.deconvConfig.iterations, 5);
}

TEST_F(CLIFrontendTest, LoadJSONBundle_EmptyJSON) {
    auto fe = makeFrontend();

    auto path = writeTempJSON("{}", "empty.json");
    fe.loadJSONBundle(path);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_FALSE(jsonBundle.hasSetup);
    EXPECT_FALSE(jsonBundle.hasDeconv);
    EXPECT_FALSE(jsonBundle.hasPSF);
}

// =============================================================
// Deconvolution: loadJSONBundle + mergeBundles integration
// =============================================================

TEST_F(CLIFrontendTest, LoadAndMerge_JSONSetupDeconvPSF_CLIOverwritten) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "combined.json");
    fe.loadJSONBundle(path);

    fe.cliBundleRef().setupConfig.imagePath = "cli_image.tif";
    fe.cliBundleRef().setupConfig.backend = "cuda";
    fe.cliBundleRef().deconvConfig.iterations = 99;
    fe.cliBundleRef().hasSetup = true;
    fe.cliBundleRef().hasDeconv = true;

    ConfigBundle merged = fe.mergeBundles(fe.jsonBundleRef(), fe.cliBundleRef());

    EXPECT_EQ(merged.setupConfig.imagePath, "inline_input.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.deconvConfig.iterations, 10);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline_gauss");
}

// =============================================================
// PSF Generation: mergePSFBundles
// =============================================================

TEST_F(CLIFrontendTest, MergePSFBundles_JSONSetupWinsOverCLI) {
    auto fe = makeFrontend();

    PSFConfigBundle jsonBundle;
    jsonBundle.setupConfig.outputPath = "json_output.tif";
    jsonBundle.setupConfig.backend = "cpu";
    jsonBundle.setupConfig.nThreads = 4;
    jsonBundle.hasSetup = true;

    PSFConfigBundle cliBundle;
    cliBundle.setupConfig.outputPath = "cli_output.tif";
    cliBundle.setupConfig.backend = "cuda";
    cliBundle.setupConfig.nThreads = 8;
    cliBundle.hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(jsonBundle, cliBundle);

    EXPECT_EQ(merged.setupConfig.outputPath, "json_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 4);
}

TEST_F(CLIFrontendTest, MergePSFBundles_NoJSON_CLIUsed) {
    auto fe = makeFrontend();

    PSFConfigBundle cliBundle;
    cliBundle.setupConfig.outputPath = "cli_output.tif";
    cliBundle.setupConfig.backend = "cuda";
    cliBundle.setupConfig.nThreads = 8;
    cliBundle.hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(PSFConfigBundle{}, cliBundle);

    EXPECT_EQ(merged.setupConfig.outputPath, "cli_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.setupConfig.nThreads, 8);
}

TEST_F(CLIFrontendTest, MergePSFBundles_PSFOnlyFromJSON) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto psfConfig = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));

    PSFConfigBundle jsonBundle;
    jsonBundle.psfConfigs = {psfConfig};
    jsonBundle.hasPSF = true;
    jsonBundle.hasSetup = true;

    PSFConfigBundle cliBundle;
    cliBundle.hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(jsonBundle, cliBundle);

    EXPECT_TRUE(merged.hasPSF);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
}

TEST_F(CLIFrontendTest, MergePSFBundles_NoPSFInJSON_NoPSFInMerged) {
    auto fe = makeFrontend();

    PSFConfigBundle jsonBundle;
    jsonBundle.hasSetup = true;

    PSFConfigBundle cliBundle;
    cliBundle.hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(jsonBundle, cliBundle);

    EXPECT_FALSE(merged.hasPSF);
    EXPECT_TRUE(merged.psfConfigs.empty());
}

// =============================================================
// PSF Generation: generatePSFRequest
// =============================================================

TEST_F(CLIFrontendTest, GeneratePSFRequest_WiresSetupAndInlinePSF) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto psfConfig = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));

    PSFConfigBundle bundle;
    bundle.setupConfig.outputPath = "psf_output.tif";
    bundle.setupConfig.backend = "cpu";
    bundle.setupConfig.nThreads = 4;
    bundle.psfConfigs = {psfConfig};
    bundle.hasPSF = true;
    bundle.hasSetup = true;

    PSFGenerationRequest request = fe.generatePSFRequest(bundle);

    EXPECT_EQ(request.getConfig()->outputPath, "psf_output.tif");
    EXPECT_EQ(request.getConfig()->backend, "cpu");
    EXPECT_EQ(request.getConfig()->nThreads, 4);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 1);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "test_gaussian");
}

TEST_F(CLIFrontendTest, GeneratePSFRequest_NoPSFConfigs) {
    auto fe = makeFrontend();

    PSFConfigBundle bundle;
    bundle.setupConfig.outputPath = "psf_output.tif";
    bundle.hasSetup = true;

    PSFGenerationRequest request = fe.generatePSFRequest(bundle);

    EXPECT_FALSE(request.hasInlinePSFConfigs());
    EXPECT_TRUE(request.getInlinePSFConfigs().empty());
}

TEST_F(CLIFrontendTest, GeneratePSFRequest_ThreadCountFromSetup) {
    auto fe = makeFrontend();

    PSFConfigBundle bundle;
    bundle.setupConfig.nThreads = 16;
    bundle.setupConfig.nIOThreads = 2;
    bundle.setupConfig.nWorkerThreads = 8;
    bundle.hasSetup = true;

    PSFGenerationRequest request = fe.generatePSFRequest(bundle);

    EXPECT_EQ(request.getConfig()->nThreads, 16);
    EXPECT_EQ(request.getConfig()->nIOThreads, 2);
    EXPECT_EQ(request.getConfig()->nWorkerThreads, 8);
}

TEST_F(CLIFrontendTest, GeneratePSFRequest_MultipleInlinePSFs) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto gauss = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));
    auto gl = factory.createConfig(json::parse(TestUtils::gibsonLanniPSFConfigJSON()));

    PSFConfigBundle bundle;
    bundle.psfConfigs = {gauss, gl};
    bundle.hasPSF = true;
    bundle.hasSetup = true;

    PSFGenerationRequest request = fe.generatePSFRequest(bundle);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 2);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "Gaussian");
    EXPECT_EQ(request.getInlinePSFConfigs()[1]->getModelName(), "GibsonLanni");
}

// =============================================================
// PSF Generation: handlePSFGeneration
// =============================================================

TEST_F(CLIFrontendTest, HandlePSFGeneration_WithInlinePSF) {
    auto fe = makeFrontend();

    PSFGeneratorFactory factory = PSFGeneratorFactory::getInstance();
    auto psfConfig = factory.createConfig(json::parse(TestUtils::gaussianPSFConfigJSON()));

    PSFConfigBundle bundle;
    bundle.setupConfig.outputPath = "psf_output.tif";
    bundle.psfConfigs = {psfConfig};
    bundle.hasPSF = true;
    bundle.hasSetup = true;

    bool result = fe.handlePSFGeneration(bundle);

    EXPECT_TRUE(result);
}

TEST_F(CLIFrontendTest, HandlePSFGeneration_NeitherFails) {
    auto fe = makeFrontend();

    PSFConfigBundle bundle;
    bundle.setupConfig.outputPath = "psf_output.tif";
    bundle.hasSetup = true;

    bool result = fe.handlePSFGeneration(bundle);

    EXPECT_FALSE(result);
}

TEST_F(CLIFrontendTest, HandlePSFGeneration_MissingOutputFails) {
    auto fe = makeFrontend();

    PSFConfigBundle bundle;
    bundle.hasSetup = true;

    bool result = fe.handlePSFGeneration(bundle);

    EXPECT_FALSE(result);
}

// =============================================================
// PSF Generation: loadPSFJSONBundle
// =============================================================

TEST_F(CLIFrontendTest, LoadPSFJSONBundle_SubObjectFormat) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
        "setup_config": {
            "output": "psf_output.tif",
            "backend": "cpu",
            "n_threads": 4,
            "n_io_threads": 1,
            "n_worker_threads": 1,
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
    })";
    auto path = writeTempJSON(jsonStr, "psf_sub.json");
    fe.loadPSFJSONBundle(path);

    auto& bundle = fe.jsonPsfBundleRef();
    EXPECT_TRUE(bundle.hasSetup);
    EXPECT_TRUE(bundle.hasPSF);

    EXPECT_EQ(bundle.setupConfig.outputPath, "psf_output.tif");
    EXPECT_EQ(bundle.setupConfig.backend, "cpu");
    EXPECT_EQ(bundle.setupConfig.nThreads, 4);
    ASSERT_EQ(bundle.psfConfigs.size(), 1);
    EXPECT_EQ(bundle.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(bundle.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(CLIFrontendTest, LoadPSFJSONBundle_RootLevelSetup) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
        "setup_config": {
            "output": "root_output.tif",
            "backend": "cpu",
            "n_threads": 6
        },
        "psf_configs": [
            {"model_name": "Gaussian", "id": "root_psf", "size_x": 16, "size_y": 16, "size_z": 8}
        ]
    })";
    auto path = writeTempJSON(jsonStr, "psf_root.json");
    fe.loadPSFJSONBundle(path);

    auto& bundle = fe.jsonPsfBundleRef();
    EXPECT_TRUE(bundle.hasSetup);
    EXPECT_TRUE(bundle.hasPSF);

    EXPECT_EQ(bundle.setupConfig.outputPath, "root_output.tif");
    EXPECT_EQ(bundle.setupConfig.backend, "cpu");
    EXPECT_EQ(bundle.setupConfig.nThreads, 6);
    ASSERT_EQ(bundle.psfConfigs.size(), 1);
    EXPECT_EQ(bundle.psfConfigs[0]->ID, "root_psf");
}

TEST_F(CLIFrontendTest, LoadPSFJSONBundle_OnlyPSFConfigs) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::gaussianPSFConfigJSONWrapper(), "psf_only.json");
    fe.loadPSFJSONBundle(path);

    auto& bundle = fe.jsonPsfBundleRef();
    EXPECT_FALSE(bundle.hasSetup);
    EXPECT_TRUE(bundle.hasPSF);

    ASSERT_EQ(bundle.psfConfigs.size(), 1);
    EXPECT_EQ(bundle.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(bundle.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(CLIFrontendTest, LoadPSFJSONBundle_EmptyJSON) {
    auto fe = makeFrontend();

    auto path = writeTempJSON("{}", "psf_empty.json");
    fe.loadPSFJSONBundle(path);

    auto& bundle = fe.jsonPsfBundleRef();
    EXPECT_FALSE(bundle.hasSetup);
    EXPECT_FALSE(bundle.hasPSF);
}

// =============================================================
// PSF Generation: loadPSFJSONBundle + mergePSFBundles integration
// =============================================================

TEST_F(CLIFrontendTest, LoadAndMergePSF_JSONSetupAndPSF_CLIOverwritten) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
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
    })";
    auto path = writeTempJSON(jsonStr, "psf_combined.json");
    fe.loadPSFJSONBundle(path);

    fe.cliPsfBundleRef().setupConfig.outputPath = "cli_output.tif";
    fe.cliPsfBundleRef().setupConfig.backend = "cuda";
    fe.cliPsfBundleRef().setupConfig.nThreads = 8;
    fe.cliPsfBundleRef().hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(fe.jsonPsfBundleRef(), fe.cliPsfBundleRef());

    EXPECT_EQ(merged.setupConfig.outputPath, "json_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cpu");
    EXPECT_EQ(merged.setupConfig.nThreads, 2);
    EXPECT_EQ(merged.setupConfig.nIOThreads, 3);
    ASSERT_EQ(merged.psfConfigs.size(), 1);
    EXPECT_EQ(merged.psfConfigs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(merged.psfConfigs[0]->ID, "inline_gauss");
}

TEST_F(CLIFrontendTest, LoadAndMergePSF_NoJSON_CLIUsed) {
    auto fe = makeFrontend();

    fe.cliPsfBundleRef().setupConfig.outputPath = "cli_output.tif";
    fe.cliPsfBundleRef().setupConfig.backend = "cuda";
    fe.cliPsfBundleRef().setupConfig.nThreads = 8;
    fe.cliPsfBundleRef().hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(fe.jsonPsfBundleRef(), fe.cliPsfBundleRef());

    EXPECT_EQ(merged.setupConfig.outputPath, "cli_output.tif");
    EXPECT_EQ(merged.setupConfig.backend, "cuda");
    EXPECT_EQ(merged.setupConfig.nThreads, 8);
    EXPECT_FALSE(merged.hasPSF);
}

// =============================================================
// PSF Generation: loadPSFJSONBundle + generatePSFRequest integration
// =============================================================

TEST_F(CLIFrontendTest, LoadAndGeneratePSF_InlinePSFFromJSON) {
    auto fe = makeFrontend();

    auto jsonStr = R"({
        "setup_config": {
            "output": "psf_output.tif",
            "backend": "cpu",
            "n_threads": 6,
            "n_io_threads": 1,
            "n_worker_threads": 1,
            "n_devices": 1
        },
        "psf_configs": [
            {
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
            }
        ]
    })";
    auto path = writeTempJSON(jsonStr, "psf_gen.json");
    fe.loadPSFJSONBundle(path);

    fe.cliPsfBundleRef().hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(fe.jsonPsfBundleRef(), fe.cliPsfBundleRef());

    PSFGenerationRequest request = fe.generatePSFRequest(merged);

    EXPECT_EQ(request.getConfig()->outputPath, "psf_output.tif");
    EXPECT_EQ(request.getConfig()->backend, "cpu");
    EXPECT_EQ(request.getConfig()->nThreads, 6);

    EXPECT_TRUE(request.hasInlinePSFConfigs());
    ASSERT_EQ(request.getInlinePSFConfigs().size(), 1);
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->getModelName(), "GibsonLanni");
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "test_gl");
}

// =============================================================
// Separate config file loading: loadSetupConfigFromFile
// =============================================================

TEST_F(CLIFrontendTest, LoadSetupConfigFile_RootLevelFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::standaloneSetupConfigJSON(), "setup_root.json");
    SetupConfigPSF config;
    fe.loadSetupConfigFromFile(path, config);

    EXPECT_EQ(config.outputPath, "standalone_output.tif");
    EXPECT_EQ(config.backend, "cuda");
    EXPECT_EQ(config.nIOThreads, 2);
    EXPECT_EQ(config.nWorkerThreads, 4);
}

TEST_F(CLIFrontendTest, LoadSetupConfigFile_SubObjectFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::setupConfigSubObjectJSON(), "setup_wrapped.json");
    SetupConfigPSF config;
    fe.loadSetupConfigFromFile(path, config);

    EXPECT_EQ(config.outputPath, "wrapped_output.tif");
    EXPECT_EQ(config.backend, "cpu");
    EXPECT_EQ(config.nIOThreads, 1);
}

// =============================================================
// Separate config file loading: loadDeconvConfigFromFile
// =============================================================

TEST_F(CLIFrontendTest, LoadDeconvConfigFile_RootLevelFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::standaloneDeconvConfigJSON(), "deconv_root.json");
    DeconvolutionConfig config;
    fe.loadDeconvConfigFromFile(path, config);

    EXPECT_EQ(config.algorithmName, "RegularizedInverseFilter");
    EXPECT_EQ(config.iterations, 30);
    EXPECT_FLOAT_EQ(config.epsilon, 1e-8f);
}

TEST_F(CLIFrontendTest, LoadDeconvConfigFile_SubObjectFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::deconvConfigSubObjectJSON(), "deconv_wrapped.json");
    DeconvolutionConfig config;
    fe.loadDeconvConfigFromFile(path, config);

    EXPECT_EQ(config.algorithmName, "Convolution");
    EXPECT_EQ(config.iterations, 1);
}

// =============================================================
// Separate config file loading: loadPSFConfigsFromFile
// =============================================================

TEST_F(CLIFrontendTest, LoadPSFConfigFile_SingleConfig) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::standaloneSinglePSFConfigJSON(), "psf_single.json");
    auto configs = fe.loadPSFConfigsFromFile(path);

    ASSERT_EQ(configs.size(), 1);
    EXPECT_EQ(configs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(configs[0]->ID, "standalone_gauss");
}

TEST_F(CLIFrontendTest, LoadPSFConfigFile_ArrayFormat) {
    auto fe = makeFrontend();

    auto path = writeTempJSON(TestUtils::standaloneArrayPSFConfigJSON(), "psf_array.json");
    auto configs = fe.loadPSFConfigsFromFile(path);

    ASSERT_EQ(configs.size(), 1);
    EXPECT_EQ(configs[0]->getModelName(), "Gaussian");
    EXPECT_EQ(configs[0]->ID, "standalone_gauss");
}

TEST_F(CLIFrontendTest, LoadPSFConfigFile_MultiFile) {
    auto fe = makeFrontend();

    auto path1 = writeTempJSON(TestUtils::standaloneSinglePSFConfigJSON(), "psf_multi1.json");
    auto path2 = writeTempJSON(TestUtils::gaussianPSFConfigJSON(), "psf_multi2.json");

    auto configs1 = fe.loadPSFConfigsFromFile(path1);
    auto configs2 = fe.loadPSFConfigsFromFile(path2);

    EXPECT_EQ(configs1.size(), 1);
    EXPECT_EQ(configs2.size(), 1);
    EXPECT_EQ(configs1[0]->ID, "standalone_gauss");
    EXPECT_EQ(configs2[0]->ID, "test_gaussian");
}

// =============================================================
// Separate configs override combined config (-c)
// =============================================================

TEST_F(CLIFrontendTest, SeparateConfigs_SetupOverridesCombined) {
    auto fe = makeFrontend();

    auto combinedPath = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "combined_override.json");
    fe.loadJSONBundle(combinedPath);

    auto setupPath = writeTempJSON(TestUtils::standaloneSetupConfigJSON(), "setup_override.json");
    fe.loadSetupConfigFromFile(setupPath, fe.jsonBundleRef().setupConfig);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_EQ(jsonBundle.setupConfig.imagePath, "standalone_input.tif");
    EXPECT_EQ(jsonBundle.setupConfig.backend, "cuda");
    EXPECT_EQ(jsonBundle.setupConfig.outputPath, "standalone_output.tif");
    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "RichardsonLucy");
    EXPECT_TRUE(jsonBundle.hasPSF);
}

TEST_F(CLIFrontendTest, SeparateConfigs_DeconvOverridesCombined) {
    auto fe = makeFrontend();

    auto combinedPath = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "combined_deconv.json");
    fe.loadJSONBundle(combinedPath);

    auto deconvPath = writeTempJSON(TestUtils::standaloneDeconvConfigJSON(), "deconv_override.json");
    fe.loadDeconvConfigFromFile(deconvPath, fe.jsonBundleRef().deconvConfig);

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_EQ(jsonBundle.setupConfig.imagePath, "inline_input.tif");
    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "RegularizedInverseFilter");
    EXPECT_EQ(jsonBundle.deconvConfig.iterations, 30);
}

TEST_F(CLIFrontendTest, SeparateConfigs_PSFAdditiveWithCombined) {
    auto fe = makeFrontend();

    auto combinedPath = writeTempJSON(TestUtils::combinedWithInlinePSFJSON(), "combined_psf_add.json");
    fe.loadJSONBundle(combinedPath);

    EXPECT_EQ(fe.jsonBundleRef().psfConfigs.size(), 1);

    auto psfPath = writeTempJSON(TestUtils::standaloneSinglePSFConfigJSON(), "psf_add.json");
    auto newConfigs = fe.loadPSFConfigsFromFile(psfPath);
    fe.jsonBundleRef().psfConfigs.insert(
        fe.jsonBundleRef().psfConfigs.end(),
        newConfigs.begin(), newConfigs.end());

    auto& jsonBundle = fe.jsonBundleRef();
    ASSERT_EQ(jsonBundle.psfConfigs.size(), 2);
    EXPECT_EQ(jsonBundle.psfConfigs[0]->ID, "inline_gauss");
    EXPECT_EQ(jsonBundle.psfConfigs[1]->ID, "standalone_gauss");
}

TEST_F(CLIFrontendTest, SeparateConfigs_AllSeparate_Deconvolution) {
    auto fe = makeFrontend();

    auto setupPath = writeTempJSON(TestUtils::standaloneSetupConfigJSON(), "allsep_setup.json");
    fe.loadSetupConfigFromFile(setupPath, fe.jsonBundleRef().setupConfig);
    fe.jsonBundleRef().hasSetup = true;

    auto deconvPath = writeTempJSON(TestUtils::standaloneDeconvConfigJSON(), "allsep_deconv.json");
    fe.loadDeconvConfigFromFile(deconvPath, fe.jsonBundleRef().deconvConfig);
    fe.jsonBundleRef().hasDeconv = true;

    auto psfPath = writeTempJSON(TestUtils::standaloneSinglePSFConfigJSON(), "allsep_psf.json");
    auto configs = fe.loadPSFConfigsFromFile(psfPath);
    fe.jsonBundleRef().psfConfigs = configs;
    fe.jsonBundleRef().hasPSF = true;

    auto& jsonBundle = fe.jsonBundleRef();
    EXPECT_EQ(jsonBundle.setupConfig.imagePath, "standalone_input.tif");
    EXPECT_EQ(jsonBundle.deconvConfig.algorithmName, "RegularizedInverseFilter");
    ASSERT_EQ(jsonBundle.psfConfigs.size(), 1);
    EXPECT_EQ(jsonBundle.psfConfigs[0]->ID, "standalone_gauss");

    fe.cliBundleRef().hasSetup = true;
    fe.cliBundleRef().hasDeconv = true;

    ConfigBundle merged = fe.mergeBundles(fe.jsonBundleRef(), fe.cliBundleRef());
    EXPECT_EQ(merged.setupConfig.imagePath, "standalone_input.tif");
    EXPECT_EQ(merged.deconvConfig.algorithmName, "RegularizedInverseFilter");
    ASSERT_EQ(merged.psfConfigs.size(), 1);

    DeconvolutionRequest request = fe.generateDeconvRequest(merged);
    EXPECT_EQ(request.getConfig()->imagePath, "standalone_input.tif");
    EXPECT_EQ(request.getDeconvolutionConfig()->algorithmName, "RegularizedInverseFilter");
    EXPECT_TRUE(request.hasInlinePSFConfigs());
}

TEST_F(CLIFrontendTest, SeparateConfigs_PSFGenerator_SetupAndPSF) {
    auto fe = makeFrontend();

    auto setupPath = writeTempJSON(TestUtils::standaloneSetupConfigJSON(), "psfgen_setup.json");
    fe.loadSetupConfigFromFile(setupPath, fe.jsonPsfBundleRef().setupConfig);
    fe.jsonPsfBundleRef().hasSetup = true;

    auto psfPath = writeTempJSON(TestUtils::standaloneSinglePSFConfigJSON(), "psfgen_psf.json");
    auto configs = fe.loadPSFConfigsFromFile(psfPath);
    fe.jsonPsfBundleRef().psfConfigs = configs;
    fe.jsonPsfBundleRef().hasPSF = true;

    auto& bundle = fe.jsonPsfBundleRef();
    EXPECT_EQ(bundle.setupConfig.outputPath, "standalone_output.tif");
    EXPECT_EQ(bundle.setupConfig.backend, "cuda");
    ASSERT_EQ(bundle.psfConfigs.size(), 1);
    EXPECT_EQ(bundle.psfConfigs[0]->ID, "standalone_gauss");

    fe.cliPsfBundleRef().hasSetup = true;

    PSFConfigBundle merged = fe.mergePSFBundles(fe.jsonPsfBundleRef(), fe.cliPsfBundleRef());
    EXPECT_EQ(merged.setupConfig.outputPath, "standalone_output.tif");
    ASSERT_EQ(merged.psfConfigs.size(), 1);

    PSFGenerationRequest request = fe.generatePSFRequest(merged);
    EXPECT_EQ(request.getConfig()->outputPath, "standalone_output.tif");
    EXPECT_TRUE(request.hasInlinePSFConfigs());
    EXPECT_EQ(request.getInlinePSFConfigs()[0]->ID, "standalone_gauss");
}
