#include <gtest/gtest.h>
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/Logging.h"

class AlgorithmFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

TEST_F(AlgorithmFactoryTest, GetAvailableAlgorithms) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algorithms = factory.getAvailableAlgorithms();
    EXPECT_GE(algorithms.size(), 7u);
}

TEST_F(AlgorithmFactoryTest, HasRichardsonLucy) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("RichardsonLucy"));
}

TEST_F(AlgorithmFactoryTest, HasRLTV) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("RichardsonLucyTotalVariation"));
}

TEST_F(AlgorithmFactoryTest, HasRegularizedInverseFilter) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("RegularizedInverseFilter"));
}

TEST_F(AlgorithmFactoryTest, HasInverseFilter) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("InverseFilter"));
}

TEST_F(AlgorithmFactoryTest, HasConvolution) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("Convolution"));
}

TEST_F(AlgorithmFactoryTest, HasRLAD) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("RichardsonLucywithAdaptiveDamping"));
}

TEST_F(AlgorithmFactoryTest, HasTestAlgorithm) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_TRUE(factory.isAlgorithmAvailable("TestAlgorithm"));
}

TEST_F(AlgorithmFactoryTest, IsAlgorithmAvailableFalse) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_FALSE(factory.isAlgorithmAvailable("NonExistentAlgorithm"));
}

TEST_F(AlgorithmFactoryTest, IsAlgorithmAvailableCaseSensitive) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_FALSE(factory.isAlgorithmAvailable("richardsonlucy"));
    EXPECT_FALSE(factory.isAlgorithmAvailable("RICHARDSONLUCY"));
}

TEST_F(AlgorithmFactoryTest, CreateRichardsonLucy) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRLTV) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucyTotalVariation";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateInverseFilter) {
    DeconvolutionConfig config;
    config.algorithmName = "InverseFilter";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateConvolution) {
    DeconvolutionConfig config;
    config.algorithmName = "Convolution";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRLAD) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucywithAdaptiveDamping";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateUnique) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createUnique(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRaw) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.create(config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateUnknownThrows) {
    DeconvolutionConfig config;
    config.algorithmName = "NonExistent";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_THROW(factory.createShared(config), std::runtime_error);
}

TEST_F(AlgorithmFactoryTest, AlgorithmConfigureAndMemoryMultiplier) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();

    struct AlgoSpec {
        std::string name;
        size_t expectedMemoryMultiplier;
    };

    std::vector<AlgoSpec> specs = {
        {"RichardsonLucy", 2},
        {"RichardsonLucyTotalVariation", 6},
        {"InverseFilter", 0},
        {"Convolution", 0},
        {"RichardsonLucywithAdaptiveDamping", 1},
    };

    for (const auto& spec : specs) {
        DeconvolutionConfig config;
        config.algorithmName = spec.name;
        auto algo = factory.createShared(config);
        ASSERT_NE(algo, nullptr) << "Failed for algorithm: " << spec.name;
        EXPECT_EQ(algo->getMemoryMultiplier(), spec.expectedMemoryMultiplier)
            << "Wrong memory multiplier for: " << spec.name;
    }
}

TEST_F(AlgorithmFactoryTest, AlgorithmClone) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);

    auto clone = algo->clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getMemoryMultiplier(), algo->getMemoryMultiplier());
}

TEST_F(AlgorithmFactoryTest, AlgorithmIsNotInitializedByDefault) {
    DeconvolutionConfig config;
    config.algorithmName = "RichardsonLucy";
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(config);
    ASSERT_NE(algo, nullptr);
    EXPECT_FALSE(algo->isInitialized());
}
