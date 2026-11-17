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

#include <gtest/gtest.h>
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEConfig.h"
#include "dolphin/deconvolution/configs/PoissonMLEUnitStepConfig.h"
#include "dolphin/deconvolution/algorithms/PoissonMLEDeconvolutionAlgorithm.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/Logging.h"

class AlgorithmFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }

    // Config classes are polymorphic: build the one registered for `algorithm`.
    static std::shared_ptr<DeconvolutionConfig> makeConfig(const std::string& algorithm) {
        return DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm(algorithm);
    }
};

TEST_F(AlgorithmFactoryTest, CreateConfigForEachAlgorithm) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    for (const auto& name : factory.getAvailableAlgorithms()) {
        auto config = factory.createConfigForAlgorithm(name);
        ASSERT_NE(config, nullptr) << "No config for " << name;
        EXPECT_EQ(config->algorithmName, name);
    }
}

TEST_F(AlgorithmFactoryTest, CreateConfigRejectsUnknownAlgorithm) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    EXPECT_THROW(factory.createConfigForAlgorithm("NonExistent"), std::runtime_error);
}

TEST_F(AlgorithmFactoryTest, CreateConfigFromJSONSelectsAlgorithm) {
    json data = {{"algorithm_name", "PoissonMLE"}, {"iterations", 7}, {"poisson_sieve", false}};
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto config = factory.createConfigFromJSON(data);
    ASSERT_NE(config, nullptr);
    EXPECT_EQ(config->algorithmName, "PoissonMLE");
    EXPECT_EQ(config->iterations, 7);

    auto* poisson = dynamic_cast<PoissonMLEConfig*>(config.get());
    ASSERT_NE(poisson, nullptr);
    EXPECT_FALSE(poisson->poissonSieve);
}

TEST_F(AlgorithmFactoryTest, CreateConfigForKnownAlgorithmReturnsFinishedConfig) {
    // Mirrors PSFGeneratorFactory::createConfig: name + json -> fully loaded config.
    json data = {{"iterations", 9}, {"poisson_lambda", 0.25f}};
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();

    auto finished = factory.createConfigForAlgorithm("PoissonMLE", data);
    ASSERT_NE(finished, nullptr);
    EXPECT_EQ(finished->algorithmName, "PoissonMLE");
    EXPECT_EQ(finished->iterations, 9);
    auto* poisson = dynamic_cast<PoissonMLEConfig*>(finished.get());
    ASSERT_NE(poisson, nullptr);
    EXPECT_FLOAT_EQ(poisson->poissonLambda, 0.25f);

    // The no-data overload stays empty (CLI binds values later).
    auto empty = factory.createConfigForAlgorithm("PoissonMLE");
    EXPECT_EQ(empty->iterations, 10); // default
}

TEST_F(AlgorithmFactoryTest, PoissonMLEConfigRejectsOtherAlgorithmConfig) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto rlConfig = factory.createConfigForAlgorithm("RichardsonLucy");

    // Bypass the factory's name lookup to exercise the type check.
    PoissonMLEDeconvolutionAlgorithm algorithm;
    EXPECT_THROW(algorithm.configure(*rlConfig), std::runtime_error);
}

TEST_F(AlgorithmFactoryTest, GetAvailableAlgorithms) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algorithms = factory.getAvailableAlgorithms();
    EXPECT_GE(algorithms.size(), 6u);
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
    auto config = makeConfig("RichardsonLucy");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRLTV) {
    auto config = makeConfig("RichardsonLucyTotalVariation");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateInverseFilter) {
    auto config = makeConfig("InverseFilter");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateConvolution) {
    auto config = makeConfig("Convolution");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRLAD) {
    auto config = makeConfig("RichardsonLucywithAdaptiveDamping");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreatePoissonMLE) {
    auto config = makeConfig("PoissonMLE");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateUnique) {
    auto config = makeConfig("RichardsonLucy");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createUnique(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateRaw) {
    auto config = makeConfig("RichardsonLucy");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.create(*config);
    ASSERT_NE(algo, nullptr);
}

TEST_F(AlgorithmFactoryTest, CreateUnknownThrows) {
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    // No config can be built for an unknown algorithm, so no algorithm exists either.
    EXPECT_THROW(factory.createConfigForAlgorithm("NonExistent"), std::runtime_error);
}

// TEST_F(AlgorithmFactoryTest, AlgorithmConfigureAndMemoryMultiplier) {
//     auto& factory = DeconvolutionAlgorithmFactory::getInstance();
//
//     struct AlgoSpec {
//         std::string name;
//         size_t expectedMemoryMultiplier;
//     };
//
//     std::vector<AlgoSpec> specs = {
//         {"RichardsonLucy", 1},
//         {"RichardsonLucyTotalVariation", 5},
//         {"InverseFilter", 0},
//         {"Convolution", 0},
//         {"RichardsonLucywithAdaptiveDamping", 1},
//         {"PoissonMLE", 6},
//     };
//
//     for (const auto& spec : specs) {
//         DeconvolutionConfig config;
//         config.algorithmName = spec.name;
//         auto algo = factory.createShared(*config);
//         ASSERT_NE(algo, nullptr) << "Failed for algorithm: " << spec.name;
//         EXPECT_EQ(algo->getMemoryMultiplier(), spec.expectedMemoryMultiplier)
//             << "Wrong memory multiplier for: " << spec.name;
//     }
// }

TEST_F(AlgorithmFactoryTest, AlgorithmClone) {
    auto config = makeConfig("RichardsonLucy");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);

    auto clone = algo->clone();
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->getMemoryMultiplier(), algo->getMemoryMultiplier());
}

TEST_F(AlgorithmFactoryTest, AlgorithmIsNotInitializedByDefault) {
    auto config = makeConfig("RichardsonLucy");
    auto& factory = DeconvolutionAlgorithmFactory::getInstance();
    auto algo = factory.createShared(*config);
    ASSERT_NE(algo, nullptr);
    EXPECT_FALSE(algo->isInitialized());
}

// =============================================================
// End-to-end: CLI values land in the algorithm's own config class
// =============================================================

TEST_F(AlgorithmFactoryTest, PoissonMLEConfigFromFullJSON) {
    json data = {
        {"algorithm_name", "PoissonMLE"},
        {"iterations", 30},
        {"signal_to_noise", 0.002f},
        {"poisson_lambda", 0.5f},
        {"poisson_background", 12.0f},
        {"poisson_prefilter", false},
        {"poisson_prefilter_scale", 0.4f},
        {"poisson_sieve", false},
        {"poisson_prior", true},
        {"poisson_sieve_passes", 3},
        {"poisson_sieve_scale", 0.2f},
    };

    auto config = DeconvolutionConfig::createFromJSON(data);
    auto* poisson = dynamic_cast<PoissonMLEConfig*>(config.get());
    ASSERT_NE(poisson, nullptr);
    EXPECT_EQ(poisson->iterations, 30);
    EXPECT_FLOAT_EQ(poisson->poissonLambda, 0.5f);
    EXPECT_FLOAT_EQ(poisson->poissonBackground, 12.0f);
    EXPECT_FALSE(poisson->poissonPrefilter);
    EXPECT_FLOAT_EQ(poisson->poissonPrefilterScale, 0.4f);
    EXPECT_FALSE(poisson->poissonSieve);
    EXPECT_TRUE(poisson->poissonPrior);
    EXPECT_EQ(poisson->poissonSievePasses, 3);
    EXPECT_FLOAT_EQ(poisson->poissonSieveScale, 0.2f);

    // The algorithm accepts exactly this config type.
    auto algorithm = DeconvolutionAlgorithmFactory::getInstance().createShared(*config);
    ASSERT_NE(algorithm, nullptr);
}

TEST_F(AlgorithmFactoryTest, UnitStepConfigIsAcceptedByPoissonMLEAlgorithm) {
    auto config = DeconvolutionAlgorithmFactory::getInstance().createConfigForAlgorithm("PoissonMLEUnitStep");
    auto* unitStep = dynamic_cast<PoissonMLEUnitStepConfig*>(config.get());
    ASSERT_NE(unitStep, nullptr);
    // UnitStep derives from PoissonMLE, so the base algorithm accepts it too.
    PoissonMLEDeconvolutionAlgorithm algorithm;
    EXPECT_NO_THROW(algorithm.configure(*config));
}
