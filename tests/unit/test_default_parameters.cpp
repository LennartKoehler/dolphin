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
#include "dolphin/DefaultParameters.h"
#include "dolphin/Logging.h"

class DefaultParametersTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logging::init();
    }
};

// The singleton must expose the shipped default values, whether loaded from
// the configs_template copy shipped next to the binary or from the identical
// built-in fallback.
TEST_F(DefaultParametersTest, ExposesShippedDefaultValues) {
    const auto& params = DefaultParameters::getDefaultParameters();
    EXPECT_FLOAT_EQ(params.lowPaddingThreshold(), 1e-3f);
    EXPECT_FLOAT_EQ(params.mediumPaddingThreshold(), 1e-4f);
    EXPECT_FLOAT_EQ(params.generationThresholdMultiplier(), 1e-2f);
}

TEST_F(DefaultParametersTest, SingletonReturnsSameInstance) {
    const auto& first = DefaultParameters::getDefaultParameters();
    const auto& second = DefaultParameters::getDefaultParameters();
    EXPECT_EQ(&first, &second);
}
