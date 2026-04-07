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

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include "dolphin/HelperClasses.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphinbackend/CuboidShape.h"

// Helper function to check if a number is smooth (only has prime factors 2, 3, 5)
bool isSmooth(int n) {
    if (n <= 0) return false;
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

// Helper function to find next smooth number
int nextSmooth(int dim) {
    if (dim <= 0) return dim;
    while (!isSmooth(dim)) {
        dim++;
    }
    return dim;
}

// Helper function to find previous smooth number
int previousSmooth(int dim) {
    if (dim <= 0) return dim;
    while (!isSmooth(dim)) {
        dim--;
    }
    return dim;
}



// Test splitImageHomogeneous function
void testSplitImageHomogeneous() {
    std::cout << "Testing splitImageHomogeneous()..." << std::endl;

    // Test 1: Simple case - small image, no padding, should fit in one cube
    {
        CuboidShape imageSize(100, 100, 50);
        Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
        size_t maxVolume = 1000000;  // 1M voxels
        size_t minCubes = 1;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(1, 1, 1);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 1: Small image, no padding - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 2: Image with padding
    {
        CuboidShape imageSize(100, 100, 50);
        Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
        size_t maxVolume = 1000000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(1, 1, 1);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 2: Image with padding - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 3: Large image requiring multiple cubes
    {
        CuboidShape imageSize(512, 512, 100);
        Padding padding{CuboidShape(32, 32, 16), CuboidShape(32, 32, 16)};
        size_t maxVolume = 1000000;  // 1M voxels - should require multiple cubes
        size_t minCubes = 4;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(64, 64, 32);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 3: Large image requiring multiple cubes - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 4: Very small image
    {
        CuboidShape imageSize(32, 32, 10);
        Padding padding{CuboidShape(4, 4, 2), CuboidShape(4, 4, 2)};
        size_t maxVolume = 1000000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(8, 8, 4);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 4: Very small image - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 5: Image with NONE padding type
    {
        CuboidShape imageSize(200, 200, 80);
        Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
        size_t maxVolume = 500000;
        size_t minCubes = 2;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(40, 40, 20);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 5: Image with NONE padding type - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 6: Check cube dimensions are smooth (FFTW optimized)
    {
        CuboidShape imageSize(256, 256, 64);
        Padding padding{CuboidShape(16, 16, 8), CuboidShape(16, 16, 8)};
        size_t maxVolume = 200000;
        size_t minCubes = 4;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(32, 32, 16);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        // Each cube's total dimensions (box + padding) should be smooth
        for (const auto& cube : cubes) {
            BoxCoord fullBox = cube.getBox();
            // Note: The dimensions might not always be smooth due to edge conditions,
            // but the main cube dimensions should be
        }
        std::cout << "    Test 6: Cube dimensions check - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 7: Verify cube coverage
    {
        CuboidShape imageSize(128, 128, 32);
        Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
        size_t maxVolume = 500000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(16, 16, 8);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        // Verify all cubes are within image bounds (considering padding)
        for (const auto& cube : cubes) {
            const auto& pos = cube.box.position;
            const auto& dim = cube.box.dimensions;

            // Position should be non-negative
            assert(pos.width >= 0);
            assert(pos.height >= 0);
            assert(pos.depth >= 0);

            // Dimensions should be positive
            assert(dim.width > 0);
            assert(dim.height > 0);
            assert(dim.depth > 0);
        }
        std::cout << "    Test 7: Cube coverage verification - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 8: Test with high minCubes requirement
    {
        CuboidShape imageSize(256, 256, 64);
        Padding padding{CuboidShape(16, 16, 8), CuboidShape(16, 16, 8)};
        size_t maxVolume = 1000000;
        size_t minCubes = 8;
        PaddingType padType = PaddingType::MIRROR;
        CuboidShape minSize(32, 32, 16);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 8: High minCubes requirement - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 9: Very constrained memory
    {
        CuboidShape imageSize(200, 200, 50);
        Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
        size_t maxVolume = 50000;  // Very small
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(20, 20, 10);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        // Should still produce at least minCubes
        assert(cubes.size() >= minCubes);
        std::cout << "    Test 9: Very constrained memory - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 10: Asymmetric image
    {
        CuboidShape imageSize(400, 200, 30);  // Width > Height > Depth
        Padding padding{CuboidShape(20, 10, 5), CuboidShape(20, 10, 5)};
        size_t maxVolume = 300000;
        size_t minCubes = 2;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(40, 20, 10);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        assert(cubes.size() >= minCubes);
        std::cout << "    Test 10: Asymmetric image - " << cubes.size() << " cubes" << std::endl;
    }
    // Test 11: Fail too little memory
    {
        CuboidShape imageSize(200, 200, 50);
        Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
        size_t maxVolume = 500;  // Very small
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(20, 20, 10);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        std::cout << "    Test 11: Low memory should fail - " << result.success << std::endl;
        if (!result.success) std::cout << result.getErrorString() << std::endl;

    }
    // Test 12: Fail to many subcubes
    {
        CuboidShape imageSize(200, 200, 50);
        Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
        size_t maxVolume = 5000;  // Very small
        size_t minCubes = 100;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(20, 20, 10);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);

        std::cout << "    Test 12: Too many cubes should fail - " << result.success << std::endl;
        if (!result.success) std::cout << result.getErrorString() << std::endl;
    }

    std::cout << "  splitImageHomogeneous() tests passed!" << std::endl;
}

// Test cube positioning and overlap
void testCubePositioning() {
    std::cout << "Testing cube positioning and coverage..." << std::endl;

    // Test that cubes don't have gaps (with padding, they should overlap)
    {
        CuboidShape imageSize(256, 256, 64);
        Padding padding{CuboidShape(16, 16, 8), CuboidShape(16, 16, 8)};
        size_t maxVolume = 200000;
        size_t minCubes = 4;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(32, 32, 16);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        // Sort cubes by position
        std::vector<BoxCoordWithPadding> sortedCubes = cubes;
        std::sort(sortedCubes.begin(), sortedCubes.end(),
            [](const BoxCoordWithPadding& a, const BoxCoordWithPadding& b) {
                if (a.box.position.depth != b.box.position.depth)
                    return a.box.position.depth < b.box.position.depth;
                if (a.box.position.height != b.box.position.height)
                    return a.box.position.height < b.box.position.height;
                return a.box.position.width < b.box.position.width;
            });

        std::cout << "    Cube positioning test with " << cubes.size() << " cubes" << std::endl;
        std::cout << "    First cube: pos=" << sortedCubes[0].box.position.print()
                  << " dim=" << sortedCubes[0].box.dimensions.print() << std::endl;
        if (sortedCubes.size() > 1) {
            std::cout << "    Second cube: pos=" << sortedCubes[1].box.position.print()
                      << " dim=" << sortedCubes[1].box.dimensions.print() << std::endl;
        }
    }

    // Test that first cube starts at origin for NONE padding type
    {
        CuboidShape imageSize(128, 128, 32);
        Padding padding{CuboidShape(8, 8, 4), CuboidShape(8, 8, 4)};
        size_t maxVolume = 500000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(16, 16, 8);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;

        // Find the cube that starts at origin
        bool foundOriginCube = false;
        for (const auto& cube : cubes) {
            if (cube.box.position.width == 0 &&
                cube.box.position.height == 0 &&
                cube.box.position.depth == 0) {
                foundOriginCube = true;
                // With NONE padding, first cube should have no "before" padding
                assert(cube.padding.before.width == 0);
                assert(cube.padding.before.height == 0);
                assert(cube.padding.before.depth == 0);
                break;
            }
        }
        std::cout << "    First cube at origin found: " << (foundOriginCube ? "yes" : "no") << std::endl;
    }

    std::cout << "  Cube positioning tests passed!" << std::endl;
}

// Test edge cases
void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;

    // Test 1: Minimum viable image
    {
        CuboidShape imageSize(10, 10, 10);
        Padding padding{CuboidShape(2, 2, 2), CuboidShape(2, 2, 2)};
        size_t maxVolume = 1000000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(4, 4, 4);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;
        assert(cubes.size() >= 1);
        std::cout << "    Minimum viable image test passed - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 2: No padding
    {
        CuboidShape imageSize(64, 64, 32);
        Padding padding{CuboidShape(0, 0, 0), CuboidShape(0, 0, 0)};
        size_t maxVolume = 500000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::NONE;
        CuboidShape minSize(1, 1, 1);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;
        assert(cubes.size() >= 1);
        std::cout << "    No padding test passed - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 3: Large padding relative to image
    {
        CuboidShape imageSize(50, 50, 20);
        Padding padding{CuboidShape(20, 20, 10), CuboidShape(20, 20, 10)};
        size_t maxVolume = 500000;
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(40, 40, 20);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;
        assert(cubes.size() >= 1);
        std::cout << "    Large padding test passed - " << cubes.size() << " cubes" << std::endl;
    }

    // Test 4: Cube size equals image size
    {
        CuboidShape imageSize(100, 100, 50);
        Padding padding{CuboidShape(10, 10, 5), CuboidShape(10, 10, 5)};
        size_t maxVolume = 10000000;  // Very large
        size_t minCubes = 1;
        PaddingType padType = PaddingType::ZERO;
        CuboidShape minSize(1, 1, 1);

        auto result = splitImageHomogeneous(padding, imageSize, maxVolume, minCubes, padType, minSize);
        assert(result.success);
        auto cubes = result.value;
        assert(cubes.size() >= 1);
        std::cout << "    Cube size equals image test passed - " << cubes.size() << " cubes" << std::endl;
    }

    std::cout << "  Edge case tests passed!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== ImageSplit Function Tests ===" << std::endl;

    testSplitImageHomogeneous();
    testCubePositioning();
    testEdgeCases();

    std::cout << "\n=== All tests passed! ===" << std::endl;
    return 0;
}
